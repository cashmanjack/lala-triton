"""Sustained-state measurement: CUDA-graph replay of back-to-back GEMMs over rotating DRAM-cold buffers,
with an NVML side-thread sampler (20 ms) and energy-counter deltas."""
import math, threading, time, numpy as np, torch, pynvml as n
from nvml_ext import handle, energy_mJ, throttle, SW_POWER, HW_SLOW


class Sampler(threading.Thread):
    def __init__(self, h, period=0.02):
        super().__init__(daemon=True); self.h, self.p, self.stop_ev, self.rows = h, period, threading.Event(), []
    def run(self):
        while not self.stop_ev.is_set():
            t = time.time()
            try:
                self.rows.append((t, n.nvmlDeviceGetPowerUsage(self.h) / 1e3, n.nvmlDeviceGetClockInfo(self.h, n.NVML_CLOCK_SM),
                                  n.nvmlDeviceGetClockInfo(self.h, n.NVML_CLOCK_MEM), n.nvmlDeviceGetTemperature(self.h, 0), throttle(self.h)))
            except n.NVMLError: pass
            time.sleep(max(0, self.p - (time.time() - t)))
    def stop(self): self.stop_ev.set(); self.join()


def make_graph(launch_fns, target_ms_per_replay=30.0, est_ms=None):
    """launch_fns: list of zero-arg callables (one per buffer set). Captures a graph of L launches cycling buffers."""
    for f in launch_fns: f()          # compile/warm
    torch.cuda.synchronize()
    if est_ms is None:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(10): launch_fns[i % len(launch_fns)]()
        e.record(); torch.cuda.synchronize(); est_ms = s.elapsed_time(e) / 10
    L = max(len(launch_fns), int(math.ceil(target_ms_per_replay / max(est_ms, 1e-3))))
    L = int(math.ceil(L / len(launch_fns)) * len(launch_fns))
    g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        launch_fns[0](); torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=st):
            for i in range(L): launch_fns[i % len(launch_fns)]()
    torch.cuda.synchronize()
    return g, L, est_ms


def measure(g, L, gpu_index, warmup_s=1.0, window_s=3.0, cooldown_s=2.0):
    h = handle(gpu_index)
    time.sleep(cooldown_s)
    temp0 = n.nvmlDeviceGetTemperature(h, 0)
    smp = Sampler(h); smp.start()
    t_start = time.time()
    while time.time() - t_start < warmup_s:
        g.replay(); torch.cuda.synchronize()
    e0 = energy_mJ(h); ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    tw = time.time(); ev0.record(); reps = 0
    while True:
        g.replay(); reps += 1
        if reps % 2 == 0:
            torch.cuda.synchronize()
            if time.time() - tw >= window_s: break
    ev1.record(); torch.cuda.synchronize(); tw1 = time.time()
    e1 = energy_mJ(h); smp.stop()
    iters = reps * L
    S = np.array(smp.rows)
    win = S[(S[:, 0] >= tw) & (S[:, 0] <= tw1)]
    allw = S[S[:, 0] >= t_start]
    # settle time: first time (from warmup start) after which the 200ms-rolling SM clock stays within 5% of the window median
    med = np.median(win[:, 2]); clk = allw[:, 2]; ts = allw[:, 0] - t_start
    roll = np.convolve(clk, np.ones(10) / 10, mode='same'); bad = np.where(np.abs(roll - med) > 0.05 * med)[0]
    bad = bad[(bad >= 5) & (bad < len(roll) - 5)]
    settle = float(ts[bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(ts) else 0.0
    thr = win[:, 5].astype(np.int64)
    return dict(ms_per_gemm=ev0.elapsed_time(ev1) / iters, iters=iters, sm_clk_med=float(med), mem_clk_med=float(np.median(win[:, 3])),
                power_mean=float(win[:, 1].mean()), energy_J_per_gemm=(e1 - e0) / 1e3 / iters, energy_power_W=(e1 - e0) / 1e3 / (tw1 - tw),
                temp0=temp0, temp_end=float(win[-1, 4]), frac_sw_power_cap=float(((thr & SW_POWER) > 0).mean()),
                frac_hw_slow=float(((thr & HW_SLOW) > 0).mean()), settle_s=settle, n_samples=len(win), ts=time.time()), allw


def buffer_sets(M, N, K, dtype, layout, make_inputs, max_bytes=256 * 2**20):
    s = 1 if dtype == 'fp8' else 2
    per = (M * K + K * N) * s + M * N * 2
    nset = max(1, math.ceil(max_bytes / per))
    sets = []
    for i in range(nset):
        sets.append(make_inputs(M, N, K, dtype, seed=i, layout=layout))
    return sets
