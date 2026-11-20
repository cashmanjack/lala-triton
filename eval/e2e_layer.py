"""Llama 3 8B decoder layer under torch.compile, bf16: stock max-autotune (ATEN,TRITON) vs lala_triton top-1 (TRITON) vs default.
Each method runs in its own process with FRESH caches (driver mode) so total compile/tuning time is honest:
   python3 eval/e2e_layer.py --profile ... --gpu 0 --out results/e2e.json            # driver: runs the 3 methods
   python3 eval/e2e_layer.py --worker ours1 ...                                      # (internal)
Layer = RMSNorm, fused QKV linear, GQA attention (SDPA, causal for prefill; KV cache length 1024 for decode), O linear, residual, RMSNorm,
fused gate_up, SiLU*mul, down linear, residual. RoPE and KV-cache update are omitted (not GEMM-relevant; stated in REPORT_5).
Cases: prefill 512 / 2048 tokens (batch 1), decode batch 1 / 8 / 32. Latency = sustained protocol (CUDA-graph replay, NVML clocks)."""
import argparse, json, os, shutil, subprocess, sys, time, re, logging
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
CASES = [('prefill', 1, 512), ('prefill', 1, 2048), ('decode', 1, 1), ('decode', 8, 1), ('decode', 32, 1)]   # (kind, batch, new tokens)
KV_LEN = 1024
ap = argparse.ArgumentParser(); ap.add_argument('--profile', required=True); ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--out', default='results/e2e.json'); ap.add_argument('--worker'); ap.add_argument('--window', default='0.5,3.0,1.0')
a = ap.parse_args()

if not a.worker:
    res = {}
    for me in ('ind_auto', 'ours1', 'default'):
        tmp = f'/var/tmp/lala_triton_e2e_{me}'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
        env = dict(os.environ, TRITON_CACHE_DIR=f'{tmp}/triton', TORCHINDUCTOR_CACHE_DIR=f'{tmp}/inductor', CUDA_VISIBLE_DEVICES=str(a.gpu))
        p = subprocess.run([sys.executable, __file__, '--profile', a.profile, '--gpu', str(a.gpu), '--worker', me, '--window', a.window], capture_output=True, text=True, env=env)
        line = [l for l in p.stdout.splitlines() if l.startswith('@@')]
        shutil.rmtree(tmp, ignore_errors=True)
        if not line: print('FAIL', me, p.stderr[-800:]); continue
        res[me] = json.loads(line[0][2:])
        res[me]['bench_s_total'] = sum(float(v) for v in re.findall(r'benchmarking takes ([\d.]+) seconds', p.stderr))   # Inductor logs to stderr
        res[me]['compile_s_total'] = res[me]['total_tuning_s'] - res[me]['bench_s_total'] - res[me]['select_s']
        print(me, json.dumps(res[me])[:300], flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); sys.exit(0)

# ---------------------------------------------------------------- worker
import torch, torch.nn.functional as F
import torch._inductor.config as ic
from lala_triton import inductor
from lala_triton._sustained import make_graph, measure
from lala_triton._nvml import handle, energy_mJ
_h = handle(0); time.sleep(1.0); _i0, _it0 = energy_mJ(_h), time.time(); time.sleep(2.0); IDLE_W = (energy_mJ(_h) - _i0) / 1e3 / (time.time() - _it0)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
H, I, NH, NKV, HD = 4096, 14336, 32, 8, 128
dev, dt = 'cuda', torch.bfloat16
g = torch.Generator(device=dev).manual_seed(0)
W = dict(qkv=torch.randn((NH + 2 * NKV) * HD, H, device=dev, generator=g).to(dt) * 0.02, o=torch.randn(H, H, device=dev, generator=g).to(dt) * 0.02,
         gu=torch.randn(2 * I, H, device=dev, generator=g).to(dt) * 0.02, down=torch.randn(H, I, device=dev, generator=g).to(dt) * 0.02,
         n1=torch.ones(H, device=dev, dtype=dt), n2=torch.ones(H, device=dev, dtype=dt))
def rms(x, w): return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(dt) * w
def layer(x, kc, vc, causal: bool):
    B, T, _ = x.shape
    qkv = F.linear(rms(x, W['n1']), W['qkv']); q, k, v = qkv.split([NH * HD, NKV * HD, NKV * HD], -1)
    q = q.view(B, T, NH, HD).transpose(1, 2); k = k.view(B, T, NKV, HD).transpose(1, 2); v = v.view(B, T, NKV, HD).transpose(1, 2)
    if not causal: k, v = kc, vc
    k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
    att = F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2).reshape(B, T, H)
    x = x + F.linear(att, W['o'])
    gu = F.linear(rms(x, W['n2']), W['gu']); gate, up = gu.chunk(2, -1)
    return x + F.linear(F.silu(gate) * up, W['down'])

me = a.worker; mode = 'default' if me == 'default' else 'max-autotune-no-cudagraphs'
ic.max_autotune_gemm_backends = {'default': 'ATEN', 'ind_auto': 'ATEN,TRITON', 'ours1': 'TRITON'}[me]; ic.fx_graph_cache = False
if me == 'ours1': inductor.enable(a.profile, k=1, bf16_layout='ABT')
autolog = []; h = logging.Handler(); h.emit = lambda r: autolog.append(r.getMessage())
for n in ('torch._inductor.select_algorithm', 'torch._inductor.select_algorithm.autotuning'): lg = logging.getLogger(n); lg.addHandler(h); lg.setLevel(logging.DEBUG)
win = tuple(map(float, a.window.split(',')))
out = dict(method=me, cases={}, torch=torch.__version__)
t_all = time.time()
for kind, B, T in CASES:
    x = torch.randn(B, T, H, device=dev, generator=g).to(dt)
    kc = torch.randn(B, NKV, KV_LEN, HD, device=dev, generator=g).to(dt); vc = torch.randn_like(kc)
    torch._dynamo.reset()
    def fn(x, kc, vc, causal=(kind == 'prefill')): return layer(x, kc, vc, causal)
    f = torch.compile(fn, mode=mode)
    n0 = len(autolog); e0 = energy_mJ(_h); t0 = time.time(); y = f(x, kc, vc); torch.cuda.synchronize(); first = time.time() - t0; tune_J = (energy_mJ(_h) - e0) / 1e3
    bench = sum(float(v) for v in re.findall(r'benchmarking takes ([\d.]+) seconds', ' '.join(autolog[n0:])))
    gr, L, _ = make_graph([lambda: f(x, kc, vc)])
    res, _ = measure(gr, L, a.gpu, *win); del gr
    out['cases'][f'{kind}_b{B}_t{T}'] = dict(first_call_s=first, bench_s=bench, compile_s=first - bench, tune_energy_J=tune_J, tune_excess_J=tune_J - IDLE_W * first, ms=res['ms_per_gemm'], energy_J_per_layer=res['energy_J_per_gemm'], sm_clk=res['sm_clk_med'], power=res['power_mean'],
                                            finite=bool(torch.isfinite(y).all()), checksum=float(y.float().abs().mean()))
out['total_tuning_s'] = sum(c['first_call_s'] for c in out['cases'].values()); out['total_bench_s'] = sum(c['bench_s'] for c in out['cases'].values())
out['idle_W'] = IDLE_W; out['total_tune_energy_J'] = sum(c['tune_energy_J'] for c in out['cases'].values()); out['total_tune_excess_J'] = sum(c['tune_excess_J'] for c in out['cases'].values())
out['select_s'] = inductor.STATE['select_s']; out['wall_s'] = time.time() - t_all
print('@@' + json.dumps(out))
