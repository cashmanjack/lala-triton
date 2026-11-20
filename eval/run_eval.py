"""Evaluation harness.  python eval/run_eval.py --profile profiles/rtx6000ada.json --models llama3-8b,llama3-70b --dtypes bf16,fp8 --out results/ [--gpu 0] [--shard 0/1]
Per (model, gemm, dtype, M): sustained-protocol kernel runtime (DRAM-cold rotating buffers, CUDA-graph replay, NVML clocks) of
  ours1 / ours5 (real Inductor patched with lala_triton, TRITON backend only), ind_auto (stock max-autotune ATEN+TRITON), ind_default (no autotune),
  cublas (torch.mm / _scaled_mm, bf16 reduced-precision reduction off), nvmm (if installed), tritonblas (lala_triton.baselines, if present), oracle (subset).
Rows are appended to results/runtime.csv (resumable; 3 interleaved reps per shape, rep order randomized).  Tuning cost: --tuning-time (fresh-process first calls)."""
import argparse, csv, json, os, random, sys, time, logging, re, math, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lala_triton.config import ROOT, TRITON_CACHE

M_GRID = [1, 8, 32, 128, 512, 2048, 8192]
METHODS = ['ours1', 'ours1tb', 'ours1naive', 'ours5', 'ind_auto', 'ind_default', 'cublas', 'nvmm', 'tritonblas', 'oracle']
ORACLE_SUBSET_M = (8, 128, 2048)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', required=True); ap.add_argument('--models', default='llama3-8b'); ap.add_argument('--dtypes', default='bf16,fp8')
    ap.add_argument('--out', default='results'); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--methods', default=','.join(METHODS)); ap.add_argument('--ms', default=','.join(map(str, M_GRID)))
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--window', default='0.5,1.5,1.0')
    ap.add_argument('--gemms', default=''); ap.add_argument('--tag', default='runtime')
    ap.add_argument('--oracle-ms', default=','.join(map(str, ORACLE_SUBSET_M)))
    ap.add_argument('--tp', type=int, default=1); ap.add_argument('--shapes', default=''); ap.add_argument('--first-call'); ap.add_argument('--tuning-time', action='store_true')
    return ap.parse_args()

def setup_env(gpu, tag=''):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ.setdefault('TRITON_CACHE_DIR', TRITON_CACHE)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', f'{ROOT}/.inductor_cache{tag}')

# ------------------------------------------------------------------ launchers
class Bench:
    def __init__(self, prof_path, gpu, layout='ABT'):
        import torch
        from lala_triton.profile import load
        self.torch, self.gpu, self.layout = torch, gpu, layout
        self.prof_path = prof_path; self.prof = load(prof_path)
        self.ONE = torch.tensor(1.0, device='cuda')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.autolog = []; h = logging.Handler(); h.emit = lambda r: self.autolog.append(r.getMessage())
        for n in ('torch._inductor.select_algorithm', 'torch._inductor.select_algorithm.autotuning'): 
            lg = logging.getLogger(n); lg.addHandler(h); lg.setLevel(logging.DEBUG)

    def base(self, dtype):
        t, one = self.torch, self.ONE
        if dtype == 'fp8': return lambda a, b: t._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=t.float16)
        return lambda a, b: t.mm(a, b)

    def compiled(self, dtype, mode, backends, enable_k=None):
        import torch
        import torch._inductor.config as ic
        from lala_triton import inductor
        torch._dynamo.reset()
        ic.max_autotune_gemm_backends = backends
        ic.fx_graph_cache = False
        if enable_k: inductor.enable(self.prof, k=enable_k, bf16_layout=self.layout)
        try: fn = torch.compile(self.base(dtype), mode=mode)
        finally: pass
        return fn, inductor

    def launchers(self, method, sets, dtype, shape_cfg=None):
        """Returns (fns over buffer sets, info). Compilation / autotuning happens here (first call)."""
        import torch
        from lala_triton import model as M3_
        info = {}
        if method == 'cublas': f = self.base(dtype); return [lambda a=a, b=b: f(a, b) for a, b, c in sets], info
        if method == 'oracle': 
            from lala_triton.kernels import launch
            cfg = shape_cfg; info['cfg'] = json.dumps(cfg)
            return [lambda a=a, b=b, c=c: launch(a, b, c, cfg) for a, b, c in sets], info
        if method == 'ours1tb':      # Amendment 5b tie-break selector (flag restored right after compile)
            from lala_triton import select as S
            S.TIEBREAK = True
            fn, ind = self.compiled(dtype, 'max-autotune-no-cudagraphs', 'TRITON', 1)
        elif method == 'ours1naive':   # ablation: naive traffic model (flag restored right after compile)
            M3_.TRAFFIC = 'naive'
            fn, ind = self.compiled(dtype, 'max-autotune-no-cudagraphs', 'TRITON', 1)
        elif method in ('ours1', 'ours5'): fn, ind = self.compiled(dtype, 'max-autotune-no-cudagraphs', 'TRITON', 1 if method == 'ours1' else 5)
        elif method == 'ind_auto': fn, ind = self.compiled(dtype, 'max-autotune-no-cudagraphs', 'ATEN,TRITON')
        elif method == 'ind_default': fn, ind = self.compiled(dtype, 'default', 'ATEN')
        else: raise ValueError(method)
        self.autolog.clear(); t0 = time.time()
        try:
            a, b, _ = sets[0]; fn(a, b); torch.cuda.synchronize()
        finally:
            ind.disable()
            if method == 'ours1tb':
                from lala_triton import select as S; S.TIEBREAK = False
            if method == 'ours1naive': M3_.TRAFFIC = 'm3'
        info['first_call_s'] = time.time() - t0
        info['autotune'] = ' | '.join(self.autolog)[:2000]
        m = re.search(r'BLOCK_M=(\d+).*?BLOCK_N=(\d+).*?BLOCK_K=(\d+)', ' '.join(l for l in self.autolog if 'BLOCK_M' in l))
        if method in ('ours1', 'ours1tb', 'ours1naive') and ind.STATE['log']: info['pick'] = json.dumps(ind.STATE['log'][-1][4])
        return [lambda a=a, b=b, fn=fn: fn(a, b) for a, b, c in sets], info

def oracle_cfg(bench, sh, M, N, K, dtype, gpu, rng_seed=5):
    """Restricted oracle: model top-20 U 100 random calibrated tiles (G=8), sustained 0.3/0.8/0.3 s, best kept (R5-V)."""
    from lala_triton.select import select_scored, _space
    from lala_triton import model as M3
    from lala_triton._measure import time_cfg
    cands = [c for _, c in select_scored(M, N, K, dtype, bench.layout, bench.prof, k=20)]
    cands += [c for _, c in select_scored(M, N, K, dtype, bench.layout, bench.prof, k=5, tiebreak=True)]   # Amendment 5b: tie-break picks always in the oracle pool
    sp = _space(bench.prof, dtype, bench.layout); rnd = random.Random(f'{rng_seed}-{M}-{N}-{K}-{dtype}').sample(sp, min(100, len(sp)))
    cands += [dict(BM=t['BM'], BN=t['BN'], BK=t['BK'], G=8, num_warps=t['num_warps'], num_stages=t['num_stages']) for t in rnd]
    seen, best = set(), (1e9, None)
    for c in cands:
        k = tuple(sorted(c.items()))
        if k in seen: continue
        seen.add(k)
        try: res, _ = time_cfg(sh, c, gpu, 0.3, 0.8, 0.3)
        except Exception: continue
        if res and res['ms_per_gemm'] < best[0]: best = (res['ms_per_gemm'], c)
    return best[1], best[0]

# ------------------------------------------------------------------ main loop
def main():
    a = parse(); a.shard = tuple(map(int, a.shard.split('/'))); win = tuple(map(float, a.window.split(',')))
    setup_env(a.gpu)
    if a.first_call: return first_call(a)
    import torch, numpy as np
    from lala_triton.models import gemm_shapes
    from lala_triton._sustained import make_graph, measure
    from lala_triton._measure import Shape
    from lala_triton import model as M3
    os.makedirs(a.out, exist_ok=True); path = f'{a.out}/{a.tag}.csv'
    done = set()
    if os.path.exists(path): done = {(r['model'], r['gemm'], r['dtype'], r['M'], r['method'], r['rep']) for r in csv.DictReader(open(path))}
    B = Bench(a.profile, a.gpu)
    methods = a.methods.split(','); ms = list(map(int, a.ms.split(','))); oracle_ms = set(map(int, a.oracle_ms.split(',')))
    avail = set(methods)
    if 'nvmm' in avail:
        try: import nMatmulHeuristics_probe  # noqa  (placeholder: nvMatmulHeuristics wrapper not available in this environment)
        except Exception: avail.discard('nvmm')
    if 'tritonblas' in avail:
        try: from lala_triton.baselines import tritonblas as TB
        except Exception: avail.discard('tritonblas')
    jobs = [(m, g, N, K, dt, M) for m in a.models.split(',') for g, (N, K) in gemm_shapes(m, a.tp).items() for dt in a.dtypes.split(',') for M in ms]
    if a.shapes: keep = {(x.split(':')[0], int(x.split(':')[1])) for x in a.shapes.split(',')}; jobs = [j for j in jobs if (j[1], j[5]) in keep]
    if a.gemms: jobs = [j for j in jobs if j[1] in a.gemms.split(',')]
    jobs = jobs[a.shard[0]::a.shard[1]]
    for (model, g, N, K, dt, M) in jobs:
        todo = [(me, r) for me in methods if me in avail and (me != 'oracle' or M in oracle_ms) for r in range(a.reps)
                if (model, g, dt, str(M), me, str(r)) not in done]
        if not todo: continue
        t0 = time.time(); sh = Shape(M, N, K, dt, B.layout); fns, infos, graphs = {}, {}, {}
        for me in sorted({me for me, _ in todo}):
            cfg = None
            if me == 'oracle': cfg, o_ms = oracle_cfg(B, sh, M, N, K, dt, a.gpu); infos[me] = dict(oracle_screen_ms=o_ms)
            if me == 'tritonblas': cfg = TB.select(M, N, K, dt, B.layout, B.prof)
            try:
                ts = time.time()
                fns[me], inf = B.launchers('oracle' if me == 'tritonblas' else me, sh.sets, dt, cfg); infos[me] = dict(infos.get(me, {}), **inf, setup_s=time.time() - ts)
                if me == 'tritonblas': infos[me]['cfg'] = json.dumps(cfg)
                # capture NOW: dynamo's cache is keyed on the (shared) lambda code object, so a later torch._dynamo.reset()/compile for
                # another method would silently swap which kernel this callable runs. A captured CUDA graph is immune.
                graphs[me] = make_graph(fns[me])[:2]
            except Exception as e:
                print('LAUNCHER FAIL', model, g, dt, M, me, repr(e)[:200], flush=True)
        order = [(me, r) for me, r in todo if me in graphs]
        for r in sorted({r for _, r in order}):
            batch = [me for me, rr in order if rr == r]; random.Random(f'{model}{g}{dt}{M}{r}').shuffle(batch)
            for me in batch:
                try:
                    gr, L = graphs[me]; res, _ = measure(gr, L, a.gpu, *win)
                except Exception as e:
                    print('MEASURE FAIL', me, repr(e)[:200], flush=True); continue
                pred = ''
                if me in ('ours1', 'ours1tb', 'ours1naive', 'oracle', 'tritonblas'):
                    try:
                        c = json.loads(infos[me].get('pick') or infos[me].get('cfg'))
                        c = dict(BM=c.get('BM', c.get('BLOCK_M')), BN=c.get('BN', c.get('BLOCK_N')), BK=c.get('BK', c.get('BLOCK_K')), G=c.get('G', c.get('GROUP_M', 8)),
                                 num_warps=c['num_warps'], num_stages=c['num_stages'])
                        pred = M3.predict(M, N, K, c, dt, B.prof, B.layout) * 1e3
                    except Exception: pred = ''
                row = dict(model=model, gemm=g, dtype=dt, M=M, N=N, K=K, method=me, rep=r, ms=res['ms_per_gemm'], sm_clk=res['sm_clk_med'], power=res['power_mean'],
                           temp0=res['temp0'], energy_J=res['energy_J_per_gemm'], energy_W=res['energy_power_W'], pred_ms=pred, pick=infos[me].get('pick') or infos[me].get('cfg') or '', setup_s=infos[me].get('setup_s', ''),
                           first_call_s=infos[me].get('first_call_s', ''), ts=time.strftime('%F %T'))
                new = not os.path.exists(path)
                with open(path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row)); 
                    if new: w.writeheader()
                    w.writerow(row)
        print(f'{model} {g} {dt} M={M} done in {time.time()-t0:.0f}s', flush=True)
        del fns, graphs; sh.close(); torch.cuda.empty_cache()

# ------------------------------------------------------------------ tuning cost (fresh process, fresh caches)
def first_call(a):
    import torch
    from lala_triton.models import gemm_shapes
    from lala_triton.kernels import make_inputs
    me, model, g, dt, M = a.first_call.split(':'); M = int(M); N, K = gemm_shapes(model)[g]
    B = Bench(a.profile, a.gpu); a_, b_, _ = make_inputs(M, N, K, dt, 'ABT')
    sel_s = 0.0
    if me.startswith('ours'):
        from lala_triton.select import select; t0 = time.time(); select(M, N, K, dt, 'ABT', B.prof, k=1 if me == 'ours1' else 5); sel_s = time.time() - t0
    from lala_triton._nvml import handle, energy_mJ
    hh = handle(0); time.sleep(1.0); i0, it0 = energy_mJ(hh), time.time(); time.sleep(2.0); idle_W = (energy_mJ(hh) - i0) / 1e3 / (time.time() - it0)
    e0 = energy_mJ(hh)
    t0 = time.time(); fns, info = B.launchers(me, [(a_, b_, None)], dt); total = time.time() - t0
    e_J = (energy_mJ(hh) - e0) / 1e3
    bench_s = sum(float(x) for x in re.findall(r'benchmarking takes ([\d.]+) seconds', ' '.join(B.autolog))) if B.autolog else 0.0
    print('@@' + json.dumps(dict(method=me, model=model, gemm=g, dtype=dt, M=M, select_s=sel_s, first_call_s=total, bench_s=bench_s, compile_s=total - bench_s, energy_J=e_J, idle_W=idle_W, energy_excess_J=e_J - idle_W * total)))

if __name__ == '__main__': main()
