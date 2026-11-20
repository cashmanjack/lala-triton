"""Phase E: dynamic M. GPU0, resumable. Stages: exh (static+bucket tuning), oracle (restricted per-M), final (re-time picks)."""
import sys, os, csv, json, random, time
R = '/home/min/a/cashman3/lala-triton-killtest'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ.setdefault('TRITON_CACHE_DIR', f'{R}/.triton_cache')
sys.path.insert(0, R)
import torch, pandas as pd, numpy as np
from triton.testing import do_bench
from common import make_inputs
from kernels.matmul_grouped import matmul
EM = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ENK = [(4096, 4096), (14336, 4096), (4096, 14336)]
BUCKETS = [(1, 16, 8), (17, 128, 72), (129, 1024, 576), (1025, 10**9, 4608)]
choice = json.load(open(f'{R}/runs/phaseC_verdict.json'))['choice']
pred = pd.read_csv(f'{R}/predictions/phaseDE_pred.csv'); pred = pred[pred.phase == 'E']
comp = pd.read_csv(f'{R}/runs/compile_all.csv'); feas = comp[(comp.ok == 1) & (comp.n_spills == 0) & (comp.dtype == 'bf16')]
CFG = {r.cfg: dict(BM=r.BM, BN=r.BN, BK=r.BK, G=r.G, num_warps=r.num_warps, num_stages=r.num_stages) for r in feas.itertuples()}
TUNE = f'{R}/runs/phaseE_tune.csv'; FINAL = f'{R}/runs/phaseE_final.csv'
def load_done(p, keys):
    return {tuple(r[k] for k in keys) for r in csv.DictReader(open(p))} if os.path.exists(p) else set()
def writer(p, fields):
    new = not os.path.exists(p); f = open(p, 'a', newline=''); w = csv.DictWriter(f, fieldnames=fields)
    if new: w.writeheader()
    return f, w
_inp = {}
def inputs(M, N, K):
    k = (M, N, K)
    if k not in _inp:
        _inp.clear(); torch.cuda.empty_cache(); _inp[k] = make_inputs(M, N, K, 'bf16', layout='ABT')
    return _inp[k]
def bench(M, N, K, cname, warmup, rep):
    a, b, c = inputs(M, N, K); cf = CFG[cname]
    try:
        return do_bench(lambda: matmul(a, b, c, cf['BM'], cf['BN'], cf['BK'], cf['G'], cf['num_warps'], cf['num_stages']), warmup=warmup, rep=rep, return_mode='median')
    except Exception as e:
        return float('nan')
# ---------- stage 1: exhaustive tuning at static (512) and bucket midpoints
f, w = writer(TUNE, ['stage', 'N', 'K', 'M', 'cfg', 'ms', 'ts'])
done = load_done(TUNE, ['stage', 'N', 'K', 'M', 'cfg'])
for (N, K) in ENK:
    for M in [512, 8, 72, 576, 4608]:
        names = list(CFG); random.Random(f'E-{N}-{K}-{M}').shuffle(names)
        for cn in names:
            if ('exh', str(N), str(K), str(M), cn) in done: continue
            w.writerow(dict(stage='exh', N=N, K=K, M=M, cfg=cn, ms=bench(M, N, K, cn, 5, 10), ts=time.time())); f.flush()
    print('exh done', N, K, flush=True)
tune = pd.read_csv(TUNE)
exh = tune[tune.stage == 'exh'].dropna(subset=['ms'])
best_at = lambda N, K, M: exh[(exh.N == N) & (exh.K == K) & (exh.M == M)].sort_values('ms').cfg.iloc[0]
picks = {}
for (N, K) in ENK:
    for M in EM:
        p = pred[(pred.N == N) & (pred.K == K) & (pred.M == M)].sort_values(f't_{choice.lower()}')
        bmid = [mid for lo, hi, mid in BUCKETS if lo <= M <= hi][0]
        picks[(N, K, M)] = dict(static=best_at(N, K, 512), bucket=best_at(N, K, bmid), model=p.cfg.iloc[0], model_top50=list(p.cfg.head(50)))
# ---------- stage 2: restricted per-M oracle
done = load_done(TUNE, ['stage', 'N', 'K', 'M', 'cfg'])
rnd = random.Random(0).sample(sorted(CFG), 150)
for (N, K) in ENK:
    for M in EM:
        pk = picks[(N, K, M)]
        cand = list(dict.fromkeys(pk['model_top50'] + [pk['static'], pk['bucket']] + rnd))
        for cn in cand:
            if ('oracle', str(N), str(K), str(M), cn) in done: continue
            w.writerow(dict(stage='oracle', N=N, K=K, M=M, cfg=cn, ms=bench(M, N, K, cn, 5, 10), ts=time.time())); f.flush()
    print('oracle done', N, K, flush=True)
f.close()
tune = pd.read_csv(TUNE); orc = tune[tune.stage == 'oracle'].dropna(subset=['ms'])
for (N, K, M), pk in picks.items():
    pk['oracle'] = orc[(orc.N == N) & (orc.K == K) & (orc.M == M)].sort_values('ms').cfg.iloc[0]
json.dump({f'{N}_{K}_{M}': {k: v for k, v in pk.items() if k != 'model_top50'} for (N, K, M), pk in picks.items()},
          open(f'{R}/runs/phaseE_picks.json', 'w'), indent=1)
# ---------- stage 3: final re-timing (do_bench 25/100, L2 flushed), 3 interleaved reps; + inductor dynamic (e)
import torch._inductor.config as ic
ic.max_autotune_gemm_backends = 'ATEN,TRITON'
ind_ok = {}
f, w = writer(FINAL, ['N', 'K', 'M', 'method', 'launcher', 'rep', 'ms', 'ts'])
done = load_done(FINAL, ['N', 'K', 'M', 'launcher', 'rep'])
for (N, K) in ENK:
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = f'{R}/runs/phaseE_ind/{N}_{K}'
    fdyn = torch.compile(lambda x, y: x @ y, mode='max-autotune-no-cudagraphs', dynamic=True)
    try:
        a, b, c = inputs(512, N, K); torch._dynamo.mark_dynamic(a, 0); t0 = time.time(); fdyn(a, b); torch.cuda.synchronize()
        ind_ok[(N, K)] = time.time() - t0
    except Exception as e:
        ind_ok[(N, K)] = repr(e)[:200]
    for M in EM:
        pk = picks[(N, K, M)]
        launchers = {}
        for meth in ('static', 'bucket', 'model', 'oracle'): launchers.setdefault(pk[meth], []).append(meth)
        items = [(cn, '|'.join(ms)) for cn, ms in launchers.items()] + ([('inductor_dyn', 'inductor')] if isinstance(ind_ok[(N, K)], float) else [])
        for rep in range(3):
            random.Random(f'EF-{N}-{K}-{M}-{rep}').shuffle(items)
            for cn, meth in items:
                if (str(N), str(K), str(M), cn, str(rep)) in done: continue
                if cn == 'inductor_dyn':
                    a, b, c = inputs(M, N, K)
                    try: ms = do_bench(lambda: fdyn(a, b), warmup=25, rep=100, return_mode='median')
                    except Exception: ms = float('nan')
                else:
                    ms = bench(M, N, K, cn, 25, 100)
                w.writerow(dict(N=N, K=K, M=M, method=meth, launcher=cn, rep=rep, ms=ms, ts=time.time())); f.flush()
    print('final done', N, K, ind_ok[(N, K)], flush=True)
json.dump({f'{k[0]}_{k[1]}': v for k, v in ind_ok.items()}, open(f'{R}/runs/phaseE_inductor.json', 'w'))
print('PHASE E DONE')
