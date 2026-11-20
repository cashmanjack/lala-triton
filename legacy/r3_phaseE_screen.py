"""Phase 2d screening worker for one (dtype, N, K): every M (+ midpoints), every E(M) config, dynamic-M harness,
do_bench(warmup=5, rep=10, median), 1 rep, seeded random order. argv: dtype N K gpu"""
import sys, os, csv, random, time
R = '/home/min/a/cashman3/lala-triton-killtest'
dt, N, K, gpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.insert(0, R)
import torch, pandas as pd
from triton.testing import do_bench
from common import make_inputs
from kernels.inductor_tmpl import launch
from r3_lib import done_set
SCR = f'{R}/runs/r3_E_screen.csv'
pred = pd.read_csv(f'{R}/predictions/r3_pred.csv')
pred = pred[(pred.phase == 'E') & (pred.dtype == dt) & (pred.N == N) & (pred.K == K)]
done = done_set(SCR, ['dtype', 'N', 'K', 'M', 'cfg'])
new = not os.path.exists(SCR)
f = open(SCR, 'a', newline=''); w = csv.writer(f)
if new: w.writerow(['dtype', 'N', 'K', 'M', 'cfg', 'G', 'ms', 'ok', 'maxerr', 'ts'])
for M, g in pred.groupby('M'):
    rows = [r for r in g.itertuples() if (dt, str(N), str(K), str(M), r.cfg) not in done]
    if not rows: continue
    a, b, c = make_inputs(int(M), N, K, dt, layout='ABT')
    ref = a.float() @ b.float(); tol = (1e-2 * ref.abs().max().item() + 1e-2) if dt == 'bf16' else 5e-2 * ref.abs().max().item()
    random.Random(f'r3E-{dt}-{N}-{K}-{M}').shuffle(rows)
    for r in rows:
        cf = dict(BM=int(r.BM), BN=int(r.BN), BK=int(r.BK), G=int(r.G), num_warps=int(r.num_warps), num_stages=int(r.num_stages))
        try:
            c.zero_(); launch(a, b, c, cf, dt, dynamic_m=True); torch.cuda.synchronize()
            err = (c.float() - ref).abs().max().item()
            ms = do_bench(lambda: launch(a, b, c, cf, dt, dynamic_m=True), warmup=5, rep=10, return_mode='median')
            w.writerow([dt, N, K, M, r.cfg, r.G, ms, int(err <= tol), err, time.time()])
        except Exception as e:
            w.writerow([dt, N, K, M, r.cfg, r.G, float('nan'), 0, float('nan'), time.time()]); print('fail', M, r.cfg, repr(e)[:150], flush=True)
        f.flush()
    del a, b, c, ref; torch.cuda.empty_cache()
    print('E screen', dt, N, K, M, len(rows), time.strftime('%H:%M:%S'), flush=True)
