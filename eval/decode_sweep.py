"""Round 6 Step 1: time EVERY feasible tile (BM <= min(64, 5d cap)) on small-M calibration shapes. Resumable, append-only CSV.
python3 eval/decode_sweep.py --gpu 0 --shard 0/2 --out results/r6/decode_sweep.csv
Calibration shapes only (synthetic (N,K)); never validation shapes. Short timing (0.15 s warmup / 0.3 s window / no cooldown): diagnosis-grade, not final."""
import argparse, csv, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--shard', default='0/1')
ap.add_argument('--out', default='results/r6/decode_sweep.csv'); ap.add_argument('--dtypes', default='bf16,fp8'); ap.add_argument('--ms', default='1,2,4,8,16,32,64')
ap.add_argument('--nk', default='2048x2048,6144x6144,12288x3072,3072x12288'); ap.add_argument('--window', type=float, default=0.3)
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
from lala_triton import space, _measure, model as M3
from lala_triton.model import tile_key
import torch
si, sn = map(int, a.shard.split('/'))
done = set()
if os.path.exists(a.out):
    for r in csv.DictReader(open(a.out)): done.add((r['dtype'], r['N'], r['K'], r['M'], r['tile']))
cols = ['dtype', 'N', 'K', 'M', 'tile', 'ok', 'n_regs', 'shared', 'occ', 'ms', 'sm_clk', 'power']
def emit(row):
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    with open(a.out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); 
        if new: w.writeheader()
        w.writerow(row)
jobs = []
for dt in a.dtypes.split(','):
    for nk in a.nk.split(','):
        N, K = map(int, nk.split('x'))
        for M in map(int, a.ms.split(',')):
            cap = M3.bm_cap(M)
            for t in space.candidate_tiles(dt):
                if t['BM'] <= min(64, cap): jobs.append((dt, N, K, M, t))
jobs = [j for i, j in enumerate(jobs) if i % sn == si]
print(len(jobs), 'jobs in shard', flush=True)
cur, sh, t0 = None, None, time.time()
for i, (dt, N, K, M, t) in enumerate(jobs):
    tk = tile_key(t['BM'], t['BN'], t['BK'], t['num_stages'], t['num_warps'])
    if (dt, str(N), str(K), str(M), tk) in done: continue
    if cur != (dt, N, K, M):
        if sh: sh.close()
        sh = _measure.Shape(M, N, K, dt, 'ABT'); cur = (dt, N, K, M)
    cfg = dict(t, G=8)
    try: res, meta = _measure.time_cfg(sh, cfg, 0, warmup_s=0.15, window_s=a.window, cooldown_s=0.0)
    except Exception as e:
        emit(dict(dtype=dt, N=N, K=K, M=M, tile=tk, ok=0)); print('fail', tk, repr(e)[:100], flush=True); continue
    s = 1 if dt == 'fp8' else 2
    occ = M3.occupancy(meta['shared'], meta['n_regs'], t['num_warps'])
    emit(dict(dtype=dt, N=N, K=K, M=M, tile=tk, ok=int(res is not None), n_regs=meta['n_regs'], shared=meta['shared'], occ=occ,
              ms=(res['ms_per_gemm'] if res else ''), sm_clk=(res['sm_clk_med'] if res else ''), power=(res['power_mean'] if res else '')))
    if i % 50 == 0: print(f'{i}/{len(jobs)} {time.time()-t0:.0f}s', dt, N, K, M, tk, flush=True)
print('shard done', flush=True)
