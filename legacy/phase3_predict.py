"""Compute eff table (Phase 1) and write M1/M2/M3 predictions for every (test shape, feasible config)."""
import sys, json, datetime, numpy as np, pandas as pd
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from common import ESIZE, OUT_ESIZE
from model import locality_model as m3, naive_model as m1, triton_perf_model as m2
from model.time_model import ideal_compute, HBM_BW, window
comp = pd.read_csv('runs/compile_all.csv')
feas = comp[(comp.ok == 1) & (comp.n_spills == 0)]
cal = pd.read_csv('runs/eff_calib.csv')
cal = cal[cal.ok == 1].copy()
cal['ideal'] = [ideal_compute(r.M, r.N, r.K, r.BM, r.BN, int(r.occ), r.dtype) for r in cal.itertuples()]
cal['eff'] = cal.ideal / (cal.ms * 1e-3)
eff = cal.groupby(['dtype', 'tile']).eff.median()
eff.to_csv('env/eff_table.csv')
cal['tflops'] = 2 * cal.M * cal.N * cal.K / (cal.ms * 1e-3)
ridge = {d: cal[cal.dtype == d].tflops.max() / HBM_BW for d in ('bf16', 'fp8')}
ridge_pub = {'bf16': 364.2e12 / HBM_BW, 'fp8': 728.5e12 / HBM_BW}
json.dump(dict(ridge_measured=ridge, ridge_published=ridge_pub,
               max_tflops={d: cal[cal.dtype == d].tflops.max() / 1e12 for d in ('bf16', 'fp8')}),
          open('env/ridge.json', 'w'), indent=1)
print('eff summary', eff.groupby('dtype').describe().round(3).to_string(), '\nridge', ridge)

LLM = [(m, n, k) for m in (128, 512, 2048, 8192) for n, k in [(4096, 4096), (14336, 4096), (4096, 14336), (28672, 8192), (8192, 28672)]]
SHAPES = [(*s, 'bf16') for s in LLM + [(2048, 2048, 65536), (4096, 4096, 4096)]] + [(*s, 'fp8') for s in LLM]
rows = []
rng = np.random.default_rng(0)
for (M, N, K, dt) in SHAPES:
    s, so = ESIZE[dt], OUT_ESIZE[dt]
    for r in feas[feas.dtype == dt].itertuples():
        cfg = dict(BM=r.BM, BN=r.BN, BK=r.BK, G=r.G, num_warps=r.num_warps, num_stages=r.num_stages)
        occ = int(r.occ); e = eff[(dt, r.tile)]
        P, W = window(M, N, r.BM, r.BN, occ)
        A, B, info = m3.traffic(M, N, K, r.BM, r.BN, r.G, W, s)
        rows.append(dict(shape=f'{M}x{N}x{K}_{dt}', M=M, N=N, K=K, dtype=dt, cfg=r.cfg, tile=r.tile, BM=r.BM, BN=r.BN,
                         BK=r.BK, G=r.G, num_warps=r.num_warps, num_stages=r.num_stages, occ=occ, eff=e, W=W,
                         m3_regime=info['regime'], m3_read=A + B, m1_read=m1.dram_read_bytes(M, N, K, cfg, occ, s),
                         t_m3=m3.predict(M, N, K, cfg, occ, e, dt, s, so), t_m1=m1.predict(M, N, K, cfg, occ, e, dt, s, so),
                         t_m2=m2.predict(M, N, K, cfg, s), m2_tiebreak=rng.random(),
                         written=datetime.datetime.now().isoformat()))
df = pd.DataFrame(rows); df.to_csv('predictions/phase3_pred.csv', index=False)
print(len(df), df.shape[0] / len(SHAPES))
print(df.groupby('shape').m3_regime.value_counts().unstack().fillna(0).astype(int).to_string())
