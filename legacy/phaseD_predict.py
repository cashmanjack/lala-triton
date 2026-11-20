"""After Phase C: model predictions (M3, M4, M1) for every feasible config on all 62 Phase D shapes + Phase E grid."""
import sys, json, datetime, pandas as pd
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from common import ESIZE, OUT_ESIZE
from model import locality_model as m3, naive_model as m1, power_model as m4
from phaseD_shapes import SHAPES, key
comp = pd.read_csv('runs/compile_all.csv'); feas = comp[(comp.ok == 1) & (comp.n_spills == 0)]
eff = pd.read_csv('env/eff_table.csv').set_index(['dtype', 'tile']).eff
EM = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ENK = [(4096, 4096), (14336, 4096), (4096, 14336)]
shapes = [(*s, 'D') for s in SHAPES] + [(m, n, k, 'bf16', 'ABT', 'E') for (n, k) in ENK for m in EM + [8, 72, 576, 4608]]
rows = []
for (M, N, K, dt, lay, ph) in shapes:
    s, so = ESIZE[dt], OUT_ESIZE[dt]
    for r in feas[feas.dtype == dt].itertuples():
        cfg = dict(BM=r.BM, BN=r.BN, BK=r.BK, G=r.G, num_warps=r.num_warps, num_stages=r.num_stages); occ = int(r.occ)
        e = eff[(dt, r.tile)]
        rows.append(dict(phase=ph, shape=key(M, N, K, dt, lay), M=M, N=N, K=K, dtype=dt, layout=lay, cfg=r.cfg, tile=r.tile, **cfg, occ=occ,
                         t_m3=m3.predict(M, N, K, cfg, occ, e, dt, s, so), t_m1=m1.predict(M, N, K, cfg, occ, e, dt, s, so),
                         t_m4=m4.predict(M, N, K, cfg, occ, r.tile, dt)))
d = pd.DataFrame(rows).drop_duplicates(['shape', 'cfg', 'phase']); d['written'] = datetime.datetime.now().isoformat()
d.to_csv('predictions/phaseDE_pred.csv', index=False); print(len(d), d['shape'].nunique())
