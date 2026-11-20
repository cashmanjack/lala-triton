"""Round-3 M3 predictions for the Inductor template families, written before any 2c/2d timing.
Every config of E(shape) for the 62 Phase D shapes and the Phase E grid (bf16 + fp8, A@B.T).
Ranks: rank_s4 over E with exact-tie collapse to G=8 within a tile; rank_s5 over G=8 entries only."""
import sys, datetime, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
sys.path.insert(0, R)
from common import ESIZE, OUT_ESIZE, cfg_key
from model import locality_model as m3
from model.time_model import window
from kernels.inductor_tmpl import extended_space, MM_STOCK, SMM_STOCK
from phaseD_shapes import SHAPES, key
meta = pd.read_csv(f'{R}/env/r3_tmpl_meta.csv').set_index(['dtype', 'layout', 'tile'])
eff = pd.read_csv(f'{R}/env/r3_eff_tmpl.csv').set_index(['dtype', 'layout', 'tile']).eff
EM = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ENK = [(4096, 4096), (14336, 4096), (4096, 14336)]
pts = [(*s, 'D') for s in SHAPES] + [(m, n, k, dt, 'ABT', 'E') for dt in ('bf16', 'fp8') for (n, k) in ENK for m in EM + [72, 576, 4608]]
def tk(bm, bn, bk, s, w): return f'{bm}x{bn}x{bk}_w{w}_s{s}'
rows = []
for (M, N, K, dt, lay, ph) in pts:
    s, so = ESIZE[dt], OUT_ESIZE[dt]
    stock = SMM_STOCK if dt == 'fp8' else MM_STOCK
    fam_med = eff.loc[(dt, lay)].median()
    for c in extended_space(M, N, K, dt):
        t = tk(c['BM'], c['BN'], c['BK'], c['num_stages'], c['num_warps'])
        mr = meta.loc[(dt, lay, t)]
        if not mr.ok: continue                      # compile failure (shared memory), as Inductor drops it
        p = stock[c['parent']]; pt = tk(p[0], p[1], p[2], p[3], p[4])
        e, esrc = (eff[(dt, lay, pt)], 'parent') if (dt, lay, pt) in eff.index else (fam_med, 'family_median')
        cfg = {k: c[k] for k in ('BM', 'BN', 'BK', 'G', 'num_warps', 'num_stages')}
        occ = int(mr.occ)
        _, W = window(M, N, c['BM'], c['BN'], occ)
        A, B, info = m3.traffic(M, N, K, c['BM'], c['BN'], c['G'], W, s)
        rows.append(dict(phase=ph, shape=key(M, N, K, dt, lay), M=M, N=N, K=K, dtype=dt, layout=lay, cfg=cfg_key(cfg), tile=t, **cfg,
                         parent_tile=pt, eff=e, eff_src=esrc, occ=occ, n_regs=int(mr.n_regs), n_spills=int(mr.n_spills),
                         m3_read=A + B, m3_regime=info['regime'], t_m3=m3.predict(M, N, K, cfg, occ, e, dt, s, so)))
d = pd.DataFrame(rows)
# tie collapse: drop G != 8 entries whose prediction equals the same tile's G=8 prediction
g8 = d[d.G == 8].set_index(['phase', 'shape', 'tile']).t_m3
ref = g8.reindex(pd.MultiIndex.from_frame(d[['phase', 'shape', 'tile']])).values
d['tie_collapsed'] = (d.G != 8) & (abs(d.t_m3 - ref) <= 1e-6 * ref)
d['rank_s4'] = d[~d.tie_collapsed].groupby(['phase', 'shape']).t_m3.rank(method='first')
d['rank_s5'] = d[d.G == 8].groupby(['phase', 'shape']).t_m3.rank(method='first')
d['written'] = datetime.datetime.now().isoformat()
d.to_csv(f'{R}/predictions/r3_pred.csv', index=False)
print(len(d), d.groupby(['phase', 'dtype']).shape.nunique().to_dict(), 'configs/shape:', d.groupby('shape').size().describe().round(1).to_dict())
print('eff_src', d.eff_src.value_counts().to_dict(), 'collapsed', int(d.tie_collapsed.sum()))
D = d[d.phase == 'D']
top = D[D.rank_s4 == 1]
print('top-1 G distribution (D):', top.G.value_counts().to_dict())
