"""Supplementary (NOT pre-registered) robustness checks."""
import glob, numpy as np, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
pred = pd.read_csv(f'{R}/predictions/phase3_pred.csv')
m = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/runs/phase3_gpu*.csv')])
m = m[~m.rep.astype(str).str.startswith('noise')]
h2 = pd.read_csv(f'{R}/runs/h2_per_shape.csv')
# 1. per-rep consistency of the G effect for the oracle tile
out = []
for r in h2.itertuples():
    tile = pred[(pred['shape'] == r.shape) & (pred.cfg == r.oracle_cfg)].tile.iloc[0]
    x = m[(m['shape'] == r.shape) & (m.tile == tile)]
    per = x.pivot_table(index='rep', columns='G', values='ms')
    gains = per[8] / per[r.bestG_for_tile] - 1
    out.append(dict(shape=r.shape, near=r.near, bestG=r.bestG_for_tile, med_gain=r.G8_vs_bestG,
                    rep_gains=list(gains.round(3)), all_reps_ge5=(gains >= 0.05).all()))
o = pd.DataFrame(out); print(o.to_string())
print('shapes with median gain>=5%:', (o.med_gain >= .05).sum(), ' of which every rep >=5%:', (o.all_reps_ge5 & (o.med_gain >= .05)).sum())
# 2. M1 with G fixed to 8 (practitioner baseline) and M3 restricted to G=8 (tile-choice only)
med = m.groupby(['shape', 'cfg']).ms.median()
rows = []
for s, p in pred.groupby('shape'):
    t = med.loc[s]; p = p.set_index('cfg').loc[t.index]; o_ = t.min()
    def reg(order, k): return t.loc[order[:k]].min() / o_
    m1g8 = p[p.G == 8].sort_values('t_m1').index
    m3g8 = p[p.G == 8].sort_values('t_m3').index
    m3 = p.sort_values(['t_m3', 'm2_tiebreak']).index
    for k in (1, 3, 5, 10):
        rows.append(dict(shape=s, k=k, M1_G8=reg(m1g8, k), M3_G8only=reg(m3g8, k), M3=reg(m3, k)))
d = pd.DataFrame(rows).merge(h2[['shape', 'near']], on='shape')
print(d.groupby(['k']).mean(numeric_only=True).round(3)); print('near-ridge'); print(d[d.near].groupby('k').mean(numeric_only=True).round(3))
