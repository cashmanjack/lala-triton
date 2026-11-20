"""H1 extension: ncu DRAM bytes on a 10% random subset of Phase 3 (shape, config) pairs."""
import sys, json, csv, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from ncu_parse import UNIT
R = '/home/min/a/cashman3/lala-triton-killtest'
recs = []
for part in (0, 1):
    pred = pd.read_csv(f'{R}/predictions/phase3_ncu_subset_part{part}.csv')
    for k, _ in pred.groupby(['sweep', 'M', 'N', 'K', 'dtype']):
        base = f'{R}/runs/ncu_phase3/' + '_'.join(map(str, k))
        idx = json.load(open(base + '.manifest.json'))
        rows = [r for r in csv.reader(open(base + '.csv')) if len(r) > 12 and r[0].isdigit()]
        ids = sorted(set(int(r[0]) for r in rows)); assert len(ids) == len(idx), base
        for kid, pi in zip(ids, idx):
            mm = {r[12]: float(r[14].replace(',', '')) * UNIT.get(r[13], 1) for r in rows if int(r[0]) == kid}
            recs.append(dict(pred.loc[pi].to_dict(), meas_read=mm['dram__bytes_read.sum']))
d = pd.DataFrame(recs)
d['err'] = (d.pred_read - d.meas_read).abs() / d.meas_read
d['err_naive'] = (d.naive_read - d.meas_read).abs() / d.meas_read
d['ratio_meas'] = d.meas_read / d.compulsory
d.to_csv(f'{R}/runs/phase3_ncu_joined.csv', index=False)
print(f'N={len(d)} median err M3={d.err.median():.3f} M1={d.err_naive.median():.3f} '
      f'spearman M3={spearmanr(d.pred_read, d.meas_read).correlation:.3f} M1={spearmanr(d.naive_read, d.meas_read).correlation:.3f}')
print(d.groupby(['dtype', 'regime']).agg(n=('err', 'size'), med_err=('err', 'median'), p90=('err', lambda x: x.quantile(.9))).round(3))
d['under'] = d.meas_read > 1.5 * d.pred_read
print('measured >1.5x predicted:', d.under.mean().round(3)); print(d[d.under].groupby(['regime']).size())
print(d.sort_values('err', ascending=False)[['shape', 'cfg', 'regime', 'W', 'pred_read', 'meas_read', 'err']].head(10).to_string())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6', 'font.size': 9})
fig, ax = plt.subplots(figsize=(5, 5))
ax.loglog(d.meas_read / 2**30, d.naive_read / 2**30, 'o', ms=2, color='#eb6834', alpha=.35, label='M1 naive per-tile')
ax.loglog(d.meas_read / 2**30, d.pred_read / 2**30, 'o', ms=2, color='#2a78d6', alpha=.5, label='M3 L2-aware')
lim = [d.meas_read.min() / 2**30 * .7, d.naive_read.max() / 2**30 * 1.3]; ax.plot(lim, lim, color='#999', lw=1)
ax.set_xlabel('measured DRAM read (GiB, ncu)'); ax.set_ylabel('predicted (GiB)'); ax.legend()
ax.set_title(f'Phase 3 random 10% subset (n={len(d)})'); fig.tight_layout(); fig.savefig(f'{R}/plots/phase3_scatter.png', dpi=140)
