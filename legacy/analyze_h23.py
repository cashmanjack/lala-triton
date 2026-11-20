import json, glob, numpy as np, pandas as pd
from scipy.stats import kendalltau
pd.set_option('display.width', 250)
R = '/home/min/a/cashman3/lala-triton-killtest'
pred = pd.read_csv(f'{R}/predictions/phase3_pred.csv')
meas = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/runs/phase3_gpu*.csv')])
ridge = json.load(open(f'{R}/env/ridge.json'))
noise = meas[meas.rep.astype(str).str.startswith('noise')]
main = meas[~meas.rep.astype(str).str.startswith('noise') & (meas.ok == 1)]
nreps = main.groupby(['shape', 'cfg']).ms.agg(['median', 'size']).reset_index()
nfeas = pred.groupby('shape').size()
complete = [s for s, g in nreps.groupby('shape') if (g['size'] >= 3).sum() >= 0.95 * nfeas[s]]
fails = meas[meas.ok == 0].groupby('shape').cfg.nunique()
KS = [1, 3, 5, 10, 20, '5%', '10%']
rows, h2 = [], []
for s in complete:
    t = nreps[(nreps['shape'] == s) & (nreps['size'] >= 3)].set_index('cfg')['median']
    p = pred[pred['shape'] == s].set_index('cfg').loc[t.index]
    n = len(t); oracle = t.min()
    nf = noise[noise['shape'] == s].ms; nfl = (nf.max() - nf.min()) / nf.median() if len(nf) else np.nan
    kk = {k: (k if isinstance(k, int) else max(1, int(round(n * int(k[:-1]) / 100)))) for k in KS}
    order = {'M3': p.sort_values(['t_m3', 'm2_tiebreak']).index, 'M1': p.sort_values(['t_m1', 'm2_tiebreak']).index,
             'M2': p.sort_values(['t_m2', 'm2_tiebreak']).index}
    tv = t.values
    cum = {m: np.minimum.accumulate(t.loc[o].values) / oracle for m, o in order.items()}
    rng = np.random.default_rng(0)
    cum['M0'] = np.mean([np.minimum.accumulate(rng.permutation(tv)) for _ in range(100)], axis=0) / oracle
    best3 = order['M3'][0]
    inten = 2 * p.M.iloc[0] * p.N.iloc[0] * p.K.iloc[0] / p.loc[best3, 'm3_read']
    dt = p.dtype.iloc[0]
    near = 0.5 <= inten / ridge['ridge_measured'][dt] <= 2
    near_pub = 0.5 <= inten / ridge['ridge_published'][dt] <= 2
    for m, c in cum.items():
        tau = kendalltau(-p.loc[t.index, f't_{m.lower()}'] if m != 'M0' else np.zeros(n), -t).correlation if m != 'M0' else 0
        kmin90 = int(np.argmax(c <= 1.05)) + 1   # smallest k reaching regret<=1.05 for this shape
        rows.append(dict(shape=s, dtype=dt, model=m, n=n, near=near, near_pub=near_pub, intensity=inten, noise=nfl,
                         tau=tau, kmin=kmin90, kmin_frac=kmin90 / n, **{f'r@{k}': c[kk[k] - 1] for k in KS}))
    # H2
    pp = p.assign(t=t)
    otile = pp.loc[t.idxmin(), 'tile']; tt = pp[pp.tile == otile]
    g8 = tt[tt.G == 8].t.min(); bestG = tt.loc[tt.t.idxmin(), 'G']
    best_g8_overall = pp[pp.G == 8].t.min(); best_g8_tile = pp.loc[pp[pp.G == 8].t.idxmin(), 'tile']
    h2.append(dict(shape=s, near=near, near_pub=near_pub, intensity=inten, noise=nfl, oracle_cfg=t.idxmin(),
                   oracle_ms=oracle, bestG_for_tile=bestG, G8_vs_bestG=g8 / tt.t.min() - 1,
                   tile_changes=best_g8_tile != otile, bestG8cfg_vs_oracle=best_g8_overall / oracle - 1,
                   G_range_for_tile=tt.t.max() / tt.t.min() - 1))
res = pd.DataFrame(rows); h2 = pd.DataFrame(h2)
res.to_csv(f'{R}/runs/h3_per_shape.csv', index=False); h2.to_csv(f'{R}/runs/h2_per_shape.csv', index=False)
print(f'complete shapes: {len(complete)}/{len(nfeas)}; correctness failures per shape:', dict(fails))
print('\n== H2 =='); print(h2.round(3).to_string())
nr = h2[h2.near]
hit = ((nr.G8_vs_bestG >= 0.05) & (nr.G8_vs_bestG > nr.noise)) | (nr.tile_changes & (nr.bestG8cfg_vs_oracle >= 0.05))
print(f'near-ridge (measured ridge) shapes: {len(nr)}; pass-condition hits: {hit.sum()} ({hit.mean() if len(nr) else float("nan"):.0%})')
nrp = h2[h2.near_pub]
hitp = ((nrp.G8_vs_bestG >= 0.05) & (nrp.G8_vs_bestG > nrp.noise)) | (nrp.tile_changes & (nrp.bestG8cfg_vs_oracle >= 0.05))
print(f'near-ridge (published ridge) shapes: {len(nrp)}; hits: {hitp.sum()}')
allhit = ((h2.G8_vs_bestG >= 0.05) & (h2.G8_vs_bestG > h2.noise)) | (h2.tile_changes & (h2.bestG8cfg_vs_oracle >= 0.05))
print(f'all shapes: hits {allhit.sum()}/{len(h2)}; max G8_vs_bestG={h2.G8_vs_bestG.max():.3f}')
print('\n== H3 ==')
cols = [f'r@{k}' for k in KS]
print(res.groupby('model')[cols + ['tau', 'kmin']].mean().round(3).to_string())
print('frac shapes regret<=1.05 at k=10%:', res.groupby('model')['r@10%'].apply(lambda x: (x <= 1.05).mean()).round(3).to_dict())
def k90(g):   # smallest k s.t. >=90% of shapes have regret<=1.05
    return np.quantile(g.kmin, 0.9, method='higher')
def k90f(g): return np.quantile(g.kmin_frac, 0.9, method='higher')
print('k for 90% of shapes (abs):', res.groupby('model').apply(k90).to_dict(), ' (frac of feasible):', res.groupby('model').apply(k90f).round(3).to_dict())
for sub, name in [(res[res.near], 'near-ridge(measured)'), (res[~res.near], 'not near-ridge')]:
    print(name, sub['shape'].nunique(), 'shapes'); print(sub.groupby('model')[cols].mean().round(3).to_string())
print('mean noise floor all', round(h2.noise.mean(),4), 'near', round(h2[h2.near].noise.mean(),4))
print(res[res.model.isin(['M1', 'M3'])].pivot_table(index='shape', columns='model', values=['r@1', 'r@5', 'r@10%', 'kmin']).round(3).to_string())
