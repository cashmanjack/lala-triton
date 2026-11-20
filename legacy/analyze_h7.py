import json, glob, numpy as np, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
E = pd.read_csv(f'{R}/runs/phaseD_eval.csv')
print('rows', len(E), 'shapes', E['shape'].nunique(), 'incorrect launchers:', E[E.ok == 0][['shape', 'launcher']].drop_duplicates().values.tolist())
rows = []
METH = ['B1', 'B2', 'B3', 'B4', 'ours1', 'ours3', 'ours5', 'nvmmh1', 'nvmmh5']
for s, x in E.groupby('shape'):
    tune = json.load(open(f'{R}/runs/phaseD_tune/{s}.json'))
    lt = x.groupby('launcher').ms_per_gemm.agg(['mean', 'std'])
    cv = float((lt['std'] / lt['mean']).median())
    t = {}
    for r in x.drop_duplicates('launcher').itertuples():
        for m in r.methods.split('|'): t[m] = lt.loc[r.launcher, 'mean']
    top = tune['picks']['B2']['top'] or ''
    b2kind = 'aten' if top.startswith('mm ') or top.startswith('_scaled_mm ') else 'triton'
    row = dict(shape=s, dtype=x.dtype.iloc[0], layout=x.layout.iloc[0], cv=cv, tol=max(0.03, cv), b2kind=b2kind, b4kind=x.b4kind.iloc[0],
               **{f't_{m}': t.get(m, np.nan) for m in METH},
               **{f'wall_{m}': tune['tuning'].get(m, {}).get('wall_s', np.nan) for m in ['B2', 'B3', 'ours1', 'ours3', 'ours5', 'nvmmh1', 'nvmmh5']},
               b2_nconf=tune['tuning']['B2'].get('n_configs'))
    rows.append(row)
d = pd.DataFrame(rows)
for m in METH: d[f'r_{m}'] = d[f't_{m}'] / d[['t_B4', 't_B2', 't_ours5', 't_B3', 't_nvmmh5']].min(axis=1)   # vs best Triton-kernel-family incl. inductor
d['ours5_vs_B2'] = d.t_ours5 / d.t_B2; d['B2_vs_B4'] = d.t_B2 / d.t_B4; d['ours5_vs_B1'] = d.t_ours5 / d.t_B1; d['B2_vs_B1'] = d.t_B2 / d.t_B1
d['wall_ratio'] = d.wall_ours5 / d.wall_B2
d.to_csv(f'{R}/runs/h7_per_shape.csv', index=False)
n = len(d)
c1 = (d.ours5_vs_B2 <= 1 + d.tol).mean(); c2 = (d.ours5_vs_B2 <= 0.95).mean(); c3 = d.wall_ratio.median()
kill = (d.B2_vs_B4 <= 1 + d.tol).mean()
print(f'\nH7 over {n} shapes: ours5 >= B2 within noise on {c1:.0%} (need 70%); ours5 >=5% faster than B2 on {c2:.0%} (need 20%); '
      f'median tuning wall ratio ours5/B2 = {c3:.3f} (need <= 0.2)')
print(f'KILL check: B2 within noise of B4 on {kill:.0%} (kill if >= 80%)')
v = 'KILL' if kill >= 0.8 else ('PASS' if (c1 >= 0.7 and c2 >= 0.2 and c3 <= 0.2) else 'FAIL')
print('H7 verdict:', v)
print('\nB2 picked ATen (cuBLAS) on', (d.b2kind == 'aten').sum(), 'shapes')
print('\nMean time ratio vs per-shape best measured (excluding cuBLAS):')
print(d.groupby(['dtype', 'layout'])[[f'r_{m}' for m in METH]].mean().round(3).to_string())
print(d[[f'r_{m}' for m in METH]].mean().round(3).to_string())
print('\nvs cuBLAS (B1): ours5 faster by >tol on', (d.ours5_vs_B1 < 1 - d.tol).sum(), 'shapes; slower by >tol on', (d.ours5_vs_B1 > 1 + d.tol).sum(),
      '; geomean ours5/B1 =', round(np.exp(np.log(d.ours5_vs_B1).mean()), 3), '; best =', round(d.ours5_vs_B1.min(), 3), 'worst =', round(d.ours5_vs_B1.max(), 3))
print('   inductor (B2) faster than cuBLAS by >tol on', (d.B2_vs_B1 < 1 - d.tol).sum(), 'shapes')
print(d.groupby(['dtype', 'layout']).apply(lambda g: pd.Series(dict(n=len(g), ours5_beats_B1=(g.ours5_vs_B1 < 1 - g.tol).sum(),
      geo=np.exp(np.log(g.ours5_vs_B1).mean())))).round(3).to_string())
print('\nTuning cost (median over shapes): ')
print(pd.DataFrame({m: dict(wall_s=d[f'wall_{m}'].median()) for m in ['B2', 'B3', 'ours1', 'ours3', 'ours5', 'nvmmh1', 'nvmmh5']}).round(2).to_string())
print('B2 configs benchmarked (median):', d.b2_nconf.median())
print('\nours5 vs nvmmh5: ours faster by >tol', (d.t_ours5 < d.t_nvmmh5 * (1 - d.tol)).sum(), ' slower by >tol', (d.t_ours5 > d.t_nvmmh5 * (1 + d.tol)).sum())
print('ours1 vs B2 within noise:', (d.t_ours1 / d.t_B2 <= 1 + d.tol).mean().round(3))
json.dump(dict(c1=c1, c2=c2, wall_ratio=c3, kill_frac=kill, verdict=v, n=n), open(f'{R}/runs/h7_verdict.json', 'w'), indent=1)
