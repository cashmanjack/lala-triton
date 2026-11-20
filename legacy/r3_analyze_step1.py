"""Round-3 Step 1 analysis: correctness envelope, best cuBLAS variant per shape, H9."""
import json, numpy as np, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_rows', 500)
c = pd.read_csv(f'{R}/runs/r3_step1_correct.csv')
t = pd.read_csv(f'{R}/runs/r3_step1_time.csv')
c['variant'] = c.env + ':' + c.launcher.where(~c.launcher.str.startswith('ours'), 'ours')
t['variant'] = t.env + ':' + t.launcher.where(~t.launcher.str.startswith('ours'), 'ours')
NAMES = {'env1:cublas': 'V1', 'env1:cublas_rpr0': 'V1r', 'env1:cublaslt': 'V2', 'env2:cublas': 'V3', 'env2:cublaslt': 'V3lt',
         'env2:nvmath': 'V4', 'env1:ours': 'Ours'}
c['V'] = c.variant.map(NAMES); t['V'] = t.variant.map(NAMES)
out = {}
# ---------------- 1a correctness
print('== 1a: envelope ratio R (pass iff R <= 2), max over shapes, per variant & dtype')
print(c.groupby(['dtype', 'V']).agg(n=('R', 'size'), R_max=('R', 'max'), R_med=('R', 'median'), e_max_max=('e_max', 'max'),
                                     e_F_med=('e_F', 'median'), n_fail=('pass_', lambda s: int((s == 0).sum()))).round(5).to_string())
fails = c[c.pass_ == 0]
print('\nfailures:\n', fails[['shape', 'V', 'maxabs', 'R', 'e_max', 'e_F', 'refmax', 'worst_ref', 'worst_err', 'flag_rpr', 'kernels']].to_string())
r2 = c[c['shape'].str.startswith('128x4096x14336_bf16')]
print('\nround-2 "max error ~6" shapes:\n', r2[['shape', 'V', 'maxabs', 'R', 'e_max', 'e_F', 'refmax', 'worst_ref', 'worst_err', 'flag_rpr', 'kernels']].to_string())
out['correct_fail'] = fails[['shape', 'V', 'R', 'maxabs']].to_dict('records')
# ---------------- timing
m = t.groupby(['shape', 'dtype', 'layout', 'V']).agg(ms=('ms_per_gemm', 'mean'), cv=('ms_per_gemm', lambda s: s.std() / s.mean()),
                                                     clk=('sm_clk_med', 'mean'), pw=('power_mean', 'mean'), cap=('frac_sw_power_cap', 'mean')).reset_index()
passed = c.set_index(['shape', 'V']).pass_
m['pass_'] = [passed.get((s, v), 0) for s, v in zip(m['shape'], m.V)]
cub = m[(m.V != 'Ours') & (m.pass_ == 1)]
best = cub.loc[cub.groupby('shape').ms.idxmin()].set_index('shape')
ours = m[m.V == 'Ours'].set_index('shape')
v1 = m[m.V == 'V1'].set_index('shape')
rows = []
for sk in ours.index:
    if sk not in best.index: continue
    o, b = ours.loc[sk], best.loc[sk]
    rows.append(dict(shape=sk, dtype=o['dtype'], layout=o['layout'], ours_ms=o.ms, ours_ok=o.pass_, best_V=b.V, best_ms=b.ms, V1_ms=v1.ms.get(sk),
                     ratio_best_over_ours=b.ms / o.ms, ratio_V1_over_ours=v1.ms.get(sk) / o.ms, clk_ours=o.clk, clk_best=b.clk, clk_V1=v1.clk.get(sk),
                     cv_ours=o.cv, cv_best=b.cv))
d = pd.DataFrame(rows)
kn = c.set_index(['shape', 'V']).kernels
d['best_kernel'] = [kn.get((s, v)) for s, v in zip(d['shape'], d.best_V)]
d['V1_kernel'] = [kn.get((s, 'V1')) for s in d['shape']]
d.to_csv(f'{R}/runs/r3_step1_per_shape.csv', index=False)
gm = lambda x: float(np.exp(np.log(x).mean()))
def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed); lx = np.log(np.asarray(x))
    bs = np.exp(rng.choice(lx, (n, len(lx))).mean(1)); return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
for dt in ('bf16', 'fp8'):
    x = d[(d.dtype == dt) & (d.ours_ok == 1)]
    if not len(x): continue
    G9 = gm(x.ratio_best_over_ours)
    res = dict(n=len(x), G9=G9, CI=boot(x.ratio_best_over_ours), G_vs_V1=gm(x.ratio_V1_over_ours), CI_V1=boot(x.ratio_V1_over_ours),
               ours_faster_than_best=int((x.ratio_best_over_ours > 1).sum()), ours_5pct_faster=int((x.ratio_best_over_ours >= 1.05).sum()),
               best_V_counts=x.best_V.value_counts().to_dict(), PASS=bool(G9 >= 1.05),
               by_layout={l: gm(g.ratio_best_over_ours) for l, g in x.groupby('layout')},
               clk_ours_med=float(x.clk_ours.median()), clk_best_med=float(x.clk_best.median()))
    out[f'H9_{dt}'] = res
    print(f'\n== H9 {dt}:', json.dumps(res, indent=1))
# per-variant geomean vs V1 (all shapes, timing only) and clocks
piv = m.pivot_table(index='shape', columns='V', values='ms'); clk = m.pivot_table(index='shape', columns='V', values='clk')
for dt in ('bf16', 'fp8'):
    p = piv[piv.index.str.contains(dt)]
    print(f'\n{dt}: geomean time / V1 per variant:', {v: round(gm(p[v] / p['V1']), 4) for v in p.columns if p[v].notna().all()})
    print(f'{dt}: median sustained SM clock per variant:', clk[clk.index.str.contains(dt)].median().round(0).to_dict())
kv = c[c.V.isin(['V1', 'V3', 'V4', 'V2'])].groupby(['dtype', 'V']).kernels.apply(lambda s: s.str.split('_').str[0].value_counts().to_dict())
print('\nkernel family by variant:', kv.to_string())
print('\nper shape:\n', d[['shape', 'ours_ms', 'best_V', 'best_ms', 'ratio_best_over_ours', 'ratio_V1_over_ours', 'clk_ours', 'clk_best', 'best_kernel']].round(4).to_string())
json.dump(out, open(f'{R}/runs/r3_h9_verdict.json', 'w'), indent=1, default=str)
