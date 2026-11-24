"""R5-V verdicts for EVERY method present in a run, each vs stock max-autotune (ind_auto) and, at M<=32, vs the restricted oracle.
python3 eval/analyze_5d.py [tag]   (reads results/<tag>_s*.csv; methods absent from the run are skipped)
(i) geomean ratio vs ind_auto <= 1.03; (ii) worst shape <= 1.15; (iii) small-M (M<=32) worst vs oracle <= 1.10."""
import glob, json, sys, numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else 'fresh_gemma'
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'results/{tag}_s*.csv'))])
t = d.groupby(['gemm', 'dtype', 'M', 'method']).ms.median().unstack('method').reset_index()
ALL = ('ind_auto', 'cublas', 'ours1', 'ours1pc', 'ours1e', 'ours1f', 'ours1ef', 'ours1tb', 'ours5d', 'ours_auto', 'tritonblas', 'oracle')
methods = [m for m in ALL if m in t.columns] + [m for m in t.columns if m not in ALL + ('gemm', 'dtype', 'M')]
gm = lambda x: float(np.exp(np.log(x.dropna()).mean()))
out = {}
for dt, g in t.groupby('dtype'):
    o = {}; sm = g[g.M <= 32]
    for sel in methods:
        if sel == 'ind_auto': continue
        r = g[sel] / g.ind_auto
        if r.notna().sum() == 0: continue
        e = dict(n=int(r.notna().sum()), geomean=gm(r), worst=float(r.max()), worst_shape=str(g.loc[r.idxmax(), ['gemm', 'M']].tolist()))
        e.update(PASS_i=bool(e['geomean'] <= 1.03), PASS_ii=bool(e['worst'] <= 1.15))
        if 'oracle' in g.columns and sel != 'oracle':
            ro = sm[sel] / sm.oracle
            if ro.notna().sum():
                e.update(smallM_n=int(ro.notna().sum()), smallM_worst_vs_oracle=float(ro.max()), smallM_worst_shape=str(sm.loc[ro.idxmax(), ['gemm', 'M']].tolist()),
                         smallM_frac_le_1p10=float((ro <= 1.10).mean()), PASS_iii=bool(ro.max() <= 1.10))
        o[sel] = e
    p = d[d.dtype == dt].groupby(['gemm', 'M', 'method']).pick.first().unstack('method')
    if 'ours1' in p and 'ours1pc' in p: o['n_pick_differs_ours1_vs_ours1pc'] = int((p.ours1.fillna('none') != p.ours1pc.fillna('none')).sum())
    out[dt] = o
out['all_pass'] = {m: bool(all(out[dt][m].get(k, False) for dt in out if m in out[dt] for k in ('PASS_i', 'PASS_ii', 'PASS_iii'))) for m in methods if m not in ('ind_auto', 'oracle') and any(m in out[dt] for dt in out)}
print(json.dumps(out, indent=1)); json.dump(out, open(f'results/{tag}_summary.json', 'w'), indent=1)
for m in methods:
    if m != 'ind_auto': t['r_' + m] = t[m] / t.ind_auto
picks = d.groupby(['gemm', 'dtype', 'M', 'method']).pick.first().unstack('method').add_prefix('pick_').reset_index()
t.merge(picks, on=['gemm', 'dtype', 'M']).to_csv(f'results/{tag}_per_shape.csv', index=False)
