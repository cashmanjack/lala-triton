"""Amendment 5d verdict: ours top-1 with BM cap + padded-row charge (ours1pc) vs plain top-1 (ours1), each vs stock max-autotune, on fresh Gemma 2 9B shapes.  python3 eval/analyze_5d.py [tag]"""
import glob, json, sys, numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else 'fresh_gemma'
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'results/{tag}_s*.csv'))])
t = d.groupby(['gemm', 'dtype', 'M', 'method']).ms.median().unstack('method').reset_index()
gm = lambda x: float(np.exp(np.log(x.dropna()).mean()))
out = {}
for dt, g in t.groupby('dtype'):
    o = {}
    for sel in ('ours1', 'ours1pc'):
        r = g[sel] / g.ind_auto; sm = g[g.M <= 32]; ro = sm[sel] / sm.oracle
        o[sel] = dict(n=int(r.notna().sum()), geomean=gm(r), worst=float(r.max()), worst_shape=str(g.loc[r.idxmax(), ['gemm', 'M']].tolist()),
                      smallM_n=int(ro.notna().sum()), smallM_worst_vs_oracle=float(ro.max()), smallM_worst_shape=str(sm.loc[ro.idxmax(), ['gemm', 'M']].tolist()),
                      smallM_frac_le_1p10=float((ro <= 1.10).mean()), PASS_i=bool(gm(r) <= 1.03), PASS_ii=bool(r.max() <= 1.15), PASS_iii=bool(ro.max() <= 1.10))
    p = d[d.dtype == dt].groupby(['gemm', 'M', 'method']).pick.first().unstack('method')
    o['n_pick_differs'] = int((p.ours1.fillna('none') != p.ours1pc.fillna('none')).sum())   # NaN = Inductor recorded no pick (bf16 qkv M=1 lowers without a template)
    out[dt] = o
out['all_pass'] = bool(all(out[dt]['ours1pc'][k] for dt in ('bf16', 'fp8') for k in ('PASS_i', 'PASS_ii', 'PASS_iii')))
print(json.dumps(out, indent=1)); json.dump(out, open(f'results/{tag}_summary.json', 'w'), indent=1)
t['r1'] = t.ours1 / t.ind_auto; t['r1pc'] = t.ours1pc / t.ind_auto; t['o1'] = t.ours1 / t.oracle; t['o1pc'] = t.ours1pc / t.oracle
picks = d.groupby(['gemm', 'dtype', 'M', 'method']).pick.first().unstack('method')[['ours1', 'ours1pc']].add_prefix('pick_').reset_index()
t = t.merge(picks, on=['gemm', 'dtype', 'M']); t.to_csv(f'results/{tag}_per_shape.csv', index=False)
