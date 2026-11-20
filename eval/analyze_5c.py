"""Amendment 5c verdict: top-5 distinct tiles (ours5d) vs old top-5 variants (ours5) and top-1, each vs stock max-autotune, on the fresh Qwen 2.5 7B set.  python3 eval/analyze_5c.py [tag]"""
import glob, json, sys, numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else 'fresh_qwen'
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'results/{tag}_s*.csv'))])
t = d.groupby(['gemm', 'dtype', 'M', 'method']).ms.median().unstack('method').reset_index()
gm = lambda x: float(np.exp(np.log(x.dropna()).mean()))
out = {}
for dt, g in t.groupby('dtype'):
    o = {}
    for sel in ('ours1', 'ours5', 'ours5d'):
        r = g[sel] / g.ind_auto; sm = g[g.M <= 32]; ro = sm[sel] / sm.oracle
        o[sel] = dict(n=int(r.notna().sum()), geomean=gm(r), worst=float(r.max()), worst_shape=str(g.loc[r.idxmax(), ['gemm', 'M']].tolist()), smallM_worst_vs_oracle=float(ro.max()),
                      smallM_worst_shape=str(sm.loc[ro.idxmax(), ['gemm', 'M']].tolist()), smallM_frac_le_1p10=float((ro <= 1.10).mean()),
                      PASS_i=bool(gm(r) <= 1.03), PASS_ii=bool(r.max() <= 1.15), PASS_iii=bool(ro.max() <= 1.10))
    f = d[(d.dtype == dt)].groupby('method').first_call_s.median(); o['median_first_call_s'] = {k: float(v) for k, v in f.items() if k in ('ours1', 'ours5', 'ours5d', 'ind_auto')}
    out[dt] = o
print(json.dumps(out, indent=1)); json.dump(out, open(f'results/{tag}_summary.json', 'w'), indent=1)
t['r1'] = t.ours1 / t.ind_auto; t['r5'] = t.ours5 / t.ind_auto; t['r5d'] = t.ours5d / t.ind_auto; t['o5d'] = t.ours5d / t.oracle; t['o1'] = t.ours1 / t.oracle
t.to_csv(f'results/{tag}_per_shape.csv', index=False)
