"""Amendment 5b verdict on the fresh held-out set (Llama 3 70B TP=8). Criteria = R5-V (i)-(iii), plain argmin (ours1) and tie-break (ours1tb) side by side."""
import glob, json, sys, numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else 'fresh70b_tp8'
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'results/{tag}_s*.csv'))])
t = d.groupby(['gemm', 'dtype', 'M', 'method']).ms.median().unstack('method').reset_index()
gm = lambda x: float(np.exp(np.log(x.dropna()).mean()))
out = {}
for sel in ('ours1', 'ours1tb'):
    t[f'r_{sel}'] = t[sel] / t.ind_auto; t[f'o_{sel}'] = t[sel] / t.oracle
for dt, g in t.groupby('dtype'):
    o = {}
    for sel in ('ours1', 'ours1tb'):
        sm = g[g.M <= 32]
        o[sel] = dict(geomean=gm(g[f'r_{sel}']), worst=float(g[f'r_{sel}'].max()), worst_shape=str(g.loc[g[f'r_{sel}'].idxmax(), ['gemm', 'M']].tolist()),
                      smallM_worst_vs_oracle=float(sm[f'o_{sel}'].max()), smallM_frac_le_1p10=float((sm[f'o_{sel}'] <= 1.10).mean()),
                      PASS_i=bool(gm(g[f'r_{sel}']) <= 1.03), PASS_ii=bool(g[f'r_{sel}'].max() <= 1.15), PASS_iii=bool(sm[f'o_{sel}'].max() <= 1.10))
    o['n_pick_differs'] = int((d[d.method == 'ours1'].groupby(['gemm', 'dtype', 'M']).pick.first().xs(dt, level='dtype') != d[d.method == 'ours1tb'].groupby(['gemm', 'dtype', 'M']).pick.first().xs(dt, level='dtype')).sum())
    out[dt] = o
print(json.dumps(out, indent=1)); json.dump(out, open(f'results/{tag}_summary.json', 'w'), indent=1); t.to_csv(f'results/{tag}_per_shape.csv', index=False)
