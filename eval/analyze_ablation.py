"""Ablation: naive-traffic time model vs M3, both vs stock Inductor max-autotune. Naive run: results/abl_naive_s*.csv (ours1 there = naive, paired with ind_auto in the same run).
M3 numbers come from results/validation.csv (a different run, so clocks differ; shapes whose pick is identical are flagged)."""
import glob, json, numpy as np, pandas as pd
gm = lambda x: float(np.exp(np.log(x.dropna()).mean()))
n = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results/abl_naive_s*.csv'))])
v = pd.read_csv('results/validation.csv')
key = ['gemm', 'dtype', 'M']
def ratios(d, num):
    t = d.groupby(key + ['method']).ms.median().unstack('method'); return (t[num] / t.ind_auto).rename('r')
rn, rm = ratios(n, 'ours1'), ratios(v, 'ours1')
pn = n[n.method == 'ours1'].groupby(key).pick.first(); pm = v[v.method == 'ours1'].groupby(key).pick.first()
df = pd.concat([rm.rename('m3'), rn.rename('naive'), (pn != pm).rename('pick_differs')], axis=1).dropna(subset=['m3', 'naive'])
out = {}
for dt, g in df.groupby(level='dtype'):
    d = g[g.pick_differs]
    out[dt] = dict(n=len(g), n_pick_differs=int(g.pick_differs.sum()), m3_geomean=gm(g.m3), naive_geomean=gm(g.naive), m3_worst=float(g.m3.max()), naive_worst=float(g.naive.max()),
                   naive_worst_shape=str(g.naive.idxmax()), m3_geomean_on_differing=gm(d.m3) if len(d) else None, naive_geomean_on_differing=gm(d.naive) if len(d) else None,
                   naive_frac_within_3pct=float((g.naive <= 1.03).mean()), m3_frac_within_3pct=float((g.m3 <= 1.03).mean()))
print(json.dumps(out, indent=1)); json.dump(out, open('results/ablation_naive.json', 'w'), indent=1); df.to_csv('results/ablation_naive_per_shape.csv')
