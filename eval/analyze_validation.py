"""Pre-registered R5-V checks from results/validation.csv. Prints verdicts and writes results/validation_summary.json + per-shape csv."""
import sys, json, math
import pandas as pd, numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'results/validation.csv'
d = pd.read_csv(path)
t = d.groupby(['model', 'gemm', 'dtype', 'M', 'method']).ms.median().unstack('method').reset_index()
t['r_ind'] = t.ours1 / t.ind_auto
t['r_oracle'] = t.ours1 / t.oracle
t['r5_ind'] = t.ours5 / t.ind_auto
t['r_cublas'] = t.ours1 / t.cublas
t['r_default'] = t.ours1 / t.ind_default
gm = lambda x: float(np.exp(np.log(x.dropna()).mean())) if x.notna().any() else float('nan')
out = {}
for dt, g in t.groupby('dtype'):
    small = g[g.M <= 32]
    out[dt] = dict(n=int(len(g)), geomean_ours1_vs_ind_auto=gm(g.r_ind), max_ratio=float(g.r_ind.max()), worst_shape=str(g.loc[g.r_ind.idxmax(), ['gemm', 'M']].tolist()),
                   geomean_ours5_vs_ind_auto=gm(g.r5_ind), geomean_ours1_vs_cublas=gm(g.r_cublas), geomean_ours1_vs_ind_default=gm(g.r_default),
                   small_M_n=int(small.r_oracle.notna().sum()), small_M_max_vs_oracle=float(small.r_oracle.max()), small_M_frac_le_1p10=float((small.r_oracle <= 1.10).mean()),
                   small_M_worst=str(small.loc[small.r_oracle.idxmax(), ['gemm', 'M']].tolist()) if small.r_oracle.notna().any() else '')
    o = out[dt]
    o['PASS_geomean_le_1.03'] = bool(o['geomean_ours1_vs_ind_auto'] <= 1.03)
    o['PASS_max_le_1.15'] = bool(o['max_ratio'] <= 1.15)
    o['PASS_smallM_all_le_1.10'] = bool(o['small_M_max_vs_oracle'] <= 1.10)
t.to_csv('results/validation_per_shape.csv', index=False)
json.dump(out, open('results/validation_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
