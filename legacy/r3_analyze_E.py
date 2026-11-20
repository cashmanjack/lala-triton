"""Round-3 Step 2d analysis: H12 per dtype, per-(N, K) geomeans over M."""
import json, numpy as np, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
gm = lambda x: float(np.exp(np.log(np.asarray(x, float)).mean()))
ev = pd.read_csv(f'{R}/runs/r3_E_eval.csv')
a = ev.groupby(['dtype', 'N', 'K', 'M', 'launcher', 'methods']).agg(ms=('ms_per_gemm', 'mean'), clk=('sm_clk_med', 'mean')).reset_index()
rows = []
for (dt, N, K, M), g in a.groupby(['dtype', 'N', 'K', 'M']):
    r = dict(dtype=dt, N=N, K=K, M=M)
    for meth in ('static', 'bucket', 'ours', 'inductor_dyn'):
        x = g[g.methods.str.split('|').apply(lambda l: meth in l)]
        r[meth] = x.ms.min() if len(x) else np.nan
    x = g[g.methods.str.contains('oracle_c')]
    r['oracle'] = x.ms.min()
    r['best_any'] = g.ms.min()
    rows.append(r)
d = pd.DataFrame(rows); d.to_csv(f'{R}/runs/r3_E_per_point.csv', index=False)
out = {}
for dt, gd in d.groupby('dtype'):
    pairs = []
    for (N, K), g in gd.groupby(['N', 'K']):
        tb, tc, td, ts, ti = gm(g.bucket), gm(g.ours), gm(g.oracle), gm(g.static), gm(g.inductor_dyn.dropna())
        clos = (tb - tc) / (tb - td) if tb > td else None
        pairs.append(dict(N=int(N), K=int(K), n_M=len(g), static_over_oracle=ts / td, bucket_over_oracle=tb / td, ours_over_oracle=tc / td,
                          inductor_dyn_over_oracle=ti / td, closure=clos, ours_beats_oracle_points=int((g.ours < g.oracle).sum())))
    npass = sum(1 for p in pairs if p['closure'] is not None and p['closure'] >= 0.5)
    out[dt] = dict(pairs=pairs, n_pass=npass, n_na=sum(p['closure'] is None for p in pairs), PASS=npass >= 2)
    print(dt, json.dumps(out[dt], indent=1))
json.dump(out, open(f'{R}/runs/r3_h12_verdict.json', 'w'), indent=1)
