"""Round 6 verdicts. python3 eval/analyze_r6.py  (reads results/r6/r6_fresh_s*.csv and *_oraclepool.csv)
Per dtype and method: (i) geomean vs ind_auto <= 1.03, (ii) worst <= 1.15, (iii) M<=32 worst vs oracle <= 1.10, (iv) bf16 no-regression vs ours1pc, (vi) 5g tuning time; plus Kendall tau on the oracle pool (M<=32)."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import kendalltau
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results/r6/r6_fresh*_s?.csv'))])
t = d.groupby(['model', 'gemm', 'dtype', 'M', 'method']).ms.median().unstack('method').reset_index()
# 5e/5f/5ef equal ours1pc by construction at M>64 (offline-verified); fill so every method covers all shapes
for m in ('ours1e', 'ours1f', 'ours1ef'):
    if m in t: t[m] = t[m].fillna(t.ours1pc.where(t.M > 64))
gm = lambda x: float(np.exp(np.log(x.dropna()).mean()))
M_ = ['ours1', 'ours1pc', 'ours1e', 'ours1f', 'ours1ef', 'ours5d', 'ours_auto', 'tritonblas', 'cublas']
out = {}
for dt, g in t.groupby('dtype'):
    sm = g[g.M <= 32]; o = {}
    for m in M_:
        if m not in g or g[m].notna().sum() == 0: continue
        r = g[m] / g.ind_auto; e = dict(n=int(r.notna().sum()), geomean=gm(r), worst=float(r.max()), worst_shape=str(g.loc[r.idxmax(), ['model', 'gemm', 'M']].tolist()))
        e.update(PASS_i=bool(e['geomean'] <= 1.03), PASS_ii=bool(e['worst'] <= 1.15))
        if 'oracle' in g and m != 'cublas':
            ro = sm[m] / sm.oracle
            if ro.notna().sum(): e.update(smallM_worst_vs_oracle=float(ro.max()), smallM_worst_shape=str(sm.loc[ro.idxmax(), ['model', 'gemm', 'M']].tolist()), smallM_geomean_vs_oracle=gm(ro), PASS_iii=bool(ro.max() <= 1.10))
        if m not in ('ours1pc', 'ours1', 'cublas', 'tritonblas') and dt == 'bf16':
            e['iv_geomean_minus_ours1pc'] = e['geomean'] - gm(g.ours1pc / g.ind_auto); x = g[m] / g.ours1pc; e['iv_shapes_gt_1.05_vs_ours1pc'] = int((x > 1.05).sum())
        e['PASS_all'] = bool(all(e.get(k, True) for k in ('PASS_i', 'PASS_ii', 'PASS_iii')))
        o[m] = e
    o['oracle_n'] = int(sm.oracle.notna().sum()) if 'oracle' in g else 0
    out[dt] = o
# tuning time (first-call seconds) and 5g benchmark fraction
fc = d[d.rep == 0].groupby(['model', 'gemm', 'dtype', 'M', 'method']).first_call_s.first().unstack('method')
out['tuning_time_geomean_s'] = {m: gm(fc[m]) for m in ('ind_auto', 'ours1pc', 'ours1e', 'ours5d', 'ours_auto') if m in fc}
if 'ours_auto' in fc: out['tuning_time_5g_over_stock_geomean'] = gm(fc.ours_auto / fc.ind_auto)
au = d[(d.method == 'ours_auto') & (d.rep == 0)].auto.dropna().map(json.loads)
out['benchmark_fraction'] = dict(all=float(np.mean([a[5] > 1 for a in au])), M_le_32=float(np.mean([a[5] > 1 for a in au if a[0] <= 32])), n=len(au))
# Kendall tau over oracle pool
pool = pd.concat([pd.read_csv(f) for f in glob.glob('results/r6/r6_fresh*_s?_oraclepool.csv')]) if glob.glob('results/r6/r6_fresh*_s?_oraclepool.csv') else None
if pool is not None:
    pool = pool.drop_duplicates(['model', 'gemm', 'dtype', 'M', 'tile']); out['tau'] = {}
    for (dt, lo, hi), h in [((dt, lo, hi), pool[(pool.dtype == dt) & (pool.M >= lo) & (pool.M <= hi)]) for dt in ('bf16', 'fp8') for lo, hi in ((1, 32), (1, 1), (8, 8), (32, 32))]:
        r = {}
        for p in ('p_cur', 'p_5e', 'p_5f', 'p_5ef'):
            ts = [kendalltau(x[p], x.ms)[0] for _, x in h.groupby(['model', 'gemm', 'M']) if len(x) > 5]; r[p] = float(np.nanmedian(ts)) if ts else None
        r['pool_per_shape_median'] = float(h.groupby(['model', 'gemm', 'M']).size().median()); out['tau'][f'{dt} M in [{lo},{hi}]'] = r
    # tie counts: pool candidates predicted within 2% of the model's best
    out['ties_in_pool_within_2pct_median'] = {p: float(np.median([(x[p] <= 1.02 * x[p].min()).sum() for _, x in pool[pool.M <= 32].groupby(['model', 'gemm', 'dtype', 'M'])])) for p in ('p_cur', 'p_5e', 'p_5f', 'p_5ef')}
print(json.dumps(out, indent=1)); json.dump(out, open('results/r6/r6_summary.json', 'w'), indent=1)
t.to_csv('results/r6/r6_per_shape.csv', index=False)
