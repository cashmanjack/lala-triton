"""Amendment 5f: build the decode table from the Step 1 sweep and cross-validate leave-one-(N,K)-out on the calibration shapes.
python3 eval/fit_5f.py -> profiles/rtx6000ada_r6.json (= rtx6000ada.json + profile['decode']), results/r6/fit_5f.json"""
import json, os, sys, math, numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lala_triton import model as M3, profile as P, decode as D
prof = P.load('profiles/rtx6000ada.json'); n_sm = prof['n_sm']
d = pd.read_csv('results/r6/decode_sweep_enriched.csv'); d['waves'] = np.ceil(d.blocks / (n_sm * d.occ))
Q = M3._P5E
def e5(g):
    q = g.dtype.map(lambda x: Q[x]['t_iter_us']); ds = g.dtype.map(lambda x: Q[x]['d_sat'])
    return np.maximum(g.pred, g.waves * g.kiter * q * 1e-6 / np.minimum(np.maximum(g.st - 1, 1), ds))
d['p5e'] = e5(d)
def rows_of(g): return g.to_dict('records')
def pred5f(g, table, fallback_col):
    out = []; D.FALLBACK.clear(); st = {'ok': 0, 'fallback': 0}
    for r in g.itertuples():
        v = D.predict(r.M, r.N, r.K, r.BM, r.BN, r.BK, r.tile, r.dtype, r.occ, r.bytes, prof, table=table)
        if v is None: v = getattr(r, fallback_col); st['fallback'] += 1
        else: st['ok'] += 1
        out.append(v)
    return np.array(out), st
def metrics(g, col, TIE=0.005):
    a, tau = [], []
    for _, h in g.groupby('shape'):
        v = h[col].values; t = h.t.values; best = t.min(); a.append(t[v <= v.min() * (1 + TIE)].mean() / best)
        nb = t <= 1.3 * best
        if nb.sum() > 3: tau.append(kendalltau(v[nb], t[nb])[0])
    a = np.array(a); return dict(median=float(np.median(a)), mean=float(a.mean()), worst=float(a.max()), tau=float(np.nanmedian(tau)), n=len(a))
res = {}
for dt in ('bf16', 'fp8'):
    g = d[d.dtype == dt].copy(); nk = sorted({(n, k) for n, k in zip(g.N, g.K)}); parts = []
    for held in nk:
        tr = g[~((g.N == held[0]) & (g.K == held[1]))]; te = g[(g.N == held[0]) & (g.K == held[1])].copy()
        tab = D.fit_table(rows_of(tr), n_sm)
        te['p5f_cur'], s1 = pred5f(te, tab, 'pred'); te['p5f_5e'], s2 = pred5f(te, tab, 'p5e'); te['frac_fb'] = s2['fallback'] / len(te); parts.append(te)
    cv = pd.concat(parts)
    res[dt] = dict(current=metrics(cv, 'pred'), e5=metrics(cv, 'p5e'), f_fallback_current=metrics(cv, 'p5f_cur'), f_fallback_5e=metrics(cv, 'p5f_5e'), fallback_fraction=float(cv.frac_fb.mean()),
                   rms_log_current=float(np.sqrt(np.mean(np.log(cv.pred / cv.t) ** 2))), rms_log_5e=float(np.sqrt(np.mean(np.log(cv.p5e / cv.t) ** 2))),
                   rms_log_5f_5e=float(np.sqrt(np.mean(np.log(cv.p5f_5e / cv.t) ** 2))))
    print(dt, json.dumps(res[dt], indent=1))
json.dump(res, open('results/r6/fit_5f.json', 'w'), indent=1)
prof['decode'] = D.fit_table(rows_of(d), n_sm); P.save(prof, 'profiles/rtx6000ada_r6.json')
print('decode table tiles', {k: len(v) for k, v in prof['decode'].items()}, 'profile size KB', os.path.getsize('profiles/rtx6000ada_r6.json') // 1024, 'vs', os.path.getsize('profiles/rtx6000ada.json') // 1024)
