"""Fit Amendment 5e (analytical achieved bandwidth + K-loop latency floor) on the Step 1 calibration sweep ONLY.
T = max(T_mem, T_loop, T_comp, T_l2) + LAUNCH; BW_ach = min(HBM, active*inflight_blk/L_mem); T_loop = waves*kiter*(t0/min(max(st-1,1),dmax)); params L_mem, t0, dmax; inflight uses max(st-1,1) (data: st=1 and st=2 behave alike) [3 params/dtype: L_mem, t0, c]
python3 eval/fit_5e.py -> lala_triton/params_5e.json"""
import json, os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lala_triton import model as M3, profile as P
prof = P.load('profiles/rtx6000ada.json'); d = pd.read_csv('results/r6/decode_sweep_enriched.csv')
def pred(df, p, dt, prof=prof):
    L, t0, dmax = np.exp(p)
    s = M3.ESIZE[dt]; n_sm = prof['n_sm']
    blocks = np.ceil(df.M / df.BM) * np.ceil(df.N / df.BN); active = np.minimum(blocks, n_sm * df.occ)
    depth = np.minimum(np.maximum(df.st - 1, 1), dmax); infl = depth * (df.BM + df.BN) * df.BK * s
    bw = np.minimum(prof['hbm_bw'], active * infl / (L * 1e-6))
    t_mem = df.bytes / bw
    waves = np.ceil(blocks / (n_sm * df.occ))
    t_loop = waves * np.ceil(df.K / df.BK) * (t0 * 1e-6 / depth)
    t_comp = np.ceil(blocks / n_sm) * 2 * df.BM * df.BN * df.K / (M3.PEAK[dt] / n_sm * 0.5)   # eff~0.5 placeholder only matters if compute bound (never at M<=64)
    t_l2 = (np.ceil(df.M / df.BM) * df.BM * df.K * s * np.ceil(df.N / df.BN) + df.N * df.K * s * np.ceil(df.M / df.BM)) / prof['l2_bw']
    return np.maximum.reduce([t_mem, t_loop, t_comp, t_l2]) + M3.LAUNCH
out = {}
for dt in ('bf16', 'fp8'):
    g = d[d.dtype == dt]; y = np.log(g.t.values)
    f = lambda p: np.sqrt(np.mean((np.log(pred(g, p, dt)) - y) ** 2))
    best = None
    for l0 in (-1, 0, 2):
        for t0 in (-1, 0, 1):
            r = minimize(f, [l0, t0, 0.7], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6, maxiter=2000))
            if best is None or r.fun < best.fun: best = r
    out[dt] = dict(L_mem_us=float(np.exp(best.x[0])), t0_us=float(np.exp(best.x[1])), depth_max=float(np.exp(best.x[2])), rms_log=float(best.fun))
    g = g.assign(p5e=pred(g, best.x, dt))
    taus = [kendalltau(h.p5e, h.t)[0] for _, h in g.groupby('shape')]; taus0 = [kendalltau(h.pred, h.t)[0] for _, h in g.groupby('shape')]
    rms0 = float(np.sqrt(np.mean((np.log(g.pred) - np.log(g.t)) ** 2)))
    pick = lambda col: np.array([h.t.iloc[h[col].values.argmin()] / h.t.min() for _, h in g.groupby('shape')])
    out[dt].update(rms_log_current=rms0, tau_5e_median=float(np.median(taus)), tau_current_median=float(np.median(taus0)),
                   pick_over_best_5e_median=float(np.median(pick('p5e'))), pick_over_best_5e_worst=float(pick('p5e').max()),
                   pick_over_best_cur_median=float(np.median(pick('pred'))), pick_over_best_cur_worst=float(pick('pred').max()))
    print(dt, json.dumps(out[dt], indent=1))
json.dump(out, open('lala_triton/params_5e.json', 'w'), indent=1)
