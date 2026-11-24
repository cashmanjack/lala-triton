"""Amendment 5e (final form): current model + K-loop latency floor.  T = max(T_current, waves*ceil(K/BK)*t_iter/min(max(stages-1,1), d_sat)).
2 fitted params/dtype (t_iter, d_sat), fitted on the Step 1 calibration sweep ONLY, objective = ranking (expected pick/best over model ties, and Kendall tau on near-best tiles).
python3 eval/fit_5e.py -> lala_triton/params_5e.json"""
import json, os, sys, itertools, numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lala_triton import model as M3, profile as P
prof = P.load('profiles/rtx6000ada.json'); n_sm = prof['n_sm']
d = pd.read_csv('results/r6/decode_sweep_enriched.csv')
d['waves'] = np.ceil(d.blocks / (n_sm * d.occ)); d['dep'] = np.maximum(d.st - 1, 1)
def floor(g, t_iter_us, dsat): return g.waves * g.kiter * t_iter_us * 1e-6 / np.minimum(g.dep, dsat)
def new_pred(g, p): return np.maximum(g.pred, floor(g, *p))
TIE = 0.005
def metrics(g, col):
    """per shape: pick/best with (a) enumeration-order argmin, (b) expectation over the tie group (within 0.5%); tau on near-best (measured <=1.3x best) tiles."""
    a, b, tau = [], [], []
    for _, h in g.groupby('shape'):
        v = h[col].values; t = h.t.values; best = t.min()
        a.append(t[v.argmin()] / best); b.append(t[v <= v.min() * (1 + TIE)].mean() / best)
        nb = t <= 1.3 * best
        if nb.sum() > 3: tau.append(kendalltau(v[nb], t[nb])[0])
    return np.array(a), np.array(b), np.array(tau)
out = {}
for dt in ('bf16', 'fp8'):
    g = d[d.dtype == dt].copy(); best = None
    for ti, ds in itertools.product(np.arange(0.2, 6.01, 0.2), (1, 1.5, 2, 2.5, 3, 4)):
        g['p'] = new_pred(g, (ti, ds)); a, b, tau = metrics(g, 'p'); obj = np.mean(np.log(b)) - 0.1 * np.nanmedian(tau)
        if best is None or obj < best[0]: best = (obj, ti, ds)
    _, ti, ds = best; g['p'] = new_pred(g, (ti, ds))
    a0, b0, tau0 = metrics(g, 'pred'); a1, b1, tau1 = metrics(g, 'p')
    out[dt] = dict(t_iter_us=float(ti), d_sat=float(ds), n_shapes=len(a1),
                   cur=dict(median_pick_enum=float(np.median(a0)), median_pick_tieexp=float(np.median(b0)), mean_pick_tieexp=float(b0.mean()), worst_tieexp=float(b0.max()), tau_nearbest_median=float(np.nanmedian(tau0))),
                   new=dict(median_pick_enum=float(np.median(a1)), median_pick_tieexp=float(np.median(b1)), mean_pick_tieexp=float(b1.mean()), worst_tieexp=float(b1.max()), tau_nearbest_median=float(np.nanmedian(tau1))))
    out[dt]['beats_current'] = bool(out[dt]['new']['median_pick_tieexp'] <= out[dt]['cur']['median_pick_tieexp'] and out[dt]['new']['tau_nearbest_median'] >= out[dt]['cur']['tau_nearbest_median'])
    print(dt, json.dumps(out[dt], indent=1))
json.dump(out, open('lala_triton/params_5e.json', 'w'), indent=1)
