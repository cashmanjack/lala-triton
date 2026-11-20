"""Phase C: fit the power model on calibration + control runs only; build M4; evaluate H3' on round-1 42-shape timings."""
import sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from ncu_parse import UNIT
import csv
from model import locality_model as m3
from model.time_model import ideal_compute, window, HBM_BW, L2_BW, LAUNCH
R = '/home/min/a/cashman3/lala-triton-killtest'
# --- ncu bytes for phase B / cal configs
pb = pd.read_csv(f'{R}/predictions/phaseB_ncu.csv'); recs = []
for k, _ in pb.groupby(['sweep', 'M', 'N', 'K', 'dtype']):
    base = f'{R}/runs/ncu_phaseB/' + '_'.join(map(str, k)); idx = json.load(open(base + '.manifest.json'))
    rows = [r for r in csv.reader(open(base + '.csv')) if len(r) > 12 and r[0].isdigit()]
    ids = sorted(set(int(r[0]) for r in rows)); assert len(ids) == len(idx)
    for kid, pi in zip(ids, idx):
        mm = {r[12]: float(r[14].replace(',', '')) * UNIT.get(r[13], 1) for r in rows if int(r[0]) == kid}
        recs.append(dict(pb.loc[pi].to_dict(), ncu_read=mm['dram__bytes_read.sum'], ncu_write=mm['dram__bytes_write.sum']))
nb = pd.DataFrame(recs); nb.to_csv(f'{R}/runs/phaseB_ncu_joined.csv', index=False)
B = pd.read_csv(f'{R}/runs/phaseB.csv')
B = B.merge(nb[['M', 'N', 'K', 'dtype', 'tile', 'G', 'ncu_read', 'ncu_write', 'pred_read']], on=['M', 'N', 'K', 'dtype', 'tile', 'G'], how='left')
B['bytes'] = B.ncu_read + B.ncu_write; B['Tsec'] = B.ms_per_gemm * 1e-3; B['F'] = 2.0 * B.M * B.N * B.K
B.to_csv(f'{R}/runs/phaseB_joined.csv', index=False)
fit = B[B.phase.isin(['cal', 'B_ctrl'])].dropna(subset=['bytes'])
P = fit.energy_power_W.values
X = np.column_stack([np.ones(len(fit)), (fit.F / fit.Tsec / 1e12) * (fit.dtype == 'bf16'), (fit.F / fit.Tsec / 1e12) * (fit.dtype == 'fp8'), fit.bytes / fit.Tsec / 1e9])
coef, *_ = np.linalg.lstsq(X, P, rcond=None)
resid = P - X @ coef
P0, a_bf, a_fp, beta = coef
pm = dict(P0=P0, alpha_bf16_W_per_TFLOPs=a_bf, alpha_fp8_W_per_TFLOPs=a_fp, beta_W_per_GBps=beta, n=len(fit), rmse=float(np.sqrt((resid ** 2).mean())),
          r2=float(1 - (resid ** 2).sum() / ((P - P.mean()) ** 2).sum()))
print('power model', {k: round(v, 4) for k, v in pm.items()})
json.dump(pm, open(f'{R}/env/power_model.json', 'w'), indent=1)
# implied: energy per byte vs per flop
print(f'  => {beta*1e3/1e9*1e12:.1f} pJ/byte DRAM, {a_bf:.3f} J/TFLOP bf16 = {a_bf:.3f} pJ/FLOP, fp8 {a_fp:.3f} pJ/FLOP')
# --- eff de-throttled
cal = pd.read_csv(f'{R}/runs/eff_calib.csv'); cal = cal[cal.ok == 1].copy()
cal['ideal'] = [ideal_compute(r.M, r.N, r.K, r.BM, r.BN, int(r.occ), r.dtype) for r in cal.itertuples()]
cal['eff_nt'] = cal.ideal / (cal.ms * 1e-3) * 2505 / cal.sm_clk
eff_nt = cal.groupby(['dtype', 'tile']).eff_nt.median()
P_CAP = 300.0
pred = pd.read_csv(f'{R}/predictions/phase3_pred.csv')
def t_m4(r):
    s = 1 if r.dtype == 'fp8' else 2
    Fl = 2.0 * r.M * r.N * r.K; By = r.m3_read + r.M * r.N * 2
    a = a_bf if r.dtype == 'bf16' else a_fp
    Tp = (a * Fl / 1e12 + beta * By / 1e9) / (P_CAP - P0)
    Tc = ideal_compute(r.M, r.N, r.K, r.BM, r.BN, int(r.occ), r.dtype) / eff_nt[(r.dtype, r.tile)]
    Td = By / HBM_BW; Tl2 = r.M * r.N * r.K * s * (1 / r.BM + 1 / r.BN) / L2_BW
    return max(Tp, Tc, Td, Tl2) + LAUNCH, Tp >= max(Tc, Td, Tl2)
o = [t_m4(r) for r in pred.itertuples()]
pred['t_m4'] = [x[0] for x in o]; pred['m4_power_bound'] = [x[1] for x in o]
pred.to_csv(f'{R}/predictions/phase3_pred_m4.csv', index=False)
print('fraction of (shape,config) predicted power-bound by M4:', pred.m4_power_bound.mean().round(3))
# --- H3' evaluation on round-1 timings
m = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/runs/phase3_gpu*.csv')]); m = m[~m.rep.astype(str).str.startswith('noise')]
med = m.groupby(['shape', 'cfg']).ms.median()
rows = []
for s, p in pred.groupby('shape'):
    t = med.loc[s]; p = p.set_index('cfg').loc[t.index]; oracle = t.min()
    orders = {'M1': p.sort_values(['t_m1', 'm2_tiebreak']).index, 'M1_G8': p[p.G == 8].sort_values(['t_m1', 'm2_tiebreak']).index,
              'M3': p.sort_values(['t_m3', 'm2_tiebreak']).index, 'M4': p.sort_values(['t_m4', 'm2_tiebreak']).index}
    for name, idx in orders.items():
        c = np.minimum.accumulate(t.loc[idx].values) / oracle
        rows.append(dict(shape=s, model=name, **{f'r@{k}': c[k - 1] for k in (1, 3, 5, 10, 20)}))
res = pd.DataFrame(rows); res.to_csv(f'{R}/runs/phaseC_regret.csv', index=False)
print(res.groupby('model')[[f'r@{k}' for k in (1, 3, 5, 10, 20)]].mean().round(3).to_string())
rng = np.random.default_rng(0); NOISE = 0.092
verdict = {}
for k in (1, 3):
    a = res[res.model == 'M1_G8'].set_index('shape')[f'r@{k}']; b = res[res.model == 'M4'].set_index('shape')[f'r@{k}']
    diff = (a - b.loc[a.index]).values
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(10000)]
    verdict[k] = dict(mean_diff=diff.mean(), ci=(np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
    print(f'k={k}: M1_G8 - M4 mean regret diff = {diff.mean():.4f}  95% CI [{verdict[k]["ci"][0]:.4f}, {verdict[k]["ci"][1]:.4f}]')
passed = verdict[1]['mean_diff'] > NOISE and verdict[3]['mean_diff'] > NOISE and verdict[1]['mean_diff'] >= 0.05
print('H3prime PASS' if passed else 'H3prime FAIL (M4 no better)')
m3k1 = res[res.model == 'M3']['r@1'].mean(); m4k1 = res[res.model == 'M4']['r@1'].mean()
choice = 'M4' if (passed or m4k1 < m3k1) else 'M3'
print('model for Phases D/E (pre-registered rule):', choice)
json.dump(dict(passed=bool(passed), choice=choice, verdict={k: dict(mean_diff=v['mean_diff'], ci=list(v['ci'])) for k, v in verdict.items()}),
          open(f'{R}/runs/phaseC_verdict.json', 'w'), indent=1)
