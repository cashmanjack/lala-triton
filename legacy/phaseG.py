import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from model import locality_model as hard, locality_soft as soft
from model.locality_model import L2_EFF
p2 = pd.read_csv('runs/phase2_joined.csv').assign(src='phase2', dtype='bf16')
p3 = pd.read_csv('runs/phase3_ncu_joined.csv').assign(src='phase3')
d = pd.concat([p2, p3], ignore_index=True)
rec = []
for r in d.itertuples():
    s = 1 if r.dtype == 'fp8' else 2
    Ah, Bh, ih = hard.traffic(r.M, r.N, r.K, r.BM, r.BN, r.G, r.W, s)
    As, Bs, is_ = soft.traffic(r.M, r.N, r.K, r.BM, r.BN, r.G, r.W, s)
    nearF2 = (ih['c'] < ih['Tn']) and abs(ih['F2'] / L2_EFF - 1) <= 0.10
    nearF3 = abs(ih['F3'] / L2_EFF - 1) <= 0.10
    rec.append(dict(hard=Ah + Bh, soft=As + Bs, near=nearF2 or nearF3))
d = pd.concat([d, pd.DataFrame(rec)], axis=1)
d['err_hard'] = (d.hard - d.meas_read).abs() / d.meas_read; d['err_soft'] = (d.soft - d.meas_read).abs() / d.meas_read
out = []
for name, x in [('phase2', d[d.src == 'phase2']), ('phase3', d[d.src == 'phase3']), ('all', d)]:
    nr = x[x.near]
    out.append(dict(set=name, n=len(x), n_near=len(nr), med_hard=x.err_hard.median(), med_soft=x.err_soft.median(),
                    rho_hard=spearmanr(x.hard, x.meas_read).correlation, rho_soft=spearmanr(x.soft, x.meas_read).correlation,
                    near_worst_hard=nr.err_hard.max(), near_worst_soft=nr.err_soft.max(), near_med_hard=nr.err_hard.median(), near_med_soft=nr.err_soft.median()))
print(pd.DataFrame(out).round(3).to_string())
cols = ['src', 'M', 'N', 'K', 'BM', 'BN', 'G', 'W', 'meas_read', 'hard', 'soft', 'err_hard', 'err_soft', 'near']
print('10 worst BEFORE (hard):'); print(d.sort_values('err_hard', ascending=False)[cols].head(10).round(3).to_string())
print('10 worst AFTER (soft):'); print(d.sort_values('err_soft', ascending=False)[cols].head(10).round(3).to_string())
# knees with soft model, phase-2 K sweep: same pre-registered rule using soft plateau levels = hard plateau levels
d.to_csv('runs/phaseG_joined.csv', index=False)
