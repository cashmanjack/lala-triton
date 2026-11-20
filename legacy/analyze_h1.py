import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from ncu_parse import parse_dir
df = parse_dir('runs/ncu_phase2', 'predictions/phase2_pred.csv')
df['err'] = (df.pred_read - df.meas_read).abs() / df.meas_read
df['err_naive'] = (df.naive_read - df.meas_read).abs() / df.meas_read
df['ratio_meas'] = df.meas_read / df.compulsory; df['ratio_pred'] = df.pred_read / df.compulsory
df.to_csv('runs/phase2_joined.csv', index=False)
rho = spearmanr(df.pred_read, df.meas_read).correlation
print(f'N={len(df)}  median abs err M3={df.err.median():.3f}  M1(naive)={df.err_naive.median():.3f}  spearman M3={rho:.3f}  '
      f'M1={spearmanr(df.naive_read, df.meas_read).correlation:.3f}')
print(df.groupby(['sweep', 'regime']).err.median().round(3))
print(df.groupby(['sweep', 'tile']).apply(lambda g: spearmanr(g.pred_read, g.meas_read).correlation).round(3))
# knees (pre-registered rule), per (tile, G) in the K sweep
ks = df[df.sweep == 'ksweep'].sort_values('K'); out = []
for (tile, G), g in ks.groupby(['tile', 'G']):
    levels = g.groupby('regime').ratio_pred.median()   # plateau ratios (K-independent)
    trans = [('R0', 'R1', 'knee_R0R1'), ('R1', 'R2', 'knee_R1R2')]
    for lo, hi, col in trans:
        kpred = g[col].dropna().iloc[0] if g[col].notna().any() else None
        if kpred is None or lo not in levels or hi not in levels: continue
        thr = np.sqrt(levels[lo] * levels[hi])
        above = g[g.ratio_meas > thr]
        # measured knee: smallest K exceeding threshold, considering only K above the previous regime's start
        kmeas = above.K.min() if len(above) else None
        out.append(dict(tile=tile, G=G, trans=f'{lo}->{hi}', thr=thr, K_pred=kpred, K_meas=kmeas,
                        fac=(max(kmeas, kpred) / min(kmeas, kpred)) if kmeas else None))
kn = pd.DataFrame(out); print(kn.round(2).to_string()); kn.to_csv('runs/phase2_knees.csv', index=False)
pd.set_option('display.width', 250)
print(ks[['tile', 'G', 'K', 'regime', 'ratio_pred', 'ratio_meas', 'l2_read_hit']].round(2).to_string())
