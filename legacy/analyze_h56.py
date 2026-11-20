import numpy as np, pandas as pd, json
from scipy.stats import spearmanr, wilcoxon
R = '/home/min/a/cashman3/lala-triton-killtest'
B = pd.read_csv(f'{R}/runs/phaseB_joined.csv'); cls = pd.read_csv(f'{R}/runs/r2_shape_classes.csv').set_index('shape')
B = B[B.phase.str.startswith('B_')].copy()
B['cycles'] = B.ms_per_gemm * 1e-3 * B.sm_clk_med * 1e6
rng = np.random.default_rng(0)
def boot_med(x):
    x = np.asarray(x); b = [np.median(rng.choice(x, len(x))) for _ in range(10000)]; return np.percentile(b, [2.5, 97.5])
rows, pairs = [], []
for shape, x in B.groupby('shape'):
    m = x.groupby('G').agg(t=('ms_per_gemm', 'mean'), clk=('sm_clk_med', 'mean'), P=('energy_power_W', 'mean'), E=('energy_J_per_gemm', 'mean'),
                           bytes=('bytes', 'mean'), cyc=('cycles', 'mean'), capfrac=('frac_sw_power_cap', 'mean'), cv=('ms_per_gemm', lambda v: v.std() / v.mean()))
    bg = int(cls.loc[shape, 'bestG']); sens = bool(cls.loc[shape, 'sens'])
    rho = spearmanr(m.clk, m.bytes).correlation
    tr = m.t[8] / m.t[bg] if bg in m.index else np.nan
    frac = np.log(m.clk[bg] / m.clk[8]) / np.log(tr) if (bg != 8 and tr > 1) else np.nan
    rows.append(dict(shape=shape, sens=sens, bestG_r1=bg, r1_gain=cls.loc[shape, 'Grange'], sustained_t8_over_tbest=tr,
                     sustained_Grange=m.t.max() / m.t.min(), sustained_bestG=int(m.t.idxmin()), rho_clk_bytes=rho,
                     rho_t_bytes=spearmanr(m.t, m.bytes).correlation, explained_frac=frac, cyc8_over_cycbest=m.cyc[8] / m.cyc[bg],
                     clk8=m.clk[8], clkbest=m.clk[bg], E_best_over_E8=m.E[bg] / m.E[8], capfrac=m.capfrac.mean(), rep_cv=m.cv.mean(),
                     bytes_range=m.bytes.max() / m.bytes.min(), P_mean=m.P.mean()))
    for rep, y in x.groupby('rep'):
        y = y.set_index('G')
        if bg in y.index and 8 in y.index and bg != 8:
            pairs.append(dict(shape=shape, sens=sens, rep=rep, logE=np.log(y.energy_J_per_gemm[bg] / y.energy_J_per_gemm[8]),
                              logT=np.log(y.ms_per_gemm[bg] / y.ms_per_gemm[8])))
d = pd.DataFrame(rows); pr = pd.DataFrame(pairs)
pd.set_option('display.width', 250)
print(d.round(3).to_string())
S = d[d.sens]
r1 = S.rho_clk_bytes.median(); ci1 = boot_med(S.rho_clk_bytes.dropna())
ef = S.explained_frac.dropna(); r2 = ef.median() if len(ef) else np.nan; ci2 = boot_med(ef) if len(ef) else (np.nan, np.nan)
print(f'\nH5(1) median rho(clock, bytes) over {len(S)} G-sensitive shapes = {r1:.3f}  95% CI {ci1.round(3)}')
print(f'H5(2) explained fraction: n usable = {len(ef)} (excluded: {len(S)-len(ef)} with bestG=8 or no sustained speedup); median = {r2:.3f}  CI {np.round(ci2,3)}')
v = 'PASS' if (r1 <= -0.5 and r2 >= 0.5) else ('KILL' if (r1 > -0.2 or (not np.isnan(r2) and r2 < 0.25)) else 'PARTIAL')
print('H5 verdict:', v, '(H5(3) skipped: clocks cannot be locked)')
ps = pr[pr.sens]
fr = (S.E_best_over_E8 <= 0.95).mean()
p = wilcoxon(ps.logE).pvalue if len(ps) > 5 else np.nan
print(f'H6: frac G-sensitive shapes with >=5% less energy at bestG vs G8 = {fr:.2f}; Wilcoxon p (n={len(ps)} pairs) = {p:.4g}; '
      f'median energy ratio = {S.E_best_over_E8.median():.3f}')
print('H6 verdict:', 'PASS' if (fr >= 0.5 and p < 0.05) else 'FAIL')
print(f'\nDiagnostics: sustained G effect (t8/tbest) median over sensitive = {S.sustained_t8_over_tbest.median():.3f}; '
      f'cycles8/cyclesbest median = {S.cyc8_over_cycbest.median():.3f}; controls sustained Grange median = {d[~d.sens].sustained_Grange.median():.3f}')
print('round-1 bestG still best under sustained protocol:', (S.sustained_bestG == S.bestG_r1).sum(), '/', len(S))
d.to_csv(f'{R}/runs/h56_per_shape.csv', index=False); pr.to_csv(f'{R}/runs/h6_pairs.csv', index=False)
json.dump(dict(h5_rho=r1, h5_rho_ci=list(ci1), h5_frac=r2, h5_frac_ci=list(ci2), h5_n_frac=len(ef), h5=v, h6_frac=fr, h6_p=p),
          open(f'{R}/runs/h56_verdict.json', 'w'), indent=1, default=float)
