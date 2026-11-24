"""Round 6 Step 1 analysis: what separates fast and slow tiles at small M, and how bad are today's ties.
python3 eval/decode_diag.py   (reads results/r6/decode_sweep_s*.csv, profiles/rtx6000ada.json)"""
import glob, json, sys, os, math, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lala_triton import model as M3, profile as P
from scipy.stats import kendalltau
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results/r6/decode_sweep_s*.csv'))]); d = d[d.ok == 1].reset_index(drop=True).copy()
prof = P.load('profiles/rtx6000ada.json'); n_sm = prof['n_sm']
pr = d.tile.map(M3.parse_tile); d['BM'], d['BN'], d['BK'], d['st'], d['w'] = [pr.map(lambda x, i=i: x[i]) for i in range(5)]
d['s'] = np.where(d.dtype == 'fp8', 1, 2)
d['blocks'] = np.ceil(d.M / d.BM) * np.ceil(d.N / d.BN); d['active'] = np.minimum(d.blocks, n_sm * d.occ)
d['kiter'] = np.ceil(d.K / d.BK); d['infl_blk'] = d.st * (d.BM + d.BN) * d.BK * d.s     # per spec 5e: stages*(BM*BK+BK*BN)*s
d['bytes'] = d.M * d.K * d.s + d.N * d.K * d.s + d.M * d.N * 2
d['t'] = d.ms * 1e-3; d['bw'] = d.bytes / d.t / 1e9
d['pred'] = [M3.predict(r.M, r.N, r.K, dict(BM=r.BM, BN=r.BN, BK=r.BK, G=8, num_warps=r.w, num_stages=r.st), r.dtype, prof, 'ABT', occ=r.occ) for r in d.itertuples()]
d['shape'] = d.dtype + ' ' + d.N.astype(str) + 'x' + d.K.astype(str) + ' M=' + d.M.astype(str)
d.to_csv('results/r6/decode_sweep_enriched.csv', index=False)
print('HBM_BW GB/s', prof['hbm_bw'] / 1e9, ' rows', len(d))
# --- how much of the within-shape log-time spread does each factor explain (R^2 of single-factor fit of log t on log factor, after removing shape means)
d['lt'] = np.log(d['t']); d['lt_c'] = d['lt'] - d.groupby('shape')['lt'].transform('mean')
def r2(cols, df):
    X = np.column_stack([np.log(df[c]) - df.groupby('shape')[c].transform(lambda v: np.log(v).mean()) for c in cols] if False else [np.log(df[c]) - np.log(df[c]).groupby(df['shape']).transform('mean') for c in cols])
    y = df.lt_c.values; b, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum()
d['activef'] = d.active; d['wf'] = d.w
facs = {'active blocks (blocks vs SMs*occ)': ['active'], 'bytes in flight / block': ['infl_blk'], 'BK (K-loop iterations)': ['kiter'], 'warps': ['wf'], 'BN': ['BN'], 'stages': ['st']}
for dt in ('bf16', 'fp8'):
    g = d[d.dtype == dt]; print(f'\n== {dt}: share of within-shape log-time variance explained (single factor / all together)')
    for k, c in facs.items(): print(f'  {k:40s} R2 = {r2(c, g):.2f}')
    print(f'  {"all six together":40s} R2 = {r2(sum(facs.values(), []), g):.2f}')
    print(f'  {"CURRENT MODEL predicted time (log)":40s} R2 = {r2(["pred"], g):.2f}')
    # also by M regime
    for lo, hi in ((1, 4), (8, 16), (32, 64)):
        gg = g[(g.M >= lo) & (g.M <= hi)]; print(f'  M in [{lo},{hi}]: all six R2={r2(sum(facs.values(), []), gg):.2f}; model R2={r2(["pred"], gg):.2f}')
# --- ties under the current model
rows = []
for (dt, sh), g in d.groupby(['dtype', 'shape']):
    for tag, cap in (('5d-cap', True),):
        b = g.pred.min(); tie = g[g.pred <= 1.02 * b]
        tau = kendalltau(g.pred, g.t)[0]
        rows.append(dict(dtype=dt, shape=sh, n_tiles=len(g), n_tied_2pct=len(tie), tied_best=tie.t.min() * 1e3, tied_worst=tie.t.max() * 1e3, tied_worst_over_best=tie.t.max() / tie.t.min(),
                         model_pick_over_best=g.loc[g.pred.idxmin(), 't'] / g.t.min(), best_ms=g.t.min() * 1e3, tau=tau))
T = pd.DataFrame(rows); T.to_csv('results/r6/decode_ties.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('\nTies under the current model (within 2% of its predicted best):'); print(T.round(3).to_string(index=False))
print('\nmedian tied-group count', T.groupby('dtype').n_tied_2pct.median().to_dict(), ' median worst/best in tie', T.groupby('dtype').tied_worst_over_best.median().round(2).to_dict(),
      ' median model-pick/best', T.groupby('dtype').model_pick_over_best.median().round(2).to_dict(), ' median tau', T.groupby('dtype').tau.median().round(2).to_dict())
# --- plots
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('plots/r6', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for a, dt in zip(ax, ('bf16', 'fp8')):
    g = d[(d.dtype == dt) & (d.M <= 16)]
    sc = a.scatter(g.active, g.bw, c=np.log2(g.infl_blk), s=6, cmap='viridis'); a.set_xscale('log'); a.axhline(prof['hbm_bw'] / 1e9, ls='--', c='r', lw=1)
    a.set_title(f'{dt}, M<=16 (BM=16)'); a.set_xlabel('active blocks (min(blocks, SMs*occ))'); plt.colorbar(sc, ax=a, label='log2 bytes in flight per block')
ax[0].set_ylabel('achieved bandwidth GB/s (bytes/time)'); plt.tight_layout(); plt.savefig('plots/r6/bw_vs_active_blocks.png', dpi=110)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, dt in zip(ax, ('bf16', 'fp8')):
    g = d[(d.dtype == dt) & (d.M <= 16)]; a.scatter(g.pred * 1e3, g.t * 1e3, c=np.log2(g.BK), s=5, cmap='plasma'); a.set_xscale('log'); a.set_yscale('log'); a.plot([.01, 1], [.01, 1], 'k:')
    a.set_xlabel('model-predicted ms'); a.set_ylabel('measured ms'); a.set_title(f'{dt}: predicted vs measured, M<=16 (color = log2 BK)')
plt.tight_layout(); plt.savefig('plots/r6/pred_vs_meas.png', dpi=110)
