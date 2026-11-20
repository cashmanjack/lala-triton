"""Dynamic-M plots from EXISTING data. Top: measured, per GEMM, time / best measured method across M (best = oracle where it exists (M<=32), else min of ours/stock/cuBLAS).
Bottom: MODEL-PREDICTED (not timed) cost of using a bucketed table (config chosen for next power-of-two M) vs the exact per-M pick."""
import sys, os, copy, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
v = pd.read_csv('results/validation.csv'); t = v.groupby(['gemm', 'dtype', 'M', 'method']).ms.median().unstack('method').reset_index()
t['best'] = t[['oracle', 'ours1', 'ind_auto', 'cublas']].min(axis=1)
C = dict(ours1='#0072B2', ind_auto='#D55E00', cublas='#999999')
fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), sharey='row')
for j, g in enumerate(['qkv', 'o', 'gate_up', 'down']):
    for i, dt in enumerate(['bf16', 'fp8']):
        pass
fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
for j, g in enumerate(['qkv', 'o', 'gate_up', 'down']):
    for i, dt in enumerate(['bf16', 'fp8']):
        a = ax[i, j]; d = t[(t.gemm == g) & (t.dtype == dt)].sort_values('M')
        for m, lab in (('ours1', 'ours top-1'), ('ind_auto', 'stock max-autotune'), ('cublas', 'cuBLAS')):
            a.plot(d.M, d[m] / d.best, 'o-', c=C[m], label=lab, ms=4)
        a.set_xscale('log', base=2); a.axvline(32.5, c='k', lw=.5, ls=':'); a.set_title(f'{g} {dt}', fontsize=10); a.set_ylim(0.9, 2.0)
        if j == 0: a.set_ylabel('time / best measured')
        if i == 1: a.set_xlabel('M (tokens)')
ax[0, 0].legend(fontsize=8); fig.suptitle('Llama 3 8B, measured (sustained): time relative to best measured method (oracle where it exists, M<=32; left of dotted line)')
fig.tight_layout(); os.makedirs('plots', exist_ok=True); fig.savefig('plots/r5_dynamic_M_measured.png', dpi=120)
# --- predicted bucket cost
from lala_triton.select import select_scored
from lala_triton.profile import load
from lala_triton import model as M3
from lala_triton.models import gemm_shapes
prof = load('profiles/rtx6000ada.json'); Ms = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192]
nxt = lambda m: 1 << (m - 1).bit_length()
fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
for j, (g, (N, K)) in enumerate(gemm_shapes('llama3-8b').items()):
    for i, dt in enumerate(['bf16', 'fp8']):
        r = []
        for M in Ms:
            ex = select_scored(M, N, K, dt, 'ABT', prof, k=1)[0]
            bc = select_scored(nxt(M), N, K, dt, 'ABT', prof, k=1)[0][1]
            r.append(M3.predict(M, N, K, bc, dt, prof, 'ABT') / ex[0])
        a = ax[i, j]; a.plot(Ms, r, 'o-', c='#009E73', ms=3); a.set_xscale('log', base=2); a.set_title(f'{g} {dt}', fontsize=10)
        if j == 0: a.set_ylabel('predicted: bucket / exact per-M')
        if i == 1: a.set_xlabel('M (tokens)')
fig.suptitle('MODEL-PREDICTED (not timed): cost of a power-of-two-bucket table vs exact per-M config'); fig.tight_layout(); fig.savefig('plots/r5_dynamic_M_buckets_predicted.png', dpi=120)
print('ok')
