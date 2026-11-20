import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '/home/min/a/cashman3/lala-triton-killtest'
F = pd.read_csv(f'{R}/runs/phaseE_final.csv')
t = F.groupby(['N', 'K', 'M', 'launcher', 'method']).ms.median().reset_index()
rows = []
for r in t.itertuples():
    for m in r.method.split('|'): rows.append(dict(N=r.N, K=r.K, M=r.M, method=m, ms=r.ms))
T = pd.DataFrame(rows).pivot_table(index=['N', 'K', 'M'], columns='method', values='ms')
T.to_csv(f'{R}/runs/phaseE_times.csv')
out = []
for (N, K), x in T.groupby(level=[0, 1]):
    g = {m: float(np.exp(np.log(x[m]).mean())) for m in x.columns if x[m].notna().all()}
    gap = g['bucket'] - g['oracle']
    closed = (g['bucket'] - g['model']) / gap if gap > 0 else np.nan
    out.append(dict(N=N, K=K, **{f'geo_{m}': v for m, v in g.items()}, closed=closed,
                    model_over_oracle=g['model'] / g['oracle'], bucket_over_oracle=g['bucket'] / g['oracle'], static_over_oracle=g['static'] / g['oracle']))
o = pd.DataFrame(out); print(o.round(4).to_string())
npass = int((o.closed >= 0.5).sum())
print(f'H8: pairs closing >=50% of bucket->oracle gap: {npass}/3 ->', 'PASS' if npass >= 2 else 'FAIL')
print('\nper-M ratio to oracle:'); print((T.div(T.oracle, axis=0)).round(3).to_string())
json.dump(dict(npass=npass, table=o.to_dict('records')), open(f'{R}/runs/h8_verdict.json', 'w'), indent=1, default=float)
C = {'static': '#888888', 'bucket': '#eb6834', 'model': '#2a78d6', 'oracle': '#1baf7a', 'inductor': '#eda100'}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6', 'font.size': 8})
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, ((N, K), x) in zip(axs, T.groupby(level=[0, 1])):
    x = x.droplevel([0, 1])
    for m in ['static', 'bucket', 'model', 'inductor']:
        if m in x: ax.plot(x.index, x[m] / x.oracle, marker='o', ms=3, color=C[m], label={'model': 'ours (M3 top-1, no benchmarking)'}.get(m, m))
    ax.axhline(1, color=C['oracle'], lw=1.5, label='per-M oracle (restricted)')
    ax.set_xscale('log', base=2); ax.set_xlabel('M (tokens)'); ax.set_title(f'A@B.T bf16, N={N}, K={K}'); ax.set_ylim(0.7, None)
axs[0].set_ylabel('time / per-M oracle'); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{R}/plots/r2_dynamic_M.png', dpi=130)
