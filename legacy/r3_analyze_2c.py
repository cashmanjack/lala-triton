"""Round-3 Step 2c analysis: H10/H11, gain decomposition, tuning cost, 2a timing check (ii)."""
import json, glob, re, numpy as np, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
from r3_lib import parse_ind_line, ind_line_ms
from common import cfg_key
gm = lambda x: float(np.exp(np.log(np.asarray(x, float)).mean()))
ev = pd.read_csv(f'{R}/runs/r3_2c_eval.csv'); cor = pd.read_csv(f'{R}/runs/r3_2c_correct.csv')
agg = ev.groupby(['shape', 'dtype', 'layout', 'launcher', 'methods']).agg(
    ms=('ms_per_gemm', 'mean'), cv=('ms_per_gemm', lambda s: s.std() / s.mean()), n=('rep', 'size'),
    clk=('sm_clk_med', 'mean'), pw=('power_mean', 'mean')).reset_index()
rows = []
for sk, g in agg.groupby('shape'):
    def t(pred):
        x = g[g.methods.apply(lambda m: any(pred(z) for z in m.split('|')))]
        return (x.ms.min(), x.loc[x.ms.idxmin()]) if len(x) else (np.nan, None)
    r = dict(shape=sk, dtype=g['dtype'].iloc[0], layout=g['layout'].iloc[0])
    for name, pred in [('S1', lambda z: z == 'S1'), ('S2', lambda z: z.startswith('S2c')), ('S3', lambda z: z.startswith('S2c') or z.startswith('S3c')),
                       ('S4_1', lambda z: z == 'S4_1'), ('S4_3', lambda z: z == 'S4_3'), ('S4_5', lambda z: z == 'S4_5'),
                       ('S5_1', lambda z: z == 'S5_1'), ('S5_3', lambda z: z == 'S5_3'), ('S5_5', lambda z: z == 'S5_5'), ('S6', lambda z: z == 'S6'), ('S6np', lambda z: z == 'S6np')]:
        v, row = t(pred)
        r[name] = v
        if row is not None: r[f'{name}_cv'] = row.cv; r[f'{name}_clk'] = row.clk; r[f'{name}_launcher'] = row.launcher
    rows.append(r)
d = pd.DataFrame(rows)
d['search_gain'] = d.S2 / d.S3; d['model_gain'] = d.S1 / d.S4_5; d['G_contrib'] = d.S5_5 / d.S4_5
d['S6_pass'] = d.S6.notna(); d['S6'] = d.S6.fillna(d.S6np)
d['S4_over_S1'] = d.S4_5 / d.S1; d['S4_over_S6'] = d.S4_5 / d.S6; d['S1_over_S6'] = d.S1 / d.S6; d['S3_over_S1'] = d.S3 / d.S1
d['S4_1_over_S1'] = d.S4_1 / d.S1; d['S4_over_S3'] = d.S4_5 / d.S3
# tuning
tun = {}
for p in glob.glob(f'{R}/runs/r3_tune/*.json'):
    j = json.load(open(p)); tun[(j['shape'], f"{j['mode']}{j['k']}")] = j
for m in ('S10', 'S41', 'S43', 'S45', 'S51', 'S53', 'S55'):
    d[f'wall_{m}'] = [tun.get((s, m), {}).get('wall_s') for s in d['shape']]
    d[f'ncfg_{m}'] = [tun.get((s, m), {}).get('n_configs') for s in d['shape']]
d['S1_pick'] = [tun.get((s, 'S10'), {}).get('top', '') for s in d['shape']]
d['S1_pick_kind'] = ['extern' if 'BLOCK_M' not in str(x) else 'triton' for x in d.S1_pick]
d['S1_short_ptr'] = [f"{tun.get((s, 'S10'), {}).get('n_tmpl_short_ptr')}/{tun.get((s, 'S10'), {}).get('n_tmpl_llir')}" for s in d['shape']]
d.to_csv(f'{R}/runs/r3_2c_per_shape.csv', index=False)
out = {}
for fam, sel in [('fp8', d.dtype == 'fp8'), ('bf16', d.dtype == 'bf16'), ('bf16_AB', (d.dtype == 'bf16') & (d.layout == 'AB')),
                 ('bf16_ABT', (d.dtype == 'bf16') & (d.layout == 'ABT')), ('all', d.dtype.notna())]:
    x = d[sel].dropna(subset=['S1', 'S4_5'])
    if not len(x): continue
    within = x.S4_over_S1 <= 1 + np.maximum(0.03, x.S1_cv)
    faster = x.S4_over_S1 <= 0.95
    wr = np.nanmedian(x.wall_S45.astype(float)) / np.nanmedian(x.wall_S10.astype(float))
    res = dict(n=len(x), frac_within_noise=float(within.mean()), frac_5pct_faster=float(faster.mean()), wall_ratio=float(wr),
               wall_S1_med=float(np.nanmedian(x.wall_S10.astype(float))), wall_S45_med=float(np.nanmedian(x.wall_S45.astype(float))),
               geomean_S4_over_S1=gm(x.S4_over_S1), geomean_S4_1_over_S1=gm(x.S4_1_over_S1), geomean_S3_over_S1=gm(x.S3_over_S1),
               geomean_S4_over_S6=gm(x.S4_over_S6.dropna()), geomean_S1_over_S6=gm(x.S1_over_S6.dropna()),
               med_search_gain=float(x.search_gain.median()), med_model_gain=float(x.model_gain.median()), med_G_contrib=float(x.G_contrib.median()),
               n_S3_5pct_better_than_S2=int((x.S3 <= 0.95 * x.S2).sum()), geomean_S4_over_S3=gm(x.S4_over_S3),
               S1_picks_extern=int((x.S1_pick_kind == 'extern').sum()))
    if fam in ('fp8', 'bf16'):
        res['PASS'] = bool(res['frac_within_noise'] >= 0.8 and res['frac_5pct_faster'] >= 0.2 and wr <= 1 / 3)
    out[fam] = res
    print(fam, json.dumps(res, indent=1))
# tuning-cost table
tc = d.groupby('dtype')[[c for c in d.columns if c.startswith('wall_') or c.startswith('ncfg_')]].median()
print('\ntuning cost (median per shape):\n', tc.T.round(2).to_string())
scr = pd.read_csv(f'{R}/runs/r3_screen.csv')
print('screening (oracle) configs per shape:', scr.groupby('shape').size().describe().round(0).to_dict())
# 2a check (ii): harness screening time vs Inductor's own S1 autotune-logged time for the same (G=8) config
chk = []
sc = scr.set_index(['shape', 'cfg']).ms
for (s, m), j in tun.items():
    if m != 'S10': continue
    for l in j.get('table', []):
        c = parse_ind_line(l)
        if 'extern' in c: continue
        k = cfg_key(c)
        if (s, k) in sc.index: chk.append(dict(shape=s, cfg=k, ind_ms=ind_line_ms(l), h_ms=sc[(s, k)]))
chk = pd.DataFrame(chk)
if len(chk):
    chk['lr'] = np.log(chk.h_ms / chk.ind_ms)
    out['check_ii'] = dict(n=len(chk), n_shapes=int(chk['shape'].nunique()), median_abs_log_ratio=float(chk.lr.abs().median()),
                           median_log_ratio=float(chk.lr.median()), PASS=bool(chk.lr.abs().median() <= 0.05))
    print('\n2a check (ii):', out['check_ii'])
    chk.to_csv(f'{R}/runs/r3_verify_timing.csv', index=False)
print('\nS1 short-ptr state (template kernels in short-ptr state / total):', d.S1_short_ptr.value_counts().head(8).to_dict())
print('\nper shape:\n', d[['shape', 'S1', 'S2', 'S3', 'S4_1', 'S4_5', 'S5_5', 'S6', 'search_gain', 'model_gain', 'G_contrib', 'S4_over_S6', 'S1_pick_kind',
                         'S4_5_clk', 'S6_clk']].round(4).to_string())
json.dump(out, open(f'{R}/runs/r3_h10_h11_verdict.json', 'w'), indent=1)
