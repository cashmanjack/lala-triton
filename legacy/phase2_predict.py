import sys, json, numpy as np, pandas as pd, datetime
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from model import locality_model as m3, naive_model as m1
from model.time_model import window
comp = pd.read_csv('runs/compile_all.csv')
TILES = [('128x128x64_w4_s3', 128, 128, 64, 4, 3), ('128x256x64_w8_s3', 128, 256, 64, 8, 3)]
occ = {t[0]: int(comp[(comp.dtype == 'bf16') & (comp.tile == t[0]) & (comp.G == 8)].occ.iloc[0]) for t in TILES}
Ks = sorted(set(int(round(k / 128) * 128) for k in np.geomspace(512, 32768, 20)))
runs = []
for t in TILES:
    for K in Ks:
        for G in (1, 4, 8, 16, 32): runs.append(('ksweep', 8192, 8192, K, t, G))
    for shp, tag in [((16384, 16384, 6144), 'gR1'), ((12288, 12288, 7168), 'gR1'), ((6144, 16384, 8192), 'gR1'),
                     ((8192, 8192, 32768), 'gR2'), ((16384, 16384, 24576), 'gR2'), ((12288, 16384, 32768), 'gR2')]:
        for G in (1, 2, 4, 8, 12, 16, 24, 32, 64): runs.append((tag, *shp, t, G))
rows = []
for tag, M, N, K, t, G in runs:
    name, BM, BN, BK, w, s = t
    cfg = dict(BM=BM, BN=BN, BK=BK, G=G, num_warps=w, num_stages=s)
    P, W = window(M, N, BM, BN, occ[name])
    A, B, i = m3.traffic(M, N, K, BM, BN, G, W, 2)
    kn = m3.knee_K(M, N, BM, BN, G, W, 2)
    rows.append(dict(sweep=tag, M=M, N=N, K=K, tile=name, BM=BM, BN=BN, BK=BK, G=G, num_warps=w, num_stages=s,
                     occ=occ[name], W=W, regime=i['regime'], r=i['r'], c=i['c'], F2=i['F2'], F3=i['F3'],
                     pred_A=A, pred_B=B, pred_read=A + B, naive_read=m1.dram_read_bytes(M, N, K, cfg, 1, 2),
                     compulsory=(M + N) * K * 2, knee_R0R1=kn.get('R0->R1'), knee_R1R2=kn.get('R1->R2'),
                     written=datetime.datetime.now().isoformat()))
df = pd.DataFrame(rows); df.to_csv('predictions/phase2_pred.csv', index=False)
print(len(df)); print(df.groupby(['sweep', 'regime']).size())
print(df[df.sweep != 'ksweep'].groupby(['sweep', 'M', 'N', 'K', 'tile']).regime.agg(lambda x: ''.join(v[1] for v in x)))
