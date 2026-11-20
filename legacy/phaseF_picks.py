import sys, pandas as pd
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from nvmmh_lib import NVMMH
from phaseD_shapes import SHAPES, key
n = NVMMH(); rows = []
for (M, N, K, dt, lay) in SHAPES:
    prec = 'QQHSH' if dt == 'fp8' else 'TST'
    p, res = n.configs(M, N, K, lay, prec, 20)
    for rank, r in enumerate(res):
        k = r['kernel']
        BM, BN, BK = k.cta_tile_m, k.cta_tile_n, k.cta_tile_k
        warps = (BM // k.warp_tile_m) * (BN // k.warp_tile_n)
        if k.split_k != 1: G, why = None, 'splitk'
        elif k.swizzle_factor == 1: G, why = 1, ''
        elif k.cta_order == 1: G, why = k.swizzle_factor, ''
        else: G, why = None, 'N-grouping'
        rows.append(dict(shape=key(M, N, K, dt, lay), rank=rank, BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=k.stages, G=G,
                         swz=k.swizzle_factor, order=k.cta_order, splitk=k.split_k, unmappable=why, nv_runtime=r['runtime']))
d = pd.DataFrame(rows); d.to_csv('predictions/phaseF_nvmmh_picks.csv', index=False)
print(len(d), d.unmappable.value_counts().to_dict()); print(d.groupby('shape').size().describe())
print(d[d['rank'] == 0][['shape', 'BM', 'BN', 'BK', 'num_warps', 'num_stages', 'G', 'swz', 'order']].head(12).to_string())
print('distinct swizzles', sorted(d.swz.unique()), 'warps', sorted(d.num_warps.unique()), 'stages', sorted(d.num_stages.unique()))
