import sys; sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from nvmmh_lib import NVMMH, SM
n = NVMMH()
for prec, (M, N, K), lay in [('TST', (8192, 28672, 8192), 'AB'), ('TST', (512, 14336, 4096), 'ABT'), ('QQHSH', (8192, 14336, 4096), 'ABT')]:
    p, res = n.configs(M, N, K, lay, prec, 5)
    print(prec, M, N, K, lay, 'n=', len(res))
    for r in res:
        k = r['kernel']; c = r['nvmmhKernelConfiguration']
        mets = {m.name: n.metric(p, c, m, prec) for m in (SM.L2_HIT_RATE, SM.GMEM_LOAD_BYTES, SM.ENERGY_JOULES, SM.RUNTIME_S)}
        print('  ', k.cta_tile_m, k.cta_tile_n, k.cta_tile_k, 'warp', k.warp_tile_m, k.warp_tile_n, k.warp_tile_k, 'stages', k.stages,
              'swz', k.swizzle_factor, 'order', k.cta_order, 'splitk', k.split_k, f"rt={r['runtime']*1e3:.3f}ms",
              {a: round(b, 4) for a, b in mets.items()})
