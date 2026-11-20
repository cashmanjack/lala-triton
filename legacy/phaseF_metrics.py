"""NVMMH predictions (GMEM load bytes, L2 hit rate, runtime) for OUR configs on round-1 ncu points. Run with venv python."""
import sys, pandas as pd
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
from nvmmh_lib import NVMMH, SM
n = NVMMH()
KC = n.h.nvmmhKernelConfiguration
def kcfg(BM, BN, BK, G, w, s, dt):
    c = KC()
    c.cta[0], c.cta[1], c.cta[2] = BM, BN, BK
    wm, wn = (BM // 2, BN // 2) if w == 4 else (BM // 2, BN // 4)
    c.warp[0], c.warp[1], c.warp[2] = max(wm, 16), max(wn, 8), BK
    c.instr[0], c.instr[1], c.instr[2] = 16, 8, (32 if dt == 'fp8' else 16)
    c.splitK = 1; c.loadStages = s; c.gridSwizzle = 1 if G == 1 else G; c.ctaOrder = 0 if G == 1 else 1
    c.cluster[0] = c.cluster[1] = 1
    return c
out = []
for src, f in [('phase2', 'runs/phase2_joined.csv'), ('phase3', 'runs/phase3_ncu_joined.csv')]:
    d = pd.read_csv(f)
    if 'dtype' not in d: d['dtype'] = 'bf16'
    for r in d.itertuples():
        prec = 'QQHSH' if r.dtype == 'fp8' else 'TST'; lay = 'ABT' if r.dtype == 'fp8' else 'AB'
        p = n.problem(int(r.M), int(r.N), int(r.K), lay)
        c = kcfg(int(r.BM), int(r.BN), int(r.BK), int(r.G), int(r.num_warps), int(r.num_stages), r.dtype)
        out.append(dict(src=src, idx=r.Index, M=r.M, N=r.N, K=r.K, dtype=r.dtype, BM=r.BM, BN=r.BN, BK=r.BK, G=r.G,
                        num_warps=r.num_warps, num_stages=r.num_stages,
                        nv_gmem_load=n.metric(p, c, SM.GMEM_LOAD_BYTES, prec), nv_l2_hit=n.metric(p, c, SM.L2_HIT_RATE, prec),
                        nv_l2_load=n.metric(p, c, SM.L2_LOAD_BYTES, prec), nv_runtime=n.metric(p, c, SM.RUNTIME_S, prec)))
pd.DataFrame(out).to_csv('predictions/phaseF_nvmmh_metrics.csv', index=False); print(len(out))
