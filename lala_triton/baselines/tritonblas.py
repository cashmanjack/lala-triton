"""tritonBLAS-STYLE baseline (approximation). Source: Zhang et al., "tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection", arXiv 2512.04226
(AMD MI300X/MI350X). The paper is a sketch, not a spec; this reimplements what it states and fills gaps with the choices listed below. NOT the authors' code, NOT validated against it.

Implemented as published:
  * compute per tile  L_MT = L_MI * ceil(MT/MI_M) * ceil(NT/MI_N) * ceil(KT/MI_K)                       (Alg. 3)
  * waves             N_waves = ceil(ceil(M/MT) * ceil(N/NT) / N_CU)          (Alg. 9; the only tail/wave-quantization term the paper has)
  * cache-tile hit rate  h = 1 - U/R, U = (mt*MT + nt*NT)*KT, R = mt*nt*(MT+NT)*KT, clamped to [0,1], with the cache tile (mt x nt) near-square (Alg. 5/6),
    and h scaled down proportionally if the per-K-step working set exceeds effective cache capacity (prose only in the paper; inert here since the working set is small)
  * memory time       T = Ld_CU * C; L1 = T/R_cache; L_mem = max(L1, (1-h)*T/R_mem + L_lat)                  (Alg. 7, one cache level)
  * tile latency      L_tile = L_mem + L_epilogue + max(L_compute, L_mem) * (ceil(K/KT) - 1); L_total = N_waves * L_tile   (Alg. 8/9)
  * choice            argmin L_total over power-of-two tiles.
Approximations / gaps filled (all mine):
  1. Hardware constants (paper gives none): L_MI from published tensor peak per SM (mma.m16n8k16 bf16, m16n8k32 fp8, whole-SM peak, no sustained eff);
     R_cache = profile L2 bandwidth, R_mem = profile HBM bandwidth (measured by calibrate), L_lat = the profile's fitted latency (so the baseline gets our latency calibration).
  2. The paper has one L1-like and one L2-like level before memory and never maps them; NVIDIA has one shared L2, so one cache level (all active SMs share it).
  3. Cache tile: N_CU_Cache = number of active SMs; the paper suggests floor(sqrt(N)) square tile, used here (11x11 for 142 SMs), clipped to the output-tile grid.
  4. Ld_CU undefined in the paper: taken as (MT+NT)*KT*bytes per K-iteration per CU. 'a' (active CUs) in the epilogue = active SMs; epilogue = a*MT*NT*out_bytes/R_mem.
  5. num_warps / num_stages / GROUP_M are not in the paper's model. Fixed rule: stages=3 if the tile exists in the profile (else nearest <=3, else smallest); warps = the profile's
     warps for that tile; GROUP_M = largest of (8,16,32,4,1) <= the cache-tile row count mt.
  6. Ties are not specified; broken by enumeration order of the shared template space (the same arbitrary rule our selector had before Amendment 5b).
  7. Candidate tiles: our template space (BM,BN,BK in the profile, power of two), not tritonBLAS's own 50-150 tile list.
  8. Stream-K (mentioned in the paper) is not modelled or used.
"""
import math
from .. import model as M3
from ..select import _space

MI = {'bf16': (16, 8, 16), 'fp8': (16, 8, 32)}
GS = (8, 16, 32, 4, 1)

def _L_total(M, N, K, BM, BN, BK, dtype, prof):
    n_sm, s, so = prof['n_sm'], M3.ESIZE[dtype], M3.OUT_ESIZE[dtype]
    mi_m, mi_n, mi_k = MI[dtype]
    L_MI = 2 * mi_m * mi_n * mi_k / (M3.PEAK[dtype] / n_sm)
    L_comp = L_MI * math.ceil(BM / mi_m) * math.ceil(BN / mi_n) * math.ceil(BK / mi_k)
    nm, nn = math.ceil(M / BM), math.ceil(N / BN)
    T_out = nm * nn; C = min(T_out, n_sm); waves = math.ceil(T_out / n_sm)
    q = max(1, int(math.isqrt(C))); mt, nt = min(nm, q), min(nn, q)
    U = (mt * BM + nt * BN) * BK; R = mt * nt * (BM + BN) * BK
    h = min(1.0, max(0.0, 1 - U / R))
    ws = U * s
    if ws > prof['l2_eff']: h *= prof['l2_eff'] / ws
    T = (BM + BN) * BK * s * C
    L_mem = max(T / prof['l2_bw'], (1 - h) * T / prof['hbm_bw'] + prof['lat'][dtype])
    L_epi = C * BM * BN * so / prof['hbm_bw']
    L_tile = L_mem + L_epi + max(L_comp, L_mem) * (math.ceil(K / BK) - 1)
    return waves * L_tile, mt

def select_scored(M, N, K, dtype, layout, prof, k=1):
    """Returns [(L_total, cfg)] best first (cfg in the harness format BM,BN,BK,G,num_warps,num_stages)."""
    groups = {}
    for t in _space(prof, dtype, layout): groups.setdefault((t['BM'], t['BN'], t['BK']), []).append(t)
    out = []
    for (BM, BN, BK), ts in groups.items():
        le3 = [t for t in ts if t['num_stages'] <= 3]
        t = min(le3 or ts, key=lambda t: (abs(t['num_stages'] - 3), -t['num_stages'])) if le3 else min(ts, key=lambda t: t['num_stages'])
        L, mt = _L_total(M, N, K, BM, BN, BK, dtype, prof)
        G = next((g for g in GS if g <= mt), 1)
        out.append((L, dict(BM=BM, BN=BN, BK=BK, G=G, num_warps=t['num_warps'], num_stages=t['num_stages'])))
    out.sort(key=lambda x: x[0])
    return out[:k]

def select(M, N, K, dtype, layout, prof):
    return select_scored(M, N, K, dtype, layout, prof, 1)[0][1]
