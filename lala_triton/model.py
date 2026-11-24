"""M3 traffic model + time model + ranking. Pure Python, no GPU. REIMPLEMENTED from PREREGISTRATION.md "Model definitions" and Amendment 5 (R5-T)."""
import math
ESIZE = {'bf16': 2, 'fp8': 1}
OUT_ESIZE = {'bf16': 2, 'fp8': 2}
PEAK = {'bf16': 364.2e12, 'fp8': 728.5e12}
LAUNCH = 4e-6
import os
PAD = os.environ.get('LALA_TRITON_PADCAP', '0') == '1'     # Amendment 5d: BM cap + padded-row A charge; default OFF until validated
DEC5E = os.environ.get('LALA_TRITON_DECODE5E', '0') == '1'   # Amendment 5e: K-loop latency floor (M<=64); default OFF until validated
DEC5F = os.environ.get('LALA_TRITON_DECODE5F', '0') == '1'   # Amendment 5f: measured decode table (M<=64); falls back to 5e if DEC5E else to the current model
import json as _json
try: _P5E = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params_5e.json')))
except OSError: _P5E = {}
TRAFFIC = os.environ.get('LALA_TRITON_TRAFFIC', 'm3')     # 'm3' (default) or 'naive' (ablation: per-tile traffic, no L2 reuse model)

def tile_key(bm, bn, bk, s, w): return f'{bm}x{bn}x{bk}_w{w}_s{s}'
def parse_tile(t):
    a, w, s = t.split('_'); bm, bn, bk = map(int, a.split('x')); return bm, bn, bk, int(s[1:]), int(w[1:])

def occupancy(shared, n_regs, num_warps, smem=100*1024, regs=65536, max_warps=48, max_cta=24):
    """Resident CTAs per SM from compiled metadata (1 KiB SMEM reserved per CTA)."""
    n = min(max_cta, max_warps // num_warps, regs // max(1, n_regs * 32 * num_warps), smem // (shared + 1024) if shared else max_cta)
    return max(1, n)

def estimate_occ(BM, BN, BK, st, w, s):
    """Pre-compile occupancy guess for tiles not in the profile: SMEM (st-1 buffers) and ~ (acc regs + 48) per thread."""
    shared = max(st - 1, 1) * (BM + BN) * BK * s
    regs = min(255, BM * BN // (32 * w) + 48)
    return occupancy(shared, regs, w)

def window(M, N, BM, BN, occ, n_sm):
    P = math.ceil(M / BM) * math.ceil(N / BN)
    return P, min(P, n_sm * occ)

def pad_rows(M, BM):
    """Amendment 5d: rows of A that the Tm = ceil(M/BM) row-tiles actually load/compute (padded to a whole tile)."""
    return math.ceil(M / BM) * BM

def bm_cap(M):
    """Amendment 5d: largest BLOCK_M the selector may pick: max(16, next_pow2(M))."""
    return max(16, 1 << max(0, (M - 1).bit_length()))

def traffic(M, N, K, BM, BN, G, W, s, L2):
    """DRAM read bytes (A + B) of the grouped-order GEMM. Returns (A, B, info).
    With PAD (Amendment 5d) A0 is charged for the padded rows ceil(M/BM)*BM instead of M (footprints F2/F3 already use whole tiles)."""
    Tm, Tn = math.ceil(M / BM), math.ceil(N / BN)
    g = max(1, min(G, Tm, W))
    c = min(Tn, math.ceil(W / g))
    r = min(Tm, max(g, W / Tn)) if c == Tn else g
    F2 = (r * BM + c * BN) * K * s
    F3 = (g * BM + N) * K * s
    A0, B0 = (pad_rows(M, BM) if PAD else M) * K * s, N * K * s
    if A0 + B0 <= L2: A, B, reg = A0, B0, 'R0'
    else:
        A = A0 if (F2 <= L2 or c == Tn) else A0 * Tn / c
        B = B0 if F3 <= L2 else B0 * Tm / r
        reg = 'R2' if A > A0 else ('R1' if B > B0 else 'R0')
    return A, B, dict(regime=reg, r=r, c=c, g=g, F2=F2, F3=F3, Tm=Tm, Tn=Tn)

def predict(M, N, K, cfg, dtype, prof, layout='ABT', occ=None, eff=None, detail=False):
    """Predicted seconds. cfg: dict BM,BN,BK,G,num_warps,num_stages."""
    BM, BN, BK, G, w, st = cfg['BM'], cfg['BN'], cfg['BK'], cfg['G'], cfg['num_warps'], cfg['num_stages']
    n_sm, s, so = prof['n_sm'], ESIZE[dtype], OUT_ESIZE[dtype]
    tk = tile_key(BM, BN, BK, st, w)
    if occ is None: occ = prof.occ_for(dtype, layout, tk) or estimate_occ(BM, BN, BK, st, w, s)
    if eff is None: eff = prof.eff_for(dtype, layout, tk)
    P, W = window(M, N, BM, BN, occ, n_sm)
    A, B, info = traffic(M, N, K, BM, BN, G, W, s, prof['l2_eff'])
    if TRAFFIC == 'naive':
        A = B = 0.5 * max(naive_read_bytes(M, N, K, BM, BN, s), (M + N) * K * s)   # naive reads split evenly; only the sum enters the time
    nbytes = A + B + M * N * so
    t_comp = math.ceil(P / n_sm) * 2 * BM * BN * K / (PEAK[dtype] / n_sm * eff)
    inflight = min(P, n_sm * occ) * max(st - 1, 1) * (BM + BN) * BK * s
    needed = prof['lat'][dtype] * prof['hbm_bw']
    attain = min(1.0, inflight / needed)
    t_dram = nbytes / (prof['hbm_bw'] * attain)
    t_l2 = ((pad_rows(M, BM) if PAD else M) * K * s * math.ceil(N / BN) + N * K * s * math.ceil(M / BM)) / prof['l2_bw']
    t = max(t_comp, t_dram, t_l2) + LAUNCH
    if M <= 64 and (DEC5E or DEC5F):
        t5f = None
        if DEC5F:
            from . import decode as _dec
            t5f = _dec.predict(M, N, K, BM, BN, BK, tk, dtype, occ, nbytes, prof)
        if t5f is not None: t = t5f
        elif DEC5E and dtype in _P5E:
            q = _P5E[dtype]; waves = math.ceil(P / (n_sm * occ))
            t = max(t, waves * math.ceil(K / BK) * q['t_iter_us'] * 1e-6 / min(max(st - 1, 1), q['d_sat']))
    if detail: return t, dict(t_comp=t_comp, t_dram=t_dram, t_l2=t_l2, bytes=nbytes, read=A + B, attain=attain, occ=occ, eff=eff, **info)
    return t

def rank(M, N, K, dtype, layout, prof, space, Gs=(8,), k=None):
    """space: iterable of tile dicts (BM,BN,BK,num_warps,num_stages, optional occ). Returns configs sorted by predicted time."""
    out = []
    if PAD:      # Amendment 5d: cap BLOCK_M during selection
        cap = bm_cap(M); sub = [t for t in space if t['BM'] <= cap]
        if not sub: raise ValueError(f'no calibrated tile with BM <= {cap} (M={M})')
        space = sub
    for t in space:
        for G in Gs:
            cfg = dict(BM=t['BM'], BN=t['BN'], BK=t['BK'], G=G, num_warps=t['num_warps'], num_stages=t['num_stages'])
            out.append((predict(M, N, K, cfg, dtype, prof, layout, occ=t.get('occ')), cfg))
    out.sort(key=lambda x: x[0])
    return out[:k] if k else out

def naive_read_bytes(M, N, K, BM, BN, s):
    """M1 (killtest Section 2 baseline): per-tile traffic, no L2 reuse."""
    return M * N * K * s * (1.0 / BM + 1.0 / BN)

def debug(M, N, K, cfg, dtype, prof, layout='ABT'):
    t, d = predict(M, N, K, cfg, dtype, prof, layout, detail=True)
    s = ESIZE[dtype]; Tm, Tn = d['Tm'], d['Tn']
    return f"t={t*1e6:.1f}us regime={d['regime']} r={d['r']:.1f} c={d['c']} F2={d['F2']/2**20:.1f}MiB F3={d['F3']/2**20:.1f}MiB bytes={d['bytes']/2**20:.1f}MiB attain={d['attain']:.2f} occ={d['occ']} comp={d['t_comp']*1e6:.1f} dram={d['t_dram']*1e6:.1f} l2={d['t_l2']*1e6:.1f}"


TIE_EPS = 0.02
def rank_tiebreak(M, N, K, dtype, layout, prof, space, Gs=(8,), k=None):
    """Amendment 5b: among candidates within TIE_EPS of the best that are DRAM-bound, prefer smallest BM, then most bytes in flight."""
    allr = rank(M, N, K, dtype, layout, prof, space, Gs=Gs)
    t0, c0 = allr[0]
    def det(c, occ=None):
        return predict(M, N, K, c, dtype, prof, layout, occ=occ, detail=True)[1]
    occ = {(t['BM'], t['BN'], t['BK'], t['num_stages'], t['num_warps']): t.get('occ') for t in space}
    def dram_bound(c):
        d = det(c, occ[(c['BM'], c['BN'], c['BK'], c['num_stages'], c['num_warps'])]); return d['t_dram'] >= d['t_comp'] and d['t_dram'] >= d['t_l2'], d
    ok, _ = dram_bound(c0)
    if not ok: return allr[:k] if k else allr
    S = []
    for idx, (t, c) in enumerate(allr):
        if t > t0 * (1 + TIE_EPS): break
        b, d = dram_bound(c)
        if b:
            s_ = ESIZE[dtype]; P, W = window(M, N, c['BM'], c['BN'], d['occ'], prof['n_sm'])
            infl = W * max(c['num_stages'] - 1, 1) * (c['BM'] + c['BN']) * c['BK'] * s_
            S.append((c['BM'], -infl, t, idx, c))
    S.sort(key=lambda x: x[:4])
    first = S[0]; out = [(first[2], first[4])] + [x for i, x in enumerate(allr) if i != first[3]]
    return out[:k] if k else out


def rank_distinct(M, N, K, dtype, layout, prof, space, Gs=(8,), k=None):
    """Amendment 5c: top-k distinct tiles, each at its best G (first G wins exact ties)."""
    out, seen = [], set()
    for t, c in rank(M, N, K, dtype, layout, prof, space, Gs=Gs):
        key = (c['BM'], c['BN'], c['BK'], c['num_warps'], c['num_stages'])
        if key in seen: continue
        seen.add(key); out.append((t, c))
        if k and len(out) >= k: break
    return out
