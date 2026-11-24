"""select(M, N, K, dtype, layout, profile, k=1) -> ranked configs in Inductor template format."""
from . import model as M3
from .profile import Profile, load as load_profile

GS = (8, 16, 32, 4, 1)           # G=8 first: Python's stable sort makes exact ties resolve to 8 (round 3: G rarely matters)
_space_cache = {}

def _space(prof, dtype, layout):
    key = (id(prof), dtype, layout)
    if key not in _space_cache:
        fam = f'{dtype}|{layout}'
        sp = []
        for tk, m in prof['meta'].get(fam, {}).items():
            bm, bn, bk, st, w = M3.parse_tile(tk)
            sp.append(dict(BM=bm, BN=bn, BK=bk, num_stages=st, num_warps=w, occ=m['occ']))
        if not sp: raise ValueError(f'profile has no calibrated tiles for {fam}; run lala_triton.calibrate')
        _space_cache[key] = sp
    return _space_cache[key]

def to_inductor(cfg):
    return dict(BLOCK_M=cfg['BM'], BLOCK_N=cfg['BN'], BLOCK_K=cfg['BK'], GROUP_M=cfg['G'], num_stages=cfg['num_stages'], num_warps=cfg['num_warps'])

import os
TIEBREAK = os.environ.get('LALA_TRITON_TIEBREAK', '0') == '1'   # Amendment 5b; default OFF until validated on the fresh held-out set

DISTINCT = os.environ.get('LALA_TRITON_DISTINCT', '0') == '1'   # Amendment 5c; default OFF until validated

def select_scored(M, N, K, dtype, layout, profile, k=1, Gs=GS, tiebreak=None, distinct=None, padcap=None):
    """padcap (Amendment 5d): None = module default (env LALA_TRITON_PADCAP), True/False overrides for this call only."""
    if padcap is not None and padcap != M3.PAD:
        old = M3.PAD; M3.PAD = padcap
        try: return select_scored(M, N, K, dtype, layout, profile, k, Gs, tiebreak, distinct)
        finally: M3.PAD = old
    prof = load_profile(profile) if isinstance(profile, str) else profile
    if TIEBREAK if tiebreak is None else tiebreak:
        return M3.rank_tiebreak(M, N, K, dtype, layout, prof, _space(prof, dtype, layout), Gs=Gs, k=k)
    if (DISTINCT if distinct is None else distinct) and k and k > 1:
        return M3.rank_distinct(M, N, K, dtype, layout, prof, _space(prof, dtype, layout), Gs=Gs, k=k)
    return M3.rank(M, N, K, dtype, layout, prof, _space(prof, dtype, layout), Gs=Gs, k=k)

AUTO_EPS = float(os.environ['LALA_TRITON_AUTO_EPS']) if os.environ.get('LALA_TRITON_AUTO_EPS') else None   # Amendment 5g; default OFF
AUTO_KMAX = 5
AUTO_STATS = []     # (M, N, K, dtype, tied_group_size, n_returned): n_returned > 1 means the caller (Inductor) will benchmark

def auto_pick(ranked, eps, kmax=AUTO_KMAX):
    """Amendment 5g. ranked = [(t, cfg)] best first (distinct tiles at their best G). If no other distinct tile is predicted within eps of the best -> [top-1] (no benchmarking).
    Otherwise up to kmax distinct tiles from the tied group, chosen greedily for diversity across (BN, warps, stages, BK, BM); ties in diversity go to the better prediction."""
    seen, tiles = set(), []
    for t, c in ranked:
        key = (c['BM'], c['BN'], c['BK'], c['num_warps'], c['num_stages'])
        if key in seen: continue
        seen.add(key); tiles.append((t, c))
    grp = [x for x in tiles if x[0] <= tiles[0][0] * (1 + eps)]
    if len(grp) <= 1: return grp, [tiles[0]]
    pick = [grp[0]]; rest = grp[1:]
    f = lambda c: (c['BM'], c['BN'], c['BK'], c['num_warps'], c['num_stages'])
    while rest and len(pick) < kmax:
        best = max(range(len(rest)), key=lambda i: (min(sum(a != b for a, b in zip(f(rest[i][1]), f(p_[1]))) for p_ in pick), -i))
        pick.append(rest.pop(best))
    return grp, pick

def select_auto_scored(M, N, K, dtype, layout, profile, eps=None, Gs=GS, **kw):
    eps = AUTO_EPS if eps is None else eps
    grp, pick = auto_pick(select_scored(M, N, K, dtype, layout, profile, k=None, Gs=Gs, **kw), eps)
    AUTO_STATS.append((M, N, K, dtype, len(grp), len(pick))); return pick

def select(M, N, K, dtype, layout, profile, k=1, Gs=GS):
    """Ranked list (best first) of Inductor-format dicts: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages, num_warps."""
    if AUTO_EPS is not None: return [to_inductor(c) for _, c in select_auto_scored(M, N, K, dtype, layout, profile)]
    return [to_inductor(c) for _, c in select_scored(M, N, K, dtype, layout, profile, k, Gs)]
