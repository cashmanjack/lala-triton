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

def select_scored(M, N, K, dtype, layout, profile, k=1, Gs=GS, tiebreak=None):
    prof = load_profile(profile) if isinstance(profile, str) else profile
    if TIEBREAK if tiebreak is None else tiebreak:
        return M3.rank_tiebreak(M, N, K, dtype, layout, prof, _space(prof, dtype, layout), Gs=Gs, k=k)
    return M3.rank(M, N, K, dtype, layout, prof, _space(prof, dtype, layout), Gs=Gs, k=k)

def select(M, N, K, dtype, layout, profile, k=1, Gs=GS):
    """Ranked list (best first) of Inductor-format dicts: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages, num_warps."""
    return [to_inductor(c) for _, c in select_scored(M, N, K, dtype, layout, profile, k, Gs)]
