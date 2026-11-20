"""Candidate tile space (PREREGISTRATION R5-C). Wider than Inductor's stock list: includes BK=16 and single-stage tiles."""
import itertools
from .model import tile_key

def candidate_tiles(dtype, max_smem=99 * 1024):
    s = 1 if dtype == 'fp8' else 2
    out = []
    for bm, bn, bk, st in itertools.product((16, 32, 64, 128, 256), (32, 64, 128, 256), (16, 32, 64, 128), (1, 2, 3, 4)):
        if bm * bn < 16 * 32: continue
        if max(st - 1, 1) * (bm + bn) * bk * s > max_smem: continue              # pre-compile filter (exact for bf16, optimistic for fp8)
        area = bm * bn
        for w in ((2, 4) if area <= 1024 else (4,) if area <= 8192 else (4, 8)):
            out.append(dict(BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=st))
    return out

def key(t): return tile_key(t['BM'], t['BN'], t['BK'], t['num_stages'], t['num_warps'])
