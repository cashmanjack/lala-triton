import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import itertools, pytest
from lala_triton import model as m
MiB = 2**20; L2 = 102 * MiB
PROF = dict(n_sm=142, l2_eff=L2, hbm_bw=8e11, l2_bw=4e12, lat=dict(bf16=5e-7, fp8=5e-7), eff={}, meta={})
from lala_triton.profile import Profile; PROF = Profile(PROF)

def cases():
    for (M, N, K) in [(8192, 8192, 512), (8192, 8192, 8192), (16384, 16384, 24576), (128, 4096, 4096), (6144, 16384, 8192)]:
        for (BM, BN) in [(128, 128), (128, 256), (64, 64)]:
            for G in (1, 8, 32):
                for occ in (1, 2):
                    yield M, N, K, BM, BN, G, occ

@pytest.mark.parametrize('M,N,K,BM,BN,G,occ', list(cases()))
def test_reads_ge_compulsory(M, N, K, BM, BN, G, occ):
    P, W = m.window(M, N, BM, BN, occ, 142)
    A, B, i = m.traffic(M, N, K, BM, BN, G, W, 2, L2)
    assert A + B >= (M + N) * K * 2 - 1
    if i['regime'] == 'R0': assert abs(A + B - (M + N) * K * 2) < 1

@pytest.mark.parametrize('M,N,K,BM,BN,G,occ', list(cases()))
def test_monotone_in_l2(M, N, K, BM, BN, G, occ):
    P, W = m.window(M, N, BM, BN, occ, 142)
    prev = float('inf')
    for l2 in (16 * MiB, 48 * MiB, 102 * MiB, 400 * MiB):
        A, B, _ = m.traffic(M, N, K, BM, BN, G, W, 2, l2)
        assert A + B <= prev + 1; prev = A + B

def test_w1_matches_naive():
    M, N, K, BM, BN = 1024, 1024, 1000000, 128, 128         # huge K: F2 > L2 (R2), so no reuse
    A, B, i = m.traffic(M, N, K, BM, BN, 8, 1, 2, L2)
    assert i['regime'] == 'R2'
    assert abs((A + B) - m.naive_read_bytes(M, N, K, BM, BN, 2)) / (A + B) < 1e-9

def test_predict_positive_and_ranked():
    space = [dict(BM=bm, BN=bn, BK=64, num_warps=4, num_stages=3) for bm in (64, 128) for bn in (64, 128, 256)]
    r = m.rank(4096, 4096, 4096, 'bf16', 'ABT', PROF, space)
    assert all(t > 0 for t, _ in r) and [t for t, _ in r] == sorted(t for t, _ in r)


def test_tiebreak_rule_5b():
    """Amendment 5b: picks within 2% of best that are DRAM-bound -> smallest BM; non-DRAM-bound best -> plain argmin."""
    import os, json
    from lala_triton.profile import load
    from lala_triton.select import select_scored
    p = os.path.join(os.path.dirname(__file__), '..', 'profiles', 'rtx6000ada.json')
    if not os.path.exists(p): return
    prof = load(p)
    a = select_scored(1, 28672, 4096, 'fp8', 'ABT', prof, k=1, tiebreak=False)[0]
    b = select_scored(1, 28672, 4096, 'fp8', 'ABT', prof, k=1, tiebreak=True)[0]
    assert b[0] <= a[0] * 1.02 and b[1]['BM'] <= a[1]['BM']
    a = select_scored(8192, 8192, 8192, 'bf16', 'ABT', prof, k=1, tiebreak=False)[0]
    b = select_scored(8192, 8192, 8192, 'bf16', 'ABT', prof, k=1, tiebreak=True)[0]
    assert a == b      # compute-bound: rule inactive


def test_distinct_topk_5c():
    """Amendment 5c: top-k returns k distinct tiles; top-1 identical to plain ranking."""
    import os
    from lala_triton.profile import load
    from lala_triton.select import select_scored
    p = os.path.join(os.path.dirname(__file__), '..', 'profiles', 'rtx6000ada.json')
    if not os.path.exists(p): return
    prof = load(p)
    r = select_scored(1, 37888, 3584, 'fp8', 'ABT', prof, k=5, distinct=True)
    keys = {(c['BM'], c['BN'], c['BK'], c['num_warps'], c['num_stages']) for _, c in r}
    assert len(r) == 5 and len(keys) == 5
    assert [t for t, _ in r] == sorted(t for t, _ in r)
    assert r[0] == select_scored(1, 37888, 3584, 'fp8', 'ABT', prof, k=1, distinct=False)[0]


def test_tritonblas_style_baseline_returns_profile_tile():
    import os
    from lala_triton.profile import load
    from lala_triton.baselines import tritonblas as TB
    p = os.path.join(os.path.dirname(__file__), '..', 'profiles', 'rtx6000ada.json')
    if not os.path.exists(p): return
    prof = load(p)
    for dt, (M, N, K) in (('bf16', (1, 6144, 4096)), ('fp8', (2048, 4096, 14336))):
        c = TB.select(M, N, K, dt, 'ABT', prof)
        assert f"{c['BM']}x{c['BN']}x{c['BK']}_w{c['num_warps']}_s{c['num_stages']}" in prof['meta'][f'{dt}|ABT']
        assert c == TB.select(M, N, K, dt, 'ABT', prof)


def test_bm_cap_and_pad_5d():
    """Amendment 5d: cap = max(16, next_pow2(M)); padded-row charge; default OFF and a no-op when M is a multiple of BM."""
    assert [m.bm_cap(x) for x in (1, 8, 16, 17, 32, 33, 128, 512, 3000, 8192)] == [16, 16, 16, 32, 32, 64, 128, 512, 4096, 8192]
    assert m.pad_rows(1, 128) == 128 and m.pad_rows(8, 16) == 16 and m.pad_rows(128, 64) == 128 and m.pad_rows(100, 64) == 128
    assert m.PAD is False
    import os
    from lala_triton.profile import load
    from lala_triton.select import select_scored
    p = os.path.join(os.path.dirname(__file__), '..', 'profiles', 'rtx6000ada.json')
    if not os.path.exists(p): return
    prof = load(p)
    for dt in ('bf16', 'fp8'):
        for M in (1, 8, 16, 17, 32, 128):
            for _, c in select_scored(M, 8192, 4096, dt, 'ABT', prof, k=50, padcap=True):
                assert c['BM'] <= m.bm_cap(M)
        for M in (512, 2048, 8192):       # cap does not bind (BM <= 256) and every BM divides M: top-1 identical
            assert select_scored(M, 8192, 4096, dt, 'ABT', prof, k=1, padcap=True) == select_scored(M, 8192, 4096, dt, 'ABT', prof, k=1, padcap=False)
    assert m.PAD is False                  # per-call override restored the module default
    # T_comp already charges whole BM x BN tiles, so the flag must not change it; only the A bytes (DRAM and L2) change
    cfg = dict(BM=128, BN=128, BK=64, G=8, num_warps=4, num_stages=3); occ = 2; eff = 0.3
    t0, d0 = m.predict(1, 4096, 4096, cfg, 'bf16', PROF, occ=occ, eff=eff, detail=True)
    m.PAD = True
    try: t1, d1 = m.predict(1, 4096, 4096, cfg, 'bf16', PROF, occ=occ, eff=eff, detail=True)
    finally: m.PAD = False
    assert d1['t_comp'] == d0['t_comp'] and d1['read'] > d0['read'] and d1['t_l2'] > d0['t_l2']
    assert abs(d1['read'] - d0['read'] - (128 - 1) * 4096 * 2) < 1
