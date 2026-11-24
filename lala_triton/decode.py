"""Amendment 5f: measured decode table. Per (dtype, tile): log(achieved BW = bytes/time) = a + b*log(active blocks) + c*log(K-loop iterations),
ridge-fitted on a small-M calibration sweep (M<=64, all tiles with BM<=min(64, 5d cap)), stored as profile['decode'][dtype][tile] = dict(a,b,c,lo_a,hi_a,lo_k,hi_k,n).
Predicts UNSEEN (N,K,M): active = min(ceil(M/BM)*ceil(N/BN), SMs*occ), kit = ceil(K/BK). Above the table's active range we clamp (BW does not fall with more blocks);
below it or outside the K-iteration range by more than TOL (log) we return None (caller falls back and the event is counted in FALLBACK)."""
import math, collections
import numpy as np
FALLBACK = collections.Counter()
TOL = 0.4          # natural-log tolerance outside the measured (active-low, kit) range
RIDGE = 0.3        # penalty on the slopes b, c
MMAX = 64

def fit_table(rows, n_sm):
    """rows: iterable of dicts with dtype,tile,M,N,K,BM,BN,BK,occ,t(seconds),bytes. Returns {dtype: {tile: params}}."""
    by = collections.defaultdict(list)
    for r in rows: by[(r['dtype'], r['tile'])].append(r)
    out = collections.defaultdict(dict)
    for (dt, tk), rs in by.items():
        la = np.log([min(math.ceil(r['M'] / r['BM']) * math.ceil(r['N'] / r['BN']), n_sm * r['occ']) for r in rs])
        lk = np.log([math.ceil(r['K'] / r['BK']) for r in rs]); y = np.log([r['bytes'] / r['t'] for r in rs])
        if len({(round(a, 3), round(k, 3)) for a, k in zip(la, lk)}) < 3: continue
        ma, mk = la.mean(), lk.mean(); X = np.column_stack([la - ma, lk - mk]); A = X.T @ X + RIDGE * np.eye(2)
        bc = np.linalg.solve(A, X.T @ (y - y.mean())); a0 = y.mean() - bc[0] * ma - bc[1] * mk
        out[dt][tk] = dict(a=float(a0), b=float(bc[0]), c=float(bc[1]), lo_a=float(la.min()), hi_a=float(la.max()), lo_k=float(lk.min()), hi_k=float(lk.max()), n=len(rs))
    return {k: dict(v) for k, v in out.items()}

def predict(M, N, K, BM, BN, BK, tile, dtype, occ, nbytes, prof, table=None):
    """Seconds, or None (fallback). Counts every outcome in FALLBACK."""
    tab = (table if table is not None else prof.get('decode', {})).get(dtype, {}).get(tile)
    if M > MMAX or not tab: FALLBACK['no_table' if M <= MMAX else 'M>64'] += 1; return None
    la = math.log(min(math.ceil(M / BM) * math.ceil(N / BN), prof['n_sm'] * occ)); lk = math.log(math.ceil(K / BK))
    if la < tab['lo_a'] - TOL or lk < tab['lo_k'] - TOL or lk > tab['hi_k'] + TOL: FALLBACK['out_of_range'] += 1; return None
    if la > tab['hi_a']: FALLBACK['clamped_hi_active'] += 1; la = tab['hi_a']
    else: FALLBACK['ok'] += 1
    bw = min(math.exp(tab['a'] + tab['b'] * la + tab['c'] * lk), prof['hbm_bw'])
    return nbytes / bw

# ------------------------------------------------------------------ calibration (used by `calibrate --decode-table` and eval/decode_sweep.py)
CAL_NK = ((2048, 2048), (6144, 6144), (12288, 3072), (3072, 12288))
CAL_MS = (1, 2, 4, 8, 16, 32, 64)
COLS = ['dtype', 'N', 'K', 'M', 'tile', 'ok', 'n_regs', 'shared', 'occ', 'ms', 'sm_clk', 'power']

def sweep(out, gpu=0, shard=(0, 1), dtypes=('bf16', 'fp8'), ms=CAL_MS, nks=CAL_NK, window_s=0.3, log=print):
    """Time every feasible tile (BM <= min(64, 5d cap), G=8, layout ABT) on the calibration (N,K) x M grid. Resumable append-only CSV. Diagnosis-grade timing
    (0.15 s warmup / window_s window / no cooldown), NOT the sustained protocol. Caller sets CUDA_VISIBLE_DEVICES before importing torch."""
    import csv, os, time
    from . import space, _measure, model as M3
    done = set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)): done.add((r['dtype'], r['N'], r['K'], r['M'], r['tile']))
    jobs = [(dt, N, K, M, t) for dt in dtypes for (N, K) in nks for M in ms for t in space.candidate_tiles(dt) if t['BM'] <= min(64, M3.bm_cap(M))]
    jobs = [j for i, j in enumerate(jobs) if i % shard[1] == shard[0]]; log(len(jobs), 'decode-table jobs in shard'); cur, sh, t0 = None, None, time.time()
    for i, (dt, N, K, M, t) in enumerate(jobs):
        tk = M3.tile_key(t['BM'], t['BN'], t['BK'], t['num_stages'], t['num_warps'])
        if (dt, str(N), str(K), str(M), tk) in done: continue
        if cur != (dt, N, K, M):
            if sh: sh.close()
            sh = _measure.Shape(M, N, K, dt, 'ABT'); cur = (dt, N, K, M)
        try: res, meta = _measure.time_cfg(sh, dict(t, G=8), 0, warmup_s=0.15, window_s=window_s, cooldown_s=0.0)
        except Exception as e: row = dict(dtype=dt, N=N, K=K, M=M, tile=tk, ok=0); res = 'x'
        if res != 'x':
            row = dict(dtype=dt, N=N, K=K, M=M, tile=tk, ok=int(res is not None), n_regs=meta['n_regs'], shared=meta['shared'], occ=M3.occupancy(meta['shared'], meta['n_regs'], t['num_warps']),
                       ms=(res['ms_per_gemm'] if res else ''), sm_clk=(res['sm_clk_med'] if res else ''), power=(res['power_mean'] if res else ''))
        new = not os.path.exists(out) or os.path.getsize(out) == 0
        with open(out, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if new: w.writeheader()
            w.writerow(row)
        if i % 100 == 0: log(f'{i}/{len(jobs)} {time.time()-t0:.0f}s')
    log('decode-table shard done')

def build_into(profile, csv_paths):
    """Fit the table from sweep CSVs and store it as profile['decode'] (in place). Returns number of tiles."""
    import csv
    from . import model as M3
    rows = []
    for p in csv_paths:
        for r in csv.DictReader(open(p)):
            if r['ok'] != '1': continue
            bm, bn, bk, st, w = M3.parse_tile(r['tile']); s = M3.ESIZE[r['dtype']]; N, K, M = int(r['N']), int(r['K']), int(r['M'])
            rows.append(dict(dtype=r['dtype'], tile=r['tile'], M=M, N=N, K=K, BM=bm, BN=bn, BK=bk, occ=int(r['occ']), t=float(r['ms']) * 1e-3, bytes=M * K * s + N * K * s + M * N * M3.OUT_ESIZE[r['dtype']]))
    profile['decode'] = fit_table(rows, profile['n_sm']); return {k: len(v) for k, v in profile['decode'].items()}
