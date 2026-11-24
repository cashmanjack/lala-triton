"""Calibrate a GPU profile.  python -m lala_triton.calibrate --out profiles/<gpu>.json [--gpu 0] [--shard 0/1] [--skip-eff]
Stages (all resumable, raw rows in runs/calib_<name>_*.csv): micro (HBM_BW, L2_BW, timing-based L2-knee sanity check) ->
eff (sustained eff of every feasible tile on 8192^3, per family) -> lat (Little's-law constant on small-M calibration shapes) -> build.
Run with --shard i/n on n GPUs in parallel, then once more with --build-only. L2_eff is a documented constant (see --l2-eff-mib)
unless counters allow the ncu double-read (not implemented here; use the timing knee as a sanity check)."""
import argparse, csv, json, math, os, random, sys, time
import numpy as np
from .config import ROOT, RUNS, TRITON_CACHE
from . import model as M3
from .profile import Profile, save

CAL_EFF = (8192, 8192, 8192)
LAT_SHAPES = [(m, n, k) for m in (1, 4, 16, 32) for (n, k) in ((8192, 8192), (16384, 4096))]
FAMS = [('bf16', 'AB'), ('bf16', 'ABT'), ('fp8', 'ABT')]
LAT_FAMS = [('bf16', 'ABT'), ('fp8', 'ABT')]

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

def append(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new: w.writeheader()
        w.writerow(row)

EFF_COLS = ['dtype', 'layout', 'tile', 'ok', 'n_regs', 'n_spills', 'shared', 'ms', 'sm_clk', 'power', 'eff', 'occ']
def rows(path):
    """Robust reader: a shard whose first row was an infeasible tile has a short header; reuse the full column list for those files."""
    if not os.path.exists(path): return []
    rr = list(csv.reader(open(path))); hdr, body = rr[0], rr[1:]
    if '_eff_' in path and 'eff' not in hdr: hdr = EFF_COLS
    return [dict(zip(hdr, r)) for r in body]

# ---------------------------------------------------------------- micro
def stage_micro(args, name):
    import torch, triton
    from triton.testing import do_bench
    from . import _l2probe as P
    path = f'{RUNS}/calib_{name}_micro.json'
    if os.path.exists(path): return json.load(open(path))
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    a = torch.empty(2**30 // 4, device='cuda'); b = torch.empty_like(a)
    t = do_bench(lambda: b.copy_(a), warmup=25, rep=100, return_mode='median') / 1e3
    hbm = 2 * a.numel() * 4 / t; del a, b
    # L2 bandwidth: 32 MiB resident, 20 passes per launch
    x = torch.zeros(P.seg_elems(32, n_sm), device='cuda'); out = torch.empty(n_sm, device='cuda')
    sc = P.seg_chunk(32, n_sm)
    tl2 = do_bench(lambda: P._probe[(n_sm,)](x, out, sc, STRIDE=1, NREP=20, num_warps=8), warmup=10, rep=50, return_mode='median') / 1e3
    l2bw = 20 * x.numel() * 4 / tl2; del x
    # timing-based knee: extra cost of a 2nd pass relative to the first (hit -> ~0, miss -> ~1)
    sizes = [16, 32, 48, 64, 72, 80, 88, 96, 104, 112, 128, 160, 192, 256]
    curve = []
    for variant, stride in (('seq', 1), ('strided', 1031)):
        for s in sizes:
            x = torch.zeros(P.seg_elems(s, n_sm), device='cuda'); sc = P.seg_chunk(s, n_sm)
            t1 = do_bench(lambda: P._probe[(n_sm,)](x, out, sc, STRIDE=stride, NREP=1, num_warps=8), warmup=5, rep=30, return_mode='median')
            t2 = do_bench(lambda: P._probe[(n_sm,)](x, out, sc, STRIDE=stride, NREP=2, num_warps=8), warmup=5, rep=30, return_mode='median')
            curve.append(dict(variant=variant, MiB=s, extra=t2 / t1 - 1)); del x
    def knee(v):
        c = [(r['MiB'], r['extra']) for r in curve if r['variant'] == v]
        lo, hi = np.mean([e for _, e in c[:3]]), np.mean([e for _, e in c[-3:]]); mid = 0.5 * (lo + hi)
        for (s0, e0), (s1, e1) in zip(c, c[1:]):
            if e0 < mid <= e1: return math.exp(math.log(s0) + (mid - e0) / (e1 - e0) * (math.log(s1) - math.log(s0)))
        return None
    res = dict(hbm_bw=hbm, l2_bw=l2bw, timing_knee_MiB=dict(seq=knee('seq'), strided=knee('strided')), curve=curve, n_sm=n_sm,
               name=torch.cuda.get_device_name(0))
    json.dump(res, open(path, 'w'), indent=1); return res

# ---------------------------------------------------------------- eff
def eff_of(M, N, K, dtype, cfg, ms, n_sm):
    P = math.ceil(M / cfg['BM']) * math.ceil(N / cfg['BN'])
    ideal = math.ceil(P / n_sm) * 2 * cfg['BM'] * cfg['BN'] * K / (M3.PEAK[dtype] / n_sm)       # eff = 1 time under T_comp
    return ideal / (ms * 1e-3)

def stage_eff(args, name, n_sm):
    from . import space, _measure
    path = f'{RUNS}/calib_{name}_eff_{args.shard[0]}of{args.shard[1]}.csv'
    done = {(r['dtype'], r['layout'], r['tile']) for r in rows(path)}
    Mx, Nx, Kx = CAL_EFF
    for dtype, layout in FAMS:
        tiles = space.candidate_tiles(dtype)
        random.Random(5).shuffle(tiles)
        tiles = [t for i, t in enumerate(tiles) if i % args.shard[1] == args.shard[0]]
        if args.quick: tiles = tiles[:8]
        sh = _measure.Shape(Mx, Nx, Kx, dtype, layout)
        for i, t in enumerate(tiles):
            tk = space.key(t)
            if (dtype, layout, tk) in done: continue
            cfg = dict(t, G=8)
            try: res, meta = _measure.time_cfg(sh, cfg, args.gpu, *args.window)
            except Exception as e: res, meta = None, dict(n_regs=-1, n_spills=-1, shared=-1); log('fail', dtype, layout, tk, repr(e)[:100])
            row = dict(dtype=dtype, layout=layout, tile=tk, ok=int(res is not None), **meta)
            if res is not None:
                row.update(ms=res['ms_per_gemm'], sm_clk=res['sm_clk_med'], power=res['power_mean'], eff=eff_of(Mx, Nx, Kx, dtype, cfg, res['ms_per_gemm'], n_sm),
                           occ=M3.occupancy(meta['shared'], meta['n_regs'], t['num_warps']))
            append(path, row)
            if i % 20 == 0: log(f'eff {dtype}/{layout} {i}/{len(tiles)} {tk}', '' if res is None else f"{res['ms_per_gemm']:.2f}ms clk={res['sm_clk_med']:.0f}")
        sh.close()

# ---------------------------------------------------------------- lat
def stage_lat(args, name, n_sm):
    from . import space, _measure
    path = f'{RUNS}/calib_{name}_lat_{args.shard[0]}of{args.shard[1]}.csv'
    done = {(r['dtype'], r['M'], r['N'], r['K'], r['tile']) for r in rows(path)}
    for dtype, layout in LAT_FAMS:
        tiles_all = space.candidate_tiles(dtype)
        for (M, N, K) in (LAT_SHAPES[::3] if args.quick else LAT_SHAPES)[args.shard[0]::args.shard[1]]:
            tiles = random.Random(f'lat-{dtype}-{M}-{N}').sample(tiles_all, 4 if args.quick else 20)
            sh = _measure.Shape(M, N, K, dtype, layout)
            for t in tiles:
                tk = space.key(t)
                if (dtype, str(M), str(N), str(K), tk) in done: continue
                cfg = dict(t, G=8)
                try: res, meta = _measure.time_cfg(sh, cfg, args.gpu, *args.window)
                except Exception as e: res, meta = None, dict(n_regs=-1, n_spills=-1, shared=-1)
                if res is None: continue
                append(path, dict(dtype=dtype, layout=layout, M=M, N=N, K=K, tile=tk, ms=res['ms_per_gemm'], sm_clk=res['sm_clk_med'], **meta,
                                  occ=M3.occupancy(meta['shared'], meta['n_regs'], t['num_warps'])))
            log('lat', dtype, M, N, K); sh.close()

# ---------------------------------------------------------------- build
def build(args, name):
    micro = json.load(open(f'{RUNS}/calib_{name}_micro.json'))
    eff, meta = {}, {}
    for r in sum([rows(f'{RUNS}/calib_{name}_eff_{i}of{args.shard[1]}.csv') for i in range(args.shard[1])], []):
        if r['ok'] != '1': continue
        k = f"{r['dtype']}|{r['layout']}"; eff.setdefault(k, {})[r['tile']] = float(r['eff'])
        meta.setdefault(k, {})[r['tile']] = dict(occ=int(r['occ']), n_regs=int(r['n_regs']), shared=int(r['shared']))
    prof = Profile(name=name, gpu=micro['name'], n_sm=micro['n_sm'], l2_eff=args.l2_eff_mib * 2**20,
                   l2_eff_source=args.l2_eff_source, hbm_bw=micro['hbm_bw'], l2_bw=micro['l2_bw'], timing_knee_MiB=micro['timing_knee_MiB'],
                   peak_tflops={k: v / 1e12 for k, v in M3.PEAK.items()}, eff=eff, meta=meta, lat={'bf16': 5e-7, 'fp8': 5e-7},
                   notes='lala_triton REIMPLEMENTATION; eff calibrated sustained on 8192^3, G=8; see PREREGISTRATION Amendment 5/5a.')
    # fit lat per dtype on calibration small-M shapes only
    lat_rows = sum([rows(f'{RUNS}/calib_{name}_lat_{i}of{args.shard[1]}.csv') for i in range(args.shard[1])], [])
    prof['lat_fit'] = {}
    for dtype in ('bf16', 'fp8'):
        rr = [r for r in lat_rows if r['dtype'] == dtype]
        if len(rr) < (3 if args.quick else 10): continue
        def loss(lat):
            prof['lat'][dtype] = lat; e = 0
            for r in rr:
                bm, bn, bk, st, w = M3.parse_tile(r['tile'])
                cfg = dict(BM=bm, BN=bn, BK=bk, G=8, num_warps=w, num_stages=st)
                p = M3.predict(int(r['M']), int(r['N']), int(r['K']), cfg, dtype, prof, r['layout'], occ=int(r['occ']))
                e += (math.log(p) - math.log(float(r['ms']) * 1e-3)) ** 2
            return e / len(rr)
        l0 = loss(1e-12); grid = np.logspace(-8, -4.5, 120); ls = [loss(l) for l in grid]; j = int(np.argmin(ls))
        prof['lat'][dtype] = float(grid[j]); prof['lat_fit'][dtype] = dict(rms_log_err=math.sqrt(ls[j]), n=len(rr),
                                                                          rms_log_err_without_term=math.sqrt(l0))
    save(prof, args.out); log('wrote', args.out, 'lat', prof['lat'], prof['lat_fit']); return prof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--decode-table', action='store_true', help='Amendment 5f: add the small-M decode table to an EXISTING profile (--out). Run with --shard i/n per GPU, then once with --build-only.')
    ap.add_argument('--skip-eff', action='store_true'); ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--l2-eff-mib', type=float, default=102.0); ap.add_argument('--l2-eff-source', default='round1 REPORT.md (no misses to 88 MiB, knee ~102 MiB); NOT re-measured')
    ap.add_argument('--window', default='0.5,1.5,1.0'); ap.add_argument('--quick', action='store_true', help='smoke test: 8 tiles per family, 2 latency shapes (profile is NOT usable)')
    args = ap.parse_args(); args.shard = tuple(map(int, args.shard.split('/'))); args.window = tuple(map(float, args.window.split(',')))
    name = os.path.splitext(os.path.basename(args.out))[0]
    os.makedirs(RUNS, exist_ok=True)
    if args.decode_table:
        from . import decode
        if args.build_only:
            import glob
            from .profile import load
            prof = load(args.out); n = decode.build_into(prof, sorted(glob.glob(f'{RUNS}/calib_{name}_decode_*of*.csv'))); save(prof, args.out); log('decode table tiles', n); return
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu); os.environ.setdefault('TRITON_CACHE_DIR', TRITON_CACHE); t0 = time.time()
        decode.sweep(f'{RUNS}/calib_{name}_decode_{args.shard[0]}of{args.shard[1]}.csv', shard=args.shard, log=log); log(f'decode-table shard done in {(time.time()-t0)/60:.1f} min'); return
    if args.build_only: build(args, name); return
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu); os.environ.setdefault('TRITON_CACHE_DIR', TRITON_CACHE)
    import torch
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    t0 = time.time()
    if args.shard[0] == 0: log('micro', stage_micro(args, name)['hbm_bw'])
    else:
        while not os.path.exists(f'{RUNS}/calib_{name}_micro.json'): time.sleep(5)
    if not args.skip_eff: stage_eff(args, name, n_sm)
    stage_lat(args, name, n_sm)
    log(f'calibration shard done in {(time.time()-t0)/60:.1f} min')

if __name__ == '__main__': main()
