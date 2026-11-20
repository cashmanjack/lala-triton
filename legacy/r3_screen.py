"""Exhaustive harness screening of E(shape): do_bench(warmup=10, rep=50, median), 1 rep, seeded random order.
argv: shape gpu. Appends runs/r3_screen.csv (resumable per (shape, cfg))."""
import sys, os, csv, json, random, time, subprocess, shutil
R = '/home/min/a/cashman3/lala-triton-killtest'
shape, gpu = sys.argv[1], sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
TC = f'/var/tmp/cashman3_r3/tc_screen/{shape}'
os.environ['TRITON_CACHE_DIR'] = TC
sys.path.insert(0, R)
import pandas as pd
SCR = f'{R}/runs/r3_screen.csv'
p = pd.read_csv(f'{R}/predictions/r3_pred.csv'); p = p[(p.phase == 'D') & (p['shape'] == shape)]
done = {r['cfg'] for r in csv.DictReader(open(SCR)) if r['shape'] == shape} if os.path.exists(SCR) else set()
cfgs = [dict(BM=int(r.BM), BN=int(r.BN), BK=int(r.BK), G=int(r.G), num_warps=int(r.num_warps), num_stages=int(r.num_stages), cfg=r.cfg)
        for r in p.itertuples() if r.cfg not in done]
if cfgs:
    t0 = time.time()
    chunks = [cfgs[i::12] for i in range(12)]
    procs = [subprocess.Popen(['/usr/bin/python3', f'{R}/r3_precompile.py', shape, json.dumps(ch)], env=dict(os.environ),
                              stdout=subprocess.DEVNULL, stderr=open(f'{R}/logs/r3/precompile.err', 'a')) for ch in chunks if ch]
    for pr in procs: pr.wait()
    tpre = time.time() - t0
    import torch
    from triton.testing import do_bench
    from common import make_inputs
    from kernels.inductor_tmpl import launch
    dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
    a, b, c = make_inputs(M, N, K, dt, layout=lay)
    ref = a.float() @ b.float(); tol = (1e-2 * ref.abs().max().item() + 1e-2) if dt == 'bf16' else 5e-2 * ref.abs().max().item()
    random.Random(f'r3screen-{shape}').shuffle(cfgs)
    new = not os.path.exists(SCR)
    f = open(SCR, 'a', newline=''); w = csv.writer(f)
    if new: w.writerow(['shape', 'cfg', 'BM', 'BN', 'BK', 'G', 'num_warps', 'num_stages', 'ms', 'ok', 'maxerr', 'n_regs', 'n_spills', 'compile_s', 'ts'])
    for cf in cfgs:
        try:
            t1 = time.time(); c.zero_(); k = launch(a, b, c, cf, dt); torch.cuda.synchronize(); tc = time.time() - t1
            err = (c.float() - ref).abs().max().item()
            ms = do_bench(lambda: launch(a, b, c, cf, dt), warmup=10, rep=50, return_mode='median')
            row = [ms, int(err <= tol), err, k.n_regs, k.n_spills, tc]
        except Exception as e:
            row = [float('nan'), 0, float('nan'), '', '', float('nan')]; print('fail', cf, repr(e)[:150], flush=True)
        w.writerow([shape, cf['cfg'], cf['BM'], cf['BN'], cf['BK'], cf['G'], cf['num_warps'], cf['num_stages'], *row, time.time()]); f.flush()
    f.close()
    print('screened', shape, len(cfgs), f'precompile {tpre:.1f}s total {time.time() - t0:.1f}s', flush=True)
shutil.rmtree(TC, ignore_errors=True)
