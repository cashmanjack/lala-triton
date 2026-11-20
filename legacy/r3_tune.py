"""Round-3 Step 2c tuning + screening on GPU1, all 62 Phase D shapes, resumable.
Per shape: S1 / S4-{1,3,5} / S5-{1,3,5} cold Inductor runs (r3_tune_one.py, each a fresh process + cache), then
exhaustive harness screening of E(shape) with do_bench(10, 50, median), seeded random order -> runs/r3_screen.csv."""
import sys, os, json, csv, random, subprocess, time, shutil
R = '/home/min/a/cashman3/lala-triton-killtest'
GPU = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
sys.path.insert(0, R)
from phaseD_shapes import SHAPES, key
TD = f'{R}/runs/r3_tune'; os.makedirs(TD, exist_ok=True)
IND = '/var/tmp/cashman3_r3/ind'          # S1 caches are kept for eval reuse; others deleted
SCR = f'{R}/runs/r3_screen.csv'
flt = sys.argv[1] if len(sys.argv) > 1 else ''


def precompile(shape, cfgs, tc):
    """Warm the Triton cache for the screening configs in parallel (compile only, no launches)."""
    chunks = [cfgs[i::12] for i in range(12)]
    procs = []
    for i, ch in enumerate(chunks):
        if not ch: continue
        procs.append(subprocess.Popen(['/usr/bin/python3', f'{R}/r3_precompile.py', shape, json.dumps(ch)],
                                      env=dict(os.environ, TRITON_CACHE_DIR=tc), stdout=subprocess.DEVNULL, stderr=open(f'{R}/logs/r3/precompile.err', 'a')))
    for p in procs: p.wait()


for s in SHAPES:
    sk = key(*s)
    if flt not in sk: continue
    # ---- cold tuning runs
    for mode, ks in (('S1', [0]), ('S4', [1, 3, 5]), ('S5', [1, 3, 5])):
        for k in ks:
            outp = f'{TD}/{sk}.{mode}{k}.json'
            if os.path.exists(outp): continue
            cdir = f'{IND}/{sk}/{mode}{k}'; shutil.rmtree(cdir, ignore_errors=True); os.makedirs(cdir)
            subprocess.run(['/usr/bin/python3', f'{R}/r3_tune_one.py', sk, mode, str(k), cdir, outp, GPU],
                           stdout=open(f'{R}/logs/r3/tune_one.log', 'a'), stderr=subprocess.STDOUT)
            if mode != 'S1': shutil.rmtree(cdir, ignore_errors=True)
    # ---- screening (subprocess so CUDA state is fresh per shape)
    subprocess.run(['/usr/bin/python3', f'{R}/r3_screen.py', sk, GPU], stdout=open(f'{R}/logs/r3/screen.log', 'a'), stderr=subprocess.STDOUT)
    print('tuned', sk, time.strftime('%H:%M:%S'), flush=True)
print('TUNE DONE', flush=True)
