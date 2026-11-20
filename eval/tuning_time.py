"""GEMM-level tuning-cost breakdown: each (method, shape) first call in a FRESH process with fresh Triton + Inductor caches.
python3 eval/tuning_time.py --profile ... --gpu 0 --out results/tuning_time.csv   (resumable)
columns: select_s, compile_s (= first call - benchmark), bench_s (Inductor's 'benchmarking takes'), total (first_call_s)."""
import argparse, csv, json, os, shutil, subprocess, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--profile', required=True); ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--out', default='results/tuning_time.csv'); ap.add_argument('--ms', default='32,2048'); ap.add_argument('--dtypes', default='bf16,fp8')
ap.add_argument('--methods', default='ours1,ours5,ind_auto'); ap.add_argument('--model', default='llama3-8b')
a = ap.parse_args()
sys.path.insert(0, os.path.join(HERE, '..')); from lala_triton.models import gemm_shapes
done = set()
if os.path.exists(a.out): done = {(r['method'], r['gemm'], r['dtype'], r['M']) for r in csv.DictReader(open(a.out))}
for g, dt, M, me in itertools.product(gemm_shapes(a.model), a.dtypes.split(','), a.ms.split(','), a.methods.split(',')):
    if (me, g, dt, M) in done: continue
    tmp = f'/var/tmp/lala_triton_tt_{os.getpid()}'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    env = dict(os.environ, TRITON_CACHE_DIR=f'{tmp}/triton', TORCHINDUCTOR_CACHE_DIR=f'{tmp}/inductor')
    p = subprocess.run([sys.executable, f'{HERE}/run_eval.py', '--profile', a.profile, '--gpu', str(a.gpu), '--first-call', f'{me}:{a.model}:{g}:{dt}:{M}'],
                       capture_output=True, text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith('@@')]
    shutil.rmtree(tmp, ignore_errors=True)
    if not line: print('FAIL', me, g, dt, M, p.stderr[-300:], flush=True); continue
    r = json.loads(line[0][2:]); new = not os.path.exists(a.out)
    import re
    r['bench_s'] = sum(float(v) for v in re.findall(r'benchmarking takes ([\d.]+) seconds', p.stderr))     # Inductor logs this to stderr
    r['compile_s'] = r['first_call_s'] - r['bench_s'] - r['select_s']
    with open(a.out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(r)); 
        if new: w.writeheader()
        w.writerow(r)
    print(r, flush=True)
