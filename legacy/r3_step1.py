"""Round-3 Step 1 driver (GPU0): cuBLAS variants in two envs + round-2 Ours-top5, correctness (fp64 ref) then
3 interleaved sustained reps per shape. Resumable, append-only CSVs. argv: [shape_filter_substring]"""
import sys, os, json, csv, random, subprocess, time
R = '/home/min/a/cashman3/lala-triton-killtest'
sys.path.insert(0, R)
from phaseD_shapes import SHAPES, key
from common import cfg_key
PY = {'env1': '/usr/bin/python3', 'env2': '/var/tmp/cashman3_r3/venv_new/bin/python'}
GPU = '0'
OUTD = os.environ.get('R3_OUT', f'{R}/runs')
FCOR, FTIM = f'{OUTD}/r3_step1_correct.csv', f'{OUTD}/r3_step1_time.csv'
flt = sys.argv[1] if len(sys.argv) > 1 else ''


class Worker:
    def __init__(self, env):
        e = dict(os.environ, TRITON_CACHE_DIR=f'{R}/.triton_cache', TMPDIR='/var/tmp/cashman3_r3/tmp')
        self.env = env
        self.p = subprocess.Popen([PY[env], f'{R}/r3_worker.py', GPU, env], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(f'{R}/logs/r3/worker_step1_{env}.err', 'a'), text=True, env=e)
    def ask(self, **msg):
        self.p.stdin.write(json.dumps(msg) + '\n'); self.p.stdin.flush()
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError(f'worker {self.env} died')
            if l.startswith('@@'): return json.loads(l[2:])


def append(path, row, fields_cache={}):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        if path not in fields_cache:
            fields_cache[path] = list(row) if new else next(csv.reader(open(path)))
        w = csv.DictWriter(f, fieldnames=fields_cache[path], extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(row)


def done_set(path, keys):
    return {tuple(r[k] for k in keys) for r in csv.DictReader(open(path))} if os.path.exists(path) else set()


W = {e: Worker(e) for e in PY}
dcor = done_set(FCOR, ['shape', 'env', 'launcher'])
dtim = done_set(FTIM, ['shape', 'env', 'launcher', 'rep'])
for s in SHAPES:
    sk = key(*s)
    if flt not in sk: continue
    dt = s[3]
    tune = json.load(open(f'{R}/runs/phaseD_tune/{sk}.json'))
    ours = 'ours:' + cfg_key(tune['picks']['ours5'])
    L = [('env1', 'cublas'), ('env1', ours), ('env2', 'cublas'), ('env2', 'nvmath')]
    if dt == 'bf16':
        L += [('env1', 'cublas_rpr0'), ('env1', 'cublaslt'), ('env2', 'cublaslt')]
    sig = {e: W[e].ask(cmd='setup', shape=sk) for e in W}
    print(sk, {e: (v.get('input_sig'), v.get('torch')) for e, v in sig.items()}, flush=True)
    good = []
    for env, lid in L:
        r = W[env].ask(cmd='correct', shape=sk, launcher=lid)
        if (sk, env, lid) not in dcor:
            nv = r.pop('nvmath', None)
            append(FCOR, dict(shape=sk, dtype=dt, env=env, launcher=lid, ok_run=r.get('ok'), error=r.get('error', ''),
                              out_dtype=r.get('out_dtype'), maxabs=r.get('maxabs'), e_max=r.get('e_max'), e_F=r.get('e_F'),
                              R=r.get('R'), pass_=r.get('pass_'), refmax=r.get('refmax'),
                              worst_ref=r.get('worst_ref'), worst_err=r.get('worst_err'), kernels=r.get('kernels'),
                              flag_rpr=r.get('flag_rpr'), flag_blas=r.get('flag_blas'), flag_allow_tf32=r.get('flag_allow_tf32'),
                              torch=r.get('flag_torch'), cuda=r.get('flag_cuda'), cublas=r.get('flag_cublas'), nvmath=json.dumps(nv) if nv else '',
                              input_sig=json.dumps(sig[env].get('input_sig')), ts=time.time()))
        if 'error' not in r: good.append((env, lid))
        else: print('  launcher failed', env, lid, r['error'][:200], flush=True)
    for rep in range(3):
        order = sorted(good); random.Random(f'R3S1-{sk}-{rep}').shuffle(order)
        for env, lid in order:
            if (sk, env, lid, str(rep)) in dtim: continue
            r = W[env].ask(cmd='time', shape=sk, launcher=lid, rep=rep)
            if not r.get('ok'): print('  time failed', env, lid, r.get('error'), flush=True); continue
            r.pop('ok'); append(FTIM, dict(shape=sk, dtype=dt, layout=s[4], env=env, launcher=lid, rep=rep, **r))
    print('done', sk, time.strftime('%H:%M:%S'), flush=True)
for w in W.values(): w.ask(cmd='bye')
print('STEP1 DONE', flush=True)
