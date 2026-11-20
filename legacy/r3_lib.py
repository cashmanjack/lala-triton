"""Shared helpers for round-3 drivers: persistent env workers, append-only CSV, Inductor log parsing."""
import os, json, csv, re, subprocess
R = '/home/min/a/cashman3/lala-triton-killtest'
PY = {'env1': '/usr/bin/python3', 'env2': '/var/tmp/cashman3_r3/venv_new/bin/python'}


class Worker:
    def __init__(self, env, gpu, tag, extra_env=None):
        e = dict(os.environ); e.update(TRITON_CACHE_DIR=f'{R}/.triton_cache', TMPDIR='/var/tmp/cashman3_r3/tmp'); e.update(extra_env or {})
        self.env = env
        self.p = subprocess.Popen([PY[env], f'{R}/r3_worker.py', gpu, env], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(f'{R}/logs/r3/worker_{tag}_{env}.err', 'a'), text=True, env=e)

    def ask(self, **msg):
        self.p.stdin.write(json.dumps(msg) + '\n'); self.p.stdin.flush()
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError(f'worker {self.env} died')
            if l.startswith('@@'): return json.loads(l[2:])

    def close(self):
        try: self.ask(cmd='bye')
        except Exception: pass
        self.p.wait(timeout=60)


_fields = {}


def append(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if path not in _fields:
        _fields[path] = list(row) if new else next(csv.reader(open(path)))
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_fields[path], extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(row)


def done_set(path, keys):
    return {tuple(r[k] for k in keys) for r in csv.DictReader(open(path))} if os.path.exists(path) else set()


def parse_ind_line(l):
    """Inductor autotune table line -> config dict (template choice) or {'extern': name}."""
    name = l.split()[0]
    if 'BLOCK_M' not in l: return dict(extern=name)
    g = lambda k: int(re.search(rf'{k}=(\d+)', l).group(1))
    return dict(BM=g('BLOCK_M'), BN=g('BLOCK_N'), BK=g('BLOCK_K'), G=g('GROUP_M'), num_warps=g('num_warps'), num_stages=g('num_stages'))


def ind_line_ms(l):
    return float(l.split()[1])
