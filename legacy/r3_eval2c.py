"""Round-3 Step 2c final evaluation (GPU0): sustained protocol, 3 interleaved reps per shape. Resumable.
Launchers: S1 stock Inductor (its tuning cache reused), harness template kernels for S2/S3 candidates (top-3 by
screening) and S4-k/S5-k picks, S6 = best passing cuBLAS variant from Step 1 (re-run in its env)."""
import sys, os, json, random, time
R = '/home/min/a/cashman3/lala-triton-killtest'
sys.path.insert(0, R)
import pandas as pd
from phaseD_shapes import SHAPES, key
from common import cfg_key
from r3_lib import Worker, append, done_set, parse_ind_line
GPU = '0'
OUT, COR = f'{R}/runs/r3_2c_eval.csv', f'{R}/runs/r3_2c_correct.csv'
flt = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None   # exact shape keys
scr = pd.read_csv(f'{R}/runs/r3_screen.csv')
s1t = pd.read_csv(f'{R}/runs/r3_step1_time.csv'); s1c = pd.read_csv(f'{R}/runs/r3_step1_correct.csv')
done = done_set(OUT, ['shape', 'launcher', 'rep'])
dcor = done_set(COR, ['shape', 'launcher'])


def best_cublas(sk):
    """Best passing cuBLAS variant (pre-registered). If none passes 1a (all fp8 shapes: sm_89 fp8 MMA accumulates below fp32),
    fall back to the fastest variant and label it 'S6np' (descriptive only)."""
    cu = s1c[(s1c['shape'] == sk) & (~s1c.launcher.str.startswith('ours'))]
    for lab, ok in (('S6', cu[cu.pass_ == 1]), ('S6np', cu)):
        t = s1t[s1t['shape'] == sk].merge(ok[['env', 'launcher']], on=['env', 'launcher']).groupby(['env', 'launcher']).ms_per_gemm.mean()
        if len(t): return lab, t.idxmin()
    return None, None


for s in SHAPES:
    sk = key(*s)
    if flt is not None and sk not in flt: continue
    sc = scr[(scr['shape'] == sk) & (scr.ok == 1)].sort_values('ms')
    meth = {'S1': ('env1', 'ind_stock')}
    for i, c in enumerate(sc[sc.G == 8].cfg.head(3)): meth[f'S2c{i + 1}'] = ('env1', 'tmpl:' + c)
    for i, c in enumerate(sc.cfg.head(3)): meth[f'S3c{i + 1}'] = ('env1', 'tmpl:' + c)
    for mode in ('S4', 'S5'):
        for k in (1, 3, 5):
            j = json.load(open(f'{R}/runs/r3_tune/{sk}.{mode}{k}.json'))
            if k == 1 and j['cfgs']:
                bm, bn, bk, st, nw, g = j['cfgs'][0]; c = dict(BM=bm, BN=bn, BK=bk, G=g, num_warps=nw, num_stages=st)
            else:
                c = parse_ind_line(j['top'])
            meth[f'{mode}_{k}'] = ('env1', 'tmpl:' + cfg_key(c))
    lab, bc = best_cublas(sk)
    if bc: meth[lab] = bc
    lids = {}
    for m, el in meth.items(): lids.setdefault(el, []).append(m)
    W = {'env1': Worker('env1', GPU, '2c', dict(TORCHINDUCTOR_CACHE_DIR=f'/var/tmp/cashman3_r3/ind/{sk}/S10',
                                               TRITON_CACHE_DIR=f'/var/tmp/cashman3_r3/ind/{sk}/S10/triton'))}
    if any(e == 'env2' for e, _ in lids): W['env2'] = Worker('env2', GPU, '2c')
    for e in W: W[e].ask(cmd='setup', shape=sk)
    good = []
    for (env, lid), ms in sorted(lids.items()):
        r = W[env].ask(cmd='correct', shape=sk, launcher=lid)
        if (sk, lid) not in dcor:
            append(COR, dict(shape=sk, env=env, launcher=lid, methods='|'.join(ms), error=r.get('error', ''), maxabs=r.get('maxabs'),
                             e_max=r.get('e_max'), e_F=r.get('e_F'), R=r.get('R'), pass_=r.get('pass_'), kernels=r.get('kernels'), ts=time.time()))
        if 'error' in r: print('  fail', env, lid, r['error'][:200], flush=True)
        else: good.append((env, lid))
    for rep in range(3):
        order = sorted(good); random.Random(f'R3C-{sk}-{rep}').shuffle(order)
        for env, lid in order:
            if (sk, lid, str(rep)) in done: continue
            r = W[env].ask(cmd='time', shape=sk, launcher=lid, rep=rep)
            if not r.get('ok'): print('  time failed', lid, r.get('error'), flush=True); continue
            r.pop('ok')
            append(OUT, dict(shape=sk, dtype=s[3], layout=s[4], env=env, launcher=lid, methods='|'.join(lids[(env, lid)]), rep=rep, **r))
    for w in W.values(): w.close()
    print('done', sk, len(good), time.strftime('%H:%M:%S'), flush=True)
print('EVAL2C DONE', flush=True)
