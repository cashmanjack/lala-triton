"""Round-3 Step 2d (dynamic M inside the template family), GPU1, bf16 + fp8, A@B.T. Resumable.
Stage 1: screening of E(M) at every M (+ bucket midpoints) with dynamic-M harness kernels, do_bench(5, 10), 1 rep.
Stage 2: picks (static@512, buckets, ours top-1, oracle top-3 by screening).
Stage 3: sustained final timing (3 interleaved reps) of every pick + Inductor dynamic mode, per (dtype, N, K, M)."""
import sys, os, csv, json, random, time, subprocess
R = '/home/min/a/cashman3/lala-triton-killtest'
GPU = '1'
sys.path.insert(0, R)
import pandas as pd
from phaseD_shapes import key
from r3_lib import Worker, append, done_set
EM = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ENK = [(4096, 4096), (14336, 4096), (4096, 14336)]
BUCKETS = [(1, 16, 8), (17, 128, 72), (129, 1024, 576), (1025, 10**9, 4608)]
SCR, PICKS, OUT, COR = f'{R}/runs/r3_E_screen.csv', f'{R}/runs/r3_E_picks.json', f'{R}/runs/r3_E_eval.csv', f'{R}/runs/r3_E_correct.csv'
pred = pd.read_csv(f'{R}/predictions/r3_pred.csv'); pred = pred[pred.phase == 'E']
DTS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['bf16', 'fp8']


def mclass(m): return 1 if m == 1 else (16 if m % 16 == 0 else 3)


# ------------------------------------------------------------- stage 1: screening
def screen(dt, N, K):
    sub = pred[(pred.dtype == dt) & (pred.N == N) & (pred.K == K)]
    done = done_set(SCR, ['dtype', 'N', 'K', 'M', 'cfg'])
    todo = sub[[(dt, str(N), str(K), str(m), c) not in done for m, c in zip(sub.M, sub.cfg)]]
    if not len(todo): return
    tc = f'/var/tmp/cashman3_r3/tc_E/{dt}_{N}_{K}'
    cl = {}
    for r in todo.itertuples():
        d = cl.setdefault(r.cfg, dict(BM=int(r.BM), BN=int(r.BN), BK=int(r.BK), G=int(r.G), num_warps=int(r.num_warps), num_stages=int(r.num_stages), Ms=[]))
        if mclass(r.M) not in [mclass(x) for x in d['Ms']]: d['Ms'].append(int(r.M))
    cfgs = list(cl.values()); shape = key(512, N, K, dt, 'ABT')
    procs = [subprocess.Popen(['/usr/bin/python3', f'{R}/r3_precompile.py', shape, json.dumps(cfgs[i::12]), 'dynm'],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, TRITON_CACHE_DIR=tc), stdout=subprocess.DEVNULL,
                              stderr=open(f'{R}/logs/r3/precompile.err', 'a')) for i in range(12) if cfgs[i::12]]
    for p in procs: p.wait()
    subprocess.run(['/usr/bin/python3', f'{R}/r3_phaseE_screen.py', dt, str(N), str(K), GPU], env=dict(os.environ, TRITON_CACHE_DIR=tc))


# ------------------------------------------------------------- stage 2: picks
def picks(dt, N, K):
    s = pd.read_csv(SCR); s = s[(s.dtype == dt) & (s.N == N) & (s.K == K) & (s.ok == 1)]
    best = lambda m: s[s.M == m].sort_values('ms').cfg.iloc[0]
    out = {}
    for M in EM:
        p = pred[(pred.dtype == dt) & (pred.N == N) & (pred.K == K) & (pred.M == M)].dropna(subset=['rank_s4']).sort_values('rank_s4')
        bmid = [mid for lo, hi, mid in BUCKETS if lo <= M <= hi][0]
        out[str(M)] = dict(static=best(512), bucket=best(bmid), ours=p.cfg.iloc[0],
                           oracle=list(s[s.M == M].sort_values('ms').cfg.head(3)))
    return out


# ------------------------------------------------------------- stage 3: sustained final timing
def final(dt, N, K, pk):
    done = done_set(OUT, ['dtype', 'N', 'K', 'M', 'launcher', 'rep'])
    dcor = done_set(COR, ['dtype', 'N', 'K', 'M', 'launcher'])
    w = Worker('env1', GPU, 'E', dict(TORCHINDUCTOR_CACHE_DIR=f'/var/tmp/cashman3_r3/ind_dyn/{dt}_{N}_{K}',
                                      TRITON_CACHE_DIR=f'/var/tmp/cashman3_r3/tc_E/{dt}_{N}_{K}'))
    for M in EM:
        p = pk[str(M)]
        meth = {'static': 'tmplD:' + p['static'], 'bucket': 'tmplD:' + p['bucket'], 'ours': 'tmplD:' + p['ours'], 'inductor_dyn': 'ind_dyn'}
        for i, c in enumerate(p['oracle']): meth[f'oracle_c{i + 1}'] = 'tmplD:' + c
        lids = {}
        for m, l in meth.items(): lids.setdefault(l, []).append(m)
        sk = key(M, N, K, dt, 'ABT')
        w.ask(cmd='setup', shape=sk)
        good = []
        for lid in sorted(lids):
            r = w.ask(cmd='correct', shape=sk, launcher=lid)
            if (dt, str(N), str(K), str(M), lid) not in dcor:
                append(COR, dict(dtype=dt, N=N, K=K, M=M, launcher=lid, methods='|'.join(lids[lid]), error=r.get('error', ''),
                                 R=r.get('R'), pass_=r.get('pass_'), e_F=r.get('e_F'), kernels=r.get('kernels'), ts=time.time()))
            if 'error' in r: print('  fail', sk, lid, r['error'][:200], flush=True)
            else: good.append(lid)
        for rep in range(3):
            order = sorted(good); random.Random(f'R3E-{sk}-{rep}').shuffle(order)
            for lid in order:
                if (dt, str(N), str(K), str(M), lid, str(rep)) in done: continue
                r = w.ask(cmd='time', shape=sk, launcher=lid, rep=rep)
                if not r.get('ok'): print('  time failed', sk, lid, r.get('error'), flush=True); continue
                r.pop('ok')
                append(OUT, dict(dtype=dt, N=N, K=K, M=M, launcher=lid, methods='|'.join(lids[lid]), rep=rep, **r))
        print('E final', sk, time.strftime('%H:%M:%S'), flush=True)
    w.close()


allp = json.load(open(PICKS)) if os.path.exists(PICKS) else {}
for dt in DTS:
    for (N, K) in ENK:
        screen(dt, N, K); print('screened', dt, N, K, time.strftime('%H:%M:%S'), flush=True)
        allp[f'{dt}_{N}_{K}'] = picks(dt, N, K); json.dump(allp, open(PICKS, 'w'), indent=1)
for dt in DTS:
    for (N, K) in ENK:
        final(dt, N, K, allp[f'{dt}_{N}_{K}'])
print('PHASE E DONE', flush=True)
