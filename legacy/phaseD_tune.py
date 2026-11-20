"""Phase D tuning for one shape (subprocess). argv: shape_key model_choice gpu. Writes runs/phaseD_tune/<shape>.json"""
import sys, os, json, time, shutil, tempfile
R = '/home/min/a/cashman3/lala-triton-killtest'
shape, choice, gpu = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
ind = f'{R}/runs/phaseD_ind/{shape}'
shutil.rmtree(ind, ignore_errors=True); os.makedirs(ind)
os.environ['TORCHINDUCTOR_CACHE_DIR'] = ind
os.environ['TRITON_CACHE_DIR'] = f'{ind}/triton'
sys.path.insert(0, R)
import torch, pandas as pd
import torch._inductor.config as ic
ic.max_autotune_gemm_backends = 'ATEN,TRITON'
from triton.testing import do_bench
from common import make_inputs
from kernels.matmul_grouped import matmul, matmul_kernel
dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
a, b, c = make_inputs(M, N, K, dt, layout=lay)
out = dict(shape=shape, choice=choice, tuning={}, picks={}, bench={})
# ---- B2 inductor (fresh cache); capture autotune table from stderr
one = torch.tensor(1.0, device='cuda')
if dt == 'fp8':
    f2 = torch.compile(lambda x, y: torch._scaled_mm(x, y, scale_a=one, scale_b=one, out_dtype=torch.float16), mode='max-autotune-no-cudagraphs')
else:
    f2 = torch.compile(lambda x, y: x @ y, mode='max-autotune-no-cudagraphs')
logp = f'{R}/runs/phaseD_tune/{shape}.inductor.log'
fd = os.dup(2); lf = os.open(logp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC); os.dup2(lf, 2)
t0 = time.time()
try:
    f2(a, b); torch.cuda.synchronize(); ok = True
except Exception as e:
    ok = repr(e)[:300]
tw = time.time() - t0
os.dup2(fd, 2); os.close(lf)
lines = open(logp).read().splitlines()
tab = [l for l in lines if l.startswith('  ') and ' ms ' in l]
nchoices = [l for l in lines if 'AUTOTUNE benchmarking takes' in l]
out['tuning']['B2'] = dict(wall_s=tw, ok=ok, n_configs=(int(nchoices[0].split('for ')[1].split(' ')[0]) if nchoices else None))
out['picks']['B2'] = dict(kind='inductor', top=tab[0].strip() if tab else None, table=[l.strip() for l in tab[:10]])

def bench_cfgs(name, cfgs, warmup, rep, max_ok=None):
    """Compile + benchmark configs on our kernel with a fresh triton cache; returns list of (cfg, ms|None), wall."""
    os.environ['TRITON_CACHE_DIR'] = tempfile.mkdtemp(prefix=f'tc_{name}_', dir=f'{ind}')
    matmul_kernel.cache.clear() if hasattr(matmul_kernel, 'cache') else None
    t0 = time.time(); res = []
    for cfg in cfgs:
        if max_ok is not None and sum(1 for _, m in res if m is not None) >= max_ok: break
        try:
            k = matmul(a, b, c, cfg['BM'], cfg['BN'], cfg['BK'], cfg['G'], cfg['num_warps'], cfg['num_stages'])
            if k.n_spills > 0: res.append((cfg, None)); continue
            ms = do_bench(lambda: matmul(a, b, c, cfg['BM'], cfg['BN'], cfg['BK'], cfg['G'], cfg['num_warps'], cfg['num_stages']),
                          warmup=warmup, rep=rep, return_mode='median')
            res.append((cfg, ms))
        except Exception as e:
            res.append((cfg, None))
    return res, time.time() - t0

def best(res):
    ok = [(c, m) for c, m in res if m is not None]
    return min(ok, key=lambda x: x[1])[0] if ok else None

# ---- B3 tutorial list (triton.autotune defaults: do_bench warmup=25, rep=100)
TUT = [(128, 256, 64, 3, 8), (64, 256, 32, 4, 4), (128, 128, 32, 4, 4), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (128, 32, 32, 4, 4),
       (64, 32, 32, 5, 2), (32, 64, 32, 5, 2), (128, 256, 128, 3, 8), (256, 128, 128, 3, 8), (256, 64, 128, 4, 4), (64, 256, 128, 4, 4),
       (128, 128, 128, 4, 4), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 32, 64, 4, 4)]
tut = [dict(BM=bm, BN=bn, BK=bk, G=8, num_warps=w, num_stages=s) for bm, bn, bk, s, w in TUT]
res, wall = bench_cfgs('B3', tut, 25, 100)
out['tuning']['B3'] = dict(wall_s=wall, n_configs=len(res)); out['picks']['B3'] = best(res); out['bench']['B3'] = res
# ---- Ours: model top-5 (do_bench warmup=5, rep=20)
pred = pd.read_csv(f'{R}/predictions/phaseDE_pred.csv')
pr = pred[(pred.phase == 'D') & (pred['shape'] == shape)].sort_values(f't_{choice.lower()}')
top = [dict(BM=int(r.BM), BN=int(r.BN), BK=int(r.BK), G=int(r.G), num_warps=int(r.num_warps), num_stages=int(r.num_stages)) for r in pr.head(5).itertuples()]
res, _ = bench_cfgs('ours', top, 5, 20)
# wall per prefix k: re-derive by timing compile+bench of prefixes is equivalent to cumulative; record per-config cumulative
out['bench']['ours'] = res
out['picks']['ours1'] = top[0]; out['picks']['ours3'] = best(res[:3]); out['picks']['ours5'] = best(res[:5])
# ---- NVMMH: top mappable configs (ranked), first 5 that compile w/o spills
nv = pd.read_csv(f'{R}/predictions/phaseF_nvmmh_picks.csv'); nv = nv[(nv['shape'] == shape) & nv.unmappable.isna()].sort_values('rank')
nvc = [dict(BM=int(r.BM), BN=int(r.BN), BK=int(r.BK), G=int(r.G), num_warps=int(r.num_warps), num_stages=int(r.num_stages)) for r in nv.itertuples()]
res, _ = bench_cfgs('nvmmh', nvc, 5, 20, max_ok=5)
okres = [(c, m) for c, m in res if m is not None]
out['bench']['nvmmh'] = res
out['picks']['nvmmh1'] = okres[0][0] if okres else None; out['picks']['nvmmh5'] = best(okres[:5])
# ---- wall-clock tuning cost for ours/nvmmh prefixes: measured separately, cold, so each prefix is honest
for name, cfgs, ks in [('ours', top, (1, 3, 5)), ('nvmmh', [c for c, _ in okres], (1, 5))]:
    for k in ks:
        r2, w2 = bench_cfgs(f'{name}{k}c', cfgs[:k], 5, 20) if k > 1 else (None, None)
        if k == 1:   # top-1: compile only, no benchmarking
            os.environ['TRITON_CACHE_DIR'] = tempfile.mkdtemp(prefix=f'tc_{name}1c_', dir=ind); matmul_kernel.cache.clear()
            t0 = time.time(); cc = cfgs[0]
            matmul(a, b, c, cc['BM'], cc['BN'], cc['BK'], cc['G'], cc['num_warps'], cc['num_stages']); torch.cuda.synchronize(); w2 = time.time() - t0
        out['tuning'][f'{name}{k}'] = dict(wall_s=w2, n_configs=min(k, len(cfgs)))
json.dump(out, open(f'{R}/runs/phaseD_tune/{shape}.json', 'w'), indent=1, default=str)
print('tuned', shape, {k: round(v['wall_s'], 1) for k, v in out['tuning'].items()})
