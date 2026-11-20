"""One cold Inductor tuning run (fresh process, fresh cache). argv: shape mode(S1|S4|S5) k cache_dir out_json gpu
S1 = stock max-autotune (ATEN,TRITON). S4/S5 = patched template config list = model top-k (TRITON only)."""
import sys, os, json, time
R = '/home/min/a/cashman3/lala-triton-killtest'
shape, mode, k, cdir, outp, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ['TORCHINDUCTOR_CACHE_DIR'] = cdir
os.environ['TRITON_CACHE_DIR'] = f'{cdir}/triton'
sys.path.insert(0, R)
import torch, pandas as pd
import torch._inductor.config as ic
torch.backends.cuda.matmul.allow_tf32 = False
from common import make_inputs
dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
a, b, c = make_inputs(M, N, K, dt, layout=lay)
cfgs = None
if mode == 'S1':
    ic.max_autotune_gemm_backends = 'ATEN,TRITON'
else:
    import inductor_patch
    ic.max_autotune_gemm_backends = 'TRITON'
    p = pd.read_csv(f'{R}/predictions/r3_pred.csv'); p = p[(p.phase == 'D') & (p['shape'] == shape)]
    rk = 'rank_s4' if mode == 'S4' else 'rank_s5'
    p = p.dropna(subset=[rk]).sort_values(rk).head(k)
    cfgs = [(int(r.BM), int(r.BN), int(r.BK), int(r.num_stages), int(r.num_warps), int(r.G)) for r in p.itertuples()]
    inductor_patch.set_configs(cfgs)
one = torch.tensor(1.0, device='cuda')
f = (lambda x, y: torch._scaled_mm(x, y, scale_a=one, scale_b=one, out_dtype=torch.float16)) if dt == 'fp8' else (lambda x, y: x @ y)
f2 = torch.compile(f, mode='max-autotune-no-cudagraphs')
logp = outp + '.log'
fd = os.dup(2); lf = os.open(logp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC); os.dup2(lf, 2)
t0 = time.time()
try:
    f2(a, b); torch.cuda.synchronize(); ok = True
except Exception as e:
    ok = repr(e)[:300]
wall = time.time() - t0
os.dup2(fd, 2); os.close(lf)
lines = open(logp).read().splitlines()
tab = [l.strip() for l in lines if l.startswith('  ') and ' ms ' in l]
nch = [l for l in lines if 'AUTOTUNE benchmarking takes' in l]
n_bench = int(nch[0].split('for ')[1].split(' ')[0]) if nch else (len(cfgs) if cfgs is not None and len(cfgs) <= 1 else None)
# Triton 3.2 short-ptr process-state quirk (Amendment 3b): which state did Inductor's template kernels compile in?
import glob
ll = [open(p).read(2000) for p in glob.glob(f'{cdir}/triton/**/triton_tem*.llir', recursive=True)]
json.dump(dict(shape=shape, mode=mode, k=k, wall_s=wall, ok=ok, n_configs=n_bench, cfgs=cfgs, top=tab[0] if tab else None, table=tab[:20],
               n_tmpl_llir=len(ll), n_tmpl_short_ptr=sum('p3:32:32' in x for x in ll)),
          open(outp, 'w'), indent=1)
print(mode, k, shape, round(wall, 2), ok, tab[0][:120] if tab else '')
