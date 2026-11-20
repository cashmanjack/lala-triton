"""Phase D evaluation for one shape (subprocess): re-time each method's pick with the shortened sustained protocol,
interleaved randomly, 3 reps. argv: shape_key gpu. Appends to runs/phaseD_eval.csv."""
import sys, os, json, csv, random
R = '/home/min/a/cashman3/lala-triton-killtest'
shape, gpu = sys.argv[1], sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ['TORCHINDUCTOR_CACHE_DIR'] = f'{R}/runs/phaseD_ind/{shape}'
os.environ['TRITON_CACHE_DIR'] = f'{R}/.triton_cache'
sys.path.insert(0, R)
import torch, numpy as np, pandas as pd
import torch._inductor.config as ic
ic.max_autotune_gemm_backends = 'ATEN,TRITON'
from common import make_inputs
from kernels.matmul_grouped import matmul
from sustained import make_graph, measure, buffer_sets
WIN = json.load(open(f'{R}/env/phaseD_window.json'))
tune = json.load(open(f'{R}/runs/phaseD_tune/{shape}.json'))
dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
picks = dict(tune['picks'])
# B4: round-1 oracle (AB / fp8) or pseudo-oracle (ABT bf16) = best do_bench among all our-kernel configs benchmarked in tuning
if lay == 'AB' or dt == 'fp8':
    h2 = pd.read_csv(f'{R}/runs/h2_per_shape.csv').set_index('shape')
    oc = h2.loc[f'{M}x{N}x{K}_{dt}', 'oracle_cfg']
    a_, g_, w_, s_ = oc.split('_'); bm, bn, bk = map(int, a_.split('x'))
    picks['B4'] = dict(BM=bm, BN=bn, BK=bk, G=int(g_[1:]), num_warps=int(w_[1:]), num_stages=int(s_[1:])); b4kind = 'oracle'
else:
    allb = [(c, m) for k in ('B3', 'ours', 'nvmmh') for c, m in tune['bench'][k] if m is not None]
    picks['B4'] = min(allb, key=lambda x: x[1])[0]; b4kind = 'pseudo-oracle'
def ckey(c): return f"{c['BM']}x{c['BN']}x{c['BK']}_G{c['G']}_w{c['num_warps']}_s{c['num_stages']}"
launch_ids = {}   # method -> launcher id
uniq = {}
for meth, c in picks.items():
    if meth == 'B2': lid = 'B2'
    elif c is None: continue
    else: lid = ckey(c); uniq[lid] = c
    launch_ids[meth] = lid
launch_ids['B1'] = 'B1'
OUT = f'{R}/runs/phaseD_eval.csv'
done = set()
if os.path.exists(OUT):
    done = {(r['shape'], r['launcher'], r['rep']) for r in csv.DictReader(open(OUT))}
sets = buffer_sets(M, N, K, dt, lay, make_inputs)
one = torch.tensor(1.0, device='cuda')
if dt == 'fp8':
    f2 = torch.compile(lambda x, y: torch._scaled_mm(x, y, scale_a=one, scale_b=one, out_dtype=torch.float16), mode='max-autotune-no-cudagraphs')
    f1 = lambda x, y: torch._scaled_mm(x, y, scale_a=one, scale_b=one, out_dtype=torch.float16)
else:
    f2 = torch.compile(lambda x, y: x @ y, mode='max-autotune-no-cudagraphs')
    f1 = lambda x, y: torch.mm(x, y)
def fns_for(lid):
    if lid == 'B1': return [lambda a=a, b=b: f1(a, b) for a, b, c in sets]
    if lid == 'B2': return [lambda a=a, b=b: f2(a, b) for a, b, c in sets]
    cf = uniq[lid]
    return [lambda a=a, b=b, c=c: matmul(a, b, c, cf['BM'], cf['BN'], cf['BK'], cf['G'], cf['num_warps'], cf['num_stages']) for a, b, c in sets]
# correctness (first buffer set)
a0, b0, c0 = sets[0]; ref = a0.float() @ b0.float(); refmax = ref.abs().max().item()
okmap = {}
for lid in set(launch_ids.values()):
    f = fns_for(lid)[0]; y = f(); y = c0 if y is None or not torch.is_tensor(y) else y
    if lid not in ('B1', 'B2'): y = c0
    torch.cuda.synchronize(); err = (y.float() - ref).abs().max().item()
    okmap[lid] = (err <= (1e-2 * refmax + 1e-2 if dt == 'bf16' else 5e-2 * refmax), err)
del ref
f = open(OUT, 'a', newline=''); wr = None
for rep in range(3):
    lids = sorted(set(launch_ids.values())); random.Random(f'D-{shape}-{rep}').shuffle(lids)
    for lid in lids:
        if (shape, lid, str(rep)) in done: continue
        g, L, est = make_graph(fns_for(lid))
        res, _ = measure(g, L, 0 if gpu == '0' else int(gpu), warmup_s=WIN['warmup_s'], window_s=WIN['window_s'], cooldown_s=WIN['cooldown_s'])
        meths = [m for m, l in launch_ids.items() if l == lid]
        row = dict(shape=shape, M=M, N=N, K=K, dtype=dt, layout=lay, launcher=lid, methods='|'.join(sorted(meths)), rep=rep,
                   ok=int(okmap[lid][0]), maxerr=okmap[lid][1], b4kind=b4kind, nsets=len(sets), **res)
        if wr is None:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if not done and os.path.getsize(OUT) == 0: wr.writeheader()
        wr.writerow(row); f.flush(); del g; torch.cuda.empty_cache()
print('eval done', shape, launch_ids)
