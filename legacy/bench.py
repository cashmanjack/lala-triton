"""Resumable timing runner. Usage: bench.py <jobfile.json> <out.csv> <gpu>
jobfile: {"shapes": [[M,N,K,dtype],...], "configs": [cfg,...] or "feasible", "reps": 3, "seed": 0,
          "noise_cfg": cfg|null, "noise_n": 10, "subsample": null|int}"""
import sys, os, csv, json, random, time
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
job = json.load(open(sys.argv[1])); out = sys.argv[2]; gpu = int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
os.environ.setdefault('TRITON_CACHE_DIR', '/home/min/a/cashman3/lala-triton-killtest/.triton_cache')
import torch, pandas as pd
from triton.testing import do_bench
from common import *
from kernels.matmul_grouped import matmul
torch.backends.cuda.matmul.allow_tf32 = False

comp = pd.read_csv(f'{ROOT}/runs/compile_all.csv')
feas = comp[(comp.ok == 1) & (comp.n_spills == 0)]
done = set()
if os.path.exists(out):
    for r in csv.DictReader(open(out)): done.add((r['shape'], r['cfg'], r['rep']))
fields = ['shape', 'M', 'N', 'K', 'dtype', 'cfg', 'tile', 'BM', 'BN', 'BK', 'G', 'num_warps', 'num_stages', 'occ',
          'rep', 'ms', 'ok', 'maxerr', 'gpu', 'sm_clk', 'mem_clk', 'temp', 'ts']
f = open(out, 'a', newline=''); wr = csv.DictWriter(f, fieldnames=fields)
if not done: wr.writeheader()
failed = set()
for (M, N, K, dtype) in job['shapes']:
    shape = f'{M}x{N}x{K}_{dtype}'
    fd = feas[feas.dtype == dtype]
    cfgs = fd.to_dict('records') if job['configs'] == 'feasible' else \
        [r for r in fd.to_dict('records') if r['cfg'] in set(job['configs'])]
    if job.get('subsample'):
        cfgs = random.Random(job['seed']).sample(cfgs, min(job['subsample'], len(cfgs)))
    todo = [(c, rep) for rep in range(job['reps']) for c in cfgs if (shape, c['cfg'], str(rep)) not in done]
    if job.get('noise_cfg'):
        nc = [c for c in fd.to_dict('records') if c['cfg'] == job['noise_cfg']][0]
        todo += [(nc, f'noise{i}') for i in range(job['noise_n']) if (shape, nc['cfg'], f'noise{i}') not in done]
    if not todo: continue
    a, b, c_out = make_inputs(M, N, K, dtype)
    ref = a.float() @ b.float()
    refmax = ref.abs().max().item()
    checked = {}
    for rep in sorted(set(r for _, r in todo), key=str):
        batch = [c for c, r in todo if r == rep]
        random.Random(f"{job['seed']}-{shape}-{rep}").shuffle(batch)
        for c in batch:
            if c['cfg'] in failed: continue
            args = (c['BM'], c['BN'], c['BK'], c['G'], c['num_warps'], c['num_stages'])
            if c['cfg'] not in checked:
                c_out.zero_(); matmul(a, b, c_out, *args); torch.cuda.synchronize()
                err = (c_out.float() - ref).abs().max().item()
                tol = (1e-2 * refmax + 1e-2) if dtype == 'bf16' else 5e-2 * refmax
                checked[c['cfg']] = (err <= tol, err)
            okc, err = checked[c['cfg']]
            ms = do_bench(lambda: matmul(a, b, c_out, *args), warmup=10, rep=50, return_mode='median') if okc else float('nan')
            st = gpu_state(gpu)
            wr.writerow(dict(shape=shape, M=M, N=N, K=K, dtype=dtype, cfg=c['cfg'], tile=c['tile'], BM=c['BM'], BN=c['BN'],
                             BK=c['BK'], G=c['G'], num_warps=c['num_warps'], num_stages=c['num_stages'], occ=c['occ'],
                             rep=rep, ms=ms, ok=int(okc), maxerr=err, gpu=gpu, **st))
            f.flush()
    del a, b, c_out, ref; torch.cuda.empty_cache()
print('done')
