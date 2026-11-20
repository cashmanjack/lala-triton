"""Phase B: sustained power/clock coupling runs + Phase C power-calibration runs. GPU0. Resumable."""
import sys, os, csv, json, random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ.setdefault('TRITON_CACHE_DIR', '/home/min/a/cashman3/lala-triton-killtest/.triton_cache')
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import numpy as np, pandas as pd, torch
from common import make_inputs, ROOT
from kernels.matmul_grouped import matmul
from sustained import make_graph, measure, buffer_sets
OUT = f'{ROOT}/runs/phaseB.csv'
cls = pd.read_csv(f'{ROOT}/runs/r2_shape_classes.csv')
sens = cls[cls.sens]; ctrl = cls[~cls.sens].sort_values('Grange').head(5)
def parse_tile(t):
    a, w, s = t.split('_'); BM, BN, BK = map(int, a.split('x')); return BM, BN, BK, int(w[1:]), int(s[1:])
jobs = []   # (phase, rep, shape, M, N, K, dtype, tile, G)
for rep in range(3):
    for r in pd.concat([sens.assign(cls='sens'), ctrl.assign(cls='ctrl')]).itertuples():
        dims, dt = r.shape.split('_'); M, N, K = map(int, dims.split('x'))
        for G in (1, 4, 8, 12, 16, 32): jobs.append(('B_' + r.cls, rep, r.shape, M, N, K, dt, r.tile, G))
comp = pd.read_csv(f'{ROOT}/runs/compile_all.csv'); feas = comp[(comp.ok == 1) & (comp.n_spills == 0)]
for dt in ('bf16', 'fp8'):
    for (M, N, K) in [(8192, 8192, 8192), (16384, 8192, 8192)]:
        for t in ['128x128x64_w4_s3', '128x256x64_w8_s3', '64x64x64_w4_s3', '32x128x64_w4_s3']:
            if dt == 'fp8':
                t128 = t.replace('x64_', 'x128_')
                t = t128 if t128 in set(feas[feas.dtype == 'fp8'].tile) else t
            if t not in set(feas[feas.dtype == dt].tile): continue
            for G in (1, 8, 32): jobs.append(('cal', 0, f'{M}x{N}x{K}_{dt}', M, N, K, dt, t, G))
done = set()
if os.path.exists(OUT):
    done = {(r['phase'], r['rep'], r['shape'], r['tile'], r['G']) for r in csv.DictReader(open(OUT))}
fields = None; f = open(OUT, 'a', newline=''); wr = None
# group by (phase, rep, shape) block; randomize config order within block
blocks = {}
for j in jobs: blocks.setdefault(j[:3], []).append(j)
for key, js in blocks.items():
    todo = [j for j in js if (j[0], str(j[1]), j[2], j[7], str(j[8])) not in done]
    if not todo: continue
    random.Random(f'B-{key}').shuffle(todo)
    _, _, shape, M, N, K, dt, _, _ = todo[0]
    sets = buffer_sets(M, N, K, dt, 'AB', make_inputs)
    for j in todo:
        phase, rep, shape, M, N, K, dt, tile, G = j
        BM, BN, BK, w, s = parse_tile(tile)
        fns = [lambda a=a, b=b, c=c: matmul(a, b, c, BM, BN, BK, G, w, s) for (a, b, c) in sets]
        g, L, est = make_graph(fns)
        res, trace = measure(g, L, 0)
        row = dict(phase=phase, rep=rep, shape=shape, M=M, N=N, K=K, dtype=dt, tile=tile, G=G, nsets=len(sets), L=L, **res)
        np.save(f'{ROOT}/runs/phaseB_traces/{phase}_{rep}_{shape}_{tile}_G{G}.npy', trace)
        if wr is None:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if not done: wr.writeheader()
        wr.writerow(row); f.flush()
        print(phase, rep, shape, tile, G, f"{res['ms_per_gemm']:.4f}ms clk={res['sm_clk_med']:.0f} P={res['power_mean']:.0f}W settle={res['settle_s']:.2f}s", flush=True)
        del g; torch.cuda.empty_cache()
    del sets; torch.cuda.empty_cache()
print('PHASE B DONE')
