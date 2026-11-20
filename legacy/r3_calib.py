"""Round-3 Step 2b calibration (GPU1). (1) compiled metadata (n_regs, spills, shared, occ) of every template tile per
family at 4096^3; (2) eff per (family, stock tile) at G=8 on the 4 round-1 calibration shapes, do_bench(10, 50, median).
Resumable. Writes env/r3_tmpl_meta.csv, runs/r3_eff_calib.csv, env/r3_eff_tmpl.csv."""
import os, sys, csv, time, random
R = '/home/min/a/cashman3/lala-triton-killtest'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TRITON_CACHE_DIR'] = '/var/tmp/cashman3_r3/tc_calib'
sys.path.insert(0, R)
import torch, pandas as pd
from triton.testing import do_bench
from common import make_inputs, occupancy
from kernels.inductor_tmpl import launch, stock_list, MM_STOCK, SMM_STOCK
from phaseD_shapes import SHAPES
from model.time_model import ideal_compute
FAM = [('bf16', 'AB'), ('bf16', 'ABT'), ('fp8', 'ABT')]
EM = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 72, 576, 4608]
ENK = [(4096, 4096), (14336, 4096), (4096, 14336)]
CAL = [(8192, 8192, 8192), (16384, 8192, 8192), (8192, 16384, 8192), (12288, 12288, 12288)]


def tkey(bm, bn, bk, s, w): return f'{bm}x{bn}x{bk}_w{w}_s{s}'


# ---- (1) metadata for the union of tiles that occur on any D or E shape
META = f'{R}/env/r3_tmpl_meta.csv'
tiles = {f: {} for f in FAM}
for (M, N, K, dt, lay) in SHAPES:
    for t in stock_list(M, N, K, dt): tiles[(dt, lay)][t[:5]] = t[5]
for dt in ('bf16', 'fp8'):
    for (N, K) in ENK:
        for M in EM:
            for t in stock_list(M, N, K, dt): tiles[(dt, 'ABT')][t[:5]] = t[5]
done = {(r['dtype'], r['layout'], r['tile']) for r in csv.DictReader(open(META))} if os.path.exists(META) else set()
f = open(META, 'a', newline=''); w = csv.writer(f)
if not done: w.writerow(['dtype', 'layout', 'tile', 'BM', 'BN', 'BK', 'num_stages', 'num_warps', 'parent', 'ok', 'n_regs', 'n_spills', 'shared', 'occ'])
for (dt, lay), ts in tiles.items():
    a, b, c = make_inputs(4096, 4096, 4096, dt, layout=lay)
    for (bm, bn, bk, s, nw), pi in sorted(ts.items()):
        if (dt, lay, tkey(bm, bn, bk, s, nw)) in done: continue
        try:
            k = launch(a, b, c, dict(BM=bm, BN=bn, BK=bk, G=8, num_warps=nw, num_stages=s), dt); torch.cuda.synchronize()
            occ = occupancy(k.metadata.shared, k.n_regs, nw)
            w.writerow([dt, lay, tkey(bm, bn, bk, s, nw), bm, bn, bk, s, nw, pi, 1, k.n_regs, k.n_spills, k.metadata.shared, occ])
        except Exception as e:
            w.writerow([dt, lay, tkey(bm, bn, bk, s, nw), bm, bn, bk, s, nw, pi, 0, '', '', '', 0]); print('compile fail', dt, lay, bm, bn, bk, s, nw, repr(e)[:200])
        f.flush()
f.close(); print('meta done', flush=True)
meta = pd.read_csv(META).set_index(['dtype', 'layout', 'tile'])

# ---- (2) eff calibration on stock tiles at G=8
CALF = f'{R}/runs/r3_eff_calib.csv'
done = {(r['dtype'], r['layout'], r['tile'], r['shape']) for r in csv.DictReader(open(CALF))} if os.path.exists(CALF) else set()
f = open(CALF, 'a', newline=''); w = csv.writer(f)
if not done: w.writerow(['dtype', 'layout', 'tile', 'shape', 'M', 'N', 'K', 'BM', 'BN', 'occ', 'ms', 'ok', 'maxerr', 'ts'])
for (dt, lay) in FAM:
    stock = SMM_STOCK if dt == 'fp8' else MM_STOCK
    for (M, N, K) in CAL:
        a, b, c = make_inputs(M, N, K, dt, layout=lay)
        ref = (a.float() @ b.float()); rmax = ref.abs().max().item()
        order = list(stock); random.Random(f'r3cal-{dt}-{lay}-{M}{N}{K}').shuffle(order)
        for (bm, bn, bk, s, nw) in order:
            tk = tkey(bm, bn, bk, s, nw); sk = f'{M}x{N}x{K}'
            if (dt, lay, tk, sk) in done: continue
            cf = dict(BM=bm, BN=bn, BK=bk, G=8, num_warps=nw, num_stages=s)
            occ = int(meta.loc[(dt, lay, tk), 'occ'])
            try:
                c.zero_(); launch(a, b, c, cf, dt); torch.cuda.synchronize()
                err = (c.float() - ref).abs().max().item()
                ms = do_bench(lambda: launch(a, b, c, cf, dt), warmup=10, rep=50, return_mode='median'); ok = 1
            except Exception as e:
                ms, ok, err = float('nan'), 0, float('nan')
            w.writerow([dt, lay, tk, sk, M, N, K, bm, bn, occ, ms, ok, err, time.time()]); f.flush()
        del a, b, c, ref; torch.cuda.empty_cache()
        print('cal', dt, lay, M, N, K, flush=True)
f.close()
cal = pd.read_csv(CALF); cal = cal[cal.ok == 1].copy()
cal['ideal'] = [ideal_compute(r.M, r.N, r.K, r.BM, r.BN, int(r.occ), r.dtype) for r in cal.itertuples()]
cal['eff'] = cal.ideal / (cal.ms * 1e-3)
eff = cal.groupby(['dtype', 'layout', 'tile']).eff.median()
eff.to_csv(f'{R}/env/r3_eff_tmpl.csv')
print(eff.groupby(['dtype', 'layout']).describe().round(3).to_string())
print('CALIB DONE', flush=True)
