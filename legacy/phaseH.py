"""Phase H (descriptive): eviction hints on L2-cliff configs. argv: mode (ncu_target|time) [M N K]"""
import sys, os, json, csv, random
R = '/home/min/a/cashman3/lala-triton-killtest'
sys.path.insert(0, R)
mode = sys.argv[1]
import torch, pandas as pd
from common import make_inputs
from kernels.matmul_evict import matmul_ev
cl = pd.read_csv(f'{R}/predictions/phaseH_cliff.csv')
VAR = {'none': ('', ''), 'Alast': ('evict_last', 'evict_first'), 'Blast': ('evict_first', 'evict_last')}
def parse(t):
    a, w, s = t.split('_'); BM, BN, BK = map(int, a.split('x')); return BM, BN, BK, int(w[1:]), int(s[1:])
if mode == 'ncu_target':
    M, N, K = map(int, sys.argv[2:5]); x = cl[(cl.M == M) & (cl.N == N) & (cl.K == K)]
    a, b, c = make_inputs(M, N, K, 'bf16'); man = []
    for r in x.itertuples():
        BM, BN, BK, w, s = parse(r.tile)
        for v, (ea, eb) in VAR.items():
            matmul_ev(a, b, c, BM, BN, BK, int(r.G), w, s, ea, eb); torch.cuda.synchronize(); man.append([r.tile, int(r.G), v])
    json.dump(man, open(f'{R}/runs/ncu_phaseH/{M}_{N}_{K}.manifest.json', 'w'))
elif mode == 'time':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from sustained import make_graph, measure, buffer_sets
    WIN = json.load(open(f'{R}/env/phaseD_window.json'))
    OUT = f'{R}/runs/phaseH_time.csv'
    done = {(r['M'], r['N'], r['K'], r['tile'], r['G'], r['variant'], r['rep']) for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else set()
    f = open(OUT, 'a', newline=''); wr = None
    for (M, N, K), x in cl.groupby(['M', 'N', 'K']):
        sets = buffer_sets(M, N, K, 'bf16', 'AB', make_inputs)
        for rep in range(3):
            items = [(r.tile, int(r.G), v) for r in x.itertuples() for v in VAR]; random.Random(f'H-{M}-{N}-{K}-{rep}').shuffle(items)
            for tile, G, v in items:
                if (str(M), str(N), str(K), tile, str(G), v, str(rep)) in done: continue
                BM, BN, BK, w, s = parse(tile); ea, eb = VAR[v]
                g, L, _ = make_graph([lambda a=a, b=b, c=c: matmul_ev(a, b, c, BM, BN, BK, G, w, s, ea, eb) for a, b, c in sets])
                res, _ = measure(g, L, 0, warmup_s=WIN['warmup_s'], window_s=WIN['window_s'], cooldown_s=WIN['cooldown_s'])
                row = dict(M=M, N=N, K=K, tile=tile, G=G, variant=v, rep=rep, **res)
                if wr is None:
                    wr = csv.DictWriter(f, fieldnames=list(row))
                    if not done: wr.writeheader()
                wr.writerow(row); f.flush(); del g
        del sets; torch.cuda.empty_cache()
    print('PHASE H TIME DONE')
