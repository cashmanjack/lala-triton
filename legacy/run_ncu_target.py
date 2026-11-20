"""Executed under ncu. argv: pred_csv sweep M N K manifest_out. Launches each (tile,G) of that shape in order."""
import sys, os, json
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import torch, pandas as pd
from common import make_inputs
from kernels.matmul_grouped import matmul
pred, sweep, M, N, K, man = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6]), sys.argv[6]
dtype = sys.argv[7] if len(sys.argv) > 7 else 'bf16'
df = pd.read_csv(pred)
df = df[(df.sweep == sweep) & (df.M == M) & (df.N == N) & (df.K == K)]
if 'dtype' in df: df = df[df.dtype == dtype]
a, b, c = make_inputs(M, N, K, dtype)
launched = []
for _, r in df.iterrows():
    try:
        matmul(a, b, c, int(r.BM), int(r.BN), int(r.BK), int(r.G), int(r.num_warps), int(r.num_stages))
        torch.cuda.synchronize(); launched.append(int(r.name))
    except Exception as e:
        print('FAIL', r.to_dict(), e, file=sys.stderr)
json.dump(launched, open(man, 'w'))
