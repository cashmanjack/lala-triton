"""Executed under ncu: launch the listed subset rows (harness template kernels) for one shape, in order."""
import sys, json
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import torch, pandas as pd
from common import make_inputs
from kernels.inductor_tmpl import launch
shape, idxs, man = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
sub = pd.read_csv('/home/min/a/cashman3/lala-triton-killtest/predictions/r3_ncu_subset.csv').set_index('idx')
dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
a, b, c = make_inputs(M, N, K, dt, layout=lay)
done = []
for i in idxs:
    r = sub.loc[i]; cf = dict(BM=int(r.BM), BN=int(r.BN), BK=int(r.BK), G=int(r.G), num_warps=int(r.num_warps), num_stages=int(r.num_stages))
    launch(a, b, c, cf, dt); torch.cuda.synchronize(); done.append(i)
json.dump(done, open(man, 'w'))
