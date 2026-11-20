import os, sys, tempfile, json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'; os.environ['TRITON_CACHE_DIR'] = tempfile.mkdtemp(dir='/var/tmp/cashman3_r3')
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import torch, triton, triton.language as tl
from triton.testing import do_bench
from common import make_inputs
from kernels.inductor_tmpl import launch
from r3_worker_cfg import cfg_from_key
mode, shape, ck = sys.argv[1:4]
if mode == 'warm':
    @triton.jit
    def _dummy(x): tl.store(x, 1.0)
    _dummy[(1,)](torch.empty(1, device='cuda'))
dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
a, b, c = make_inputs(M, N, K, dt, layout=lay); cf = cfg_from_key(ck)
k = launch(a, b, c, cf, dt)
ms = sorted(do_bench(lambda: launch(a, b, c, cf, dt), warmup=10, rep=50, return_mode='median') for _ in range(5))[2]
print(json.dumps(dict(mode=mode, shape=shape, cfg=ck, n_regs=k.n_regs, spills=k.n_spills, ms=ms)))
