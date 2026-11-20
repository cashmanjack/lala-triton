import os, sys, tempfile
os.environ['CUDA_VISIBLE_DEVICES'] = '1'; os.environ['TRITON_CACHE_DIR'] = tempfile.mkdtemp(dir='/var/tmp/cashman3_r3')
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import torch, triton, triton.language as tl
from common import make_inputs
from kernels.inductor_tmpl import launch
if sys.argv[1] == 'warm':
    @triton.jit
    def _dummy(x): tl.store(x, 1.0)
    _dummy[(1,)](torch.empty(1, device='cuda'))
a, b, c = make_inputs(8192, 4096, 14336, 'bf16', layout='AB')
k = launch(a, b, c, dict(BM=64, BN=128, BK=32, G=16, num_warps=4, num_stages=3), 'bf16')
print(sys.argv[1], 'n_regs', k.n_regs, 'short_ptr_in_llir', 'p3:32:32' in k.asm['llir'])
