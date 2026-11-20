import os, sys, glob, tempfile
d = tempfile.mkdtemp(prefix='ind_dump_', dir='/var/tmp/cashman3_r3'); os.environ['TORCHINDUCTOR_CACHE_DIR'] = d; os.environ['TRITON_CACHE_DIR'] = d + '/tr'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import torch, inductor_patch
import torch._inductor.config as ic
ic.max_autotune_gemm_backends = 'TRITON'
from common import make_inputs
inductor_patch.set_configs([(64, 128, 32, 3, 4, 4)])
for dt, lay in [('bf16', 'AB'), ('bf16', 'ABT'), ('fp8', 'ABT')]:
    torch._dynamo.reset()
    a, b, c = make_inputs(512, 4096, 4096, dt, layout=lay); one = torch.tensor(1.0, device='cuda')
    f = (lambda x, y: torch._scaled_mm(x, y, scale_a=one, scale_b=one, out_dtype=torch.float16)) if dt == 'fp8' else (lambda x, y: x @ y)
    y = torch.compile(f, mode='max-autotune-no-cudagraphs')(a, b); torch.cuda.synchronize()
    ref = (a.float() @ b.float()); print(dt, lay, (y.float() - ref).abs().max().item(), y.dtype)
print(d)
for p in glob.glob(d + '/**/*.py', recursive=True):
    s = open(p).read()
    if 'triton_tem' in s and '@triton.jit' in s and 'def triton_' in s and 'GROUP_M' in s: print('=====', p); print(s[:5000])
