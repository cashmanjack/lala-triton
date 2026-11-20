import os, sys, tempfile, torch
os.environ['TORCHINDUCTOR_CACHE_DIR'] = tempfile.mkdtemp(prefix='ind_')
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import torch._inductor.config as ic; ic.max_autotune_gemm_backends = 'ATEN,TRITON'
from sustained import make_graph
a = torch.randn(512, 4096, device='cuda', dtype=torch.bfloat16); w = torch.randn(14336, 4096, device='cuda', dtype=torch.bfloat16).t()
f = torch.compile(lambda a, b: a @ b, mode='max-autotune-no-cudagraphs')
ref = a.float() @ w.float()
y = f(a, w); torch.cuda.synchronize()
outs = []
g, L, est = make_graph([lambda: outs.append(f(a, w))])
g.replay(); torch.cuda.synchronize(); print('graph ok', L, est, (outs[-1].float() - ref).abs().max().item())
