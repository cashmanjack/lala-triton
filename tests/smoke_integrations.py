"""GPU smoke test: compile + run one GEMM through each integration.  python3 tests/smoke_integrations.py [profile]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, triton, triton.language as tl
prof = sys.argv[1] if len(sys.argv) > 1 else 'profiles/rtx6000ada.json'
from lala_triton.select import select
from lala_triton import inductor
from lala_triton.triton_prune import prune_configs_by
from lala_triton.kernels import _mm
print('select:', select(128, 4096, 4096, 'bf16', 'ABT', prof, k=2))
a = torch.randn(128, 4096, device='cuda').bfloat16(); b = torch.randn(4096, 4096, device='cuda').bfloat16().t()
ref = a.float() @ b.float()
inductor.enable(prof, k=2, bf16_layout='ABT')
import torch._inductor.config as ic; ic.max_autotune_gemm_backends = 'TRITON'
y = torch.compile(lambda x, w: torch.mm(x, w), mode='max-autotune-no-cudagraphs')(a, b); inductor.disable()
print('inductor ok, err', (y.float() - ref).abs().max().item(), 'picked', inductor.STATE['log'][-1][4])
cfgs = [triton.Config(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=64, GROUP_M=8), num_warps=4, num_stages=3) for bm in (16, 64, 128) for bn in (32, 128, 256)]
tuned = triton.autotune(configs=cfgs, key=['M', 'N', 'K'], prune_configs_by=prune_configs_by(prof, 'bf16', 'ABT', k=2))(_mm)
c = torch.empty(128, 4096, device='cuda', dtype=torch.bfloat16)
grid = lambda m: (triton.cdiv(128, m['BLOCK_M']) * triton.cdiv(4096, m['BLOCK_N']),)
tuned[grid](a, b, c, 128, 4096, 4096, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), EVEN_K=True)
torch.cuda.synchronize()
print('triton_prune ok, err', (c.float() - ref).abs().max().item(), 'kept', len(tuned.configs), '-> best', tuned.best_config)
