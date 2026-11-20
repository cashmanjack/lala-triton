import os, time, torch, tempfile
os.environ['TORCHINDUCTOR_CACHE_DIR'] = tempfile.mkdtemp(prefix='ind_')
import torch._inductor.config as ic
print('backends default:', ic.max_autotune_gemm_backends)
ic.max_autotune_gemm_backends = 'ATEN,TRITON'
a = torch.randn(512, 4096, device='cuda', dtype=torch.bfloat16); w = torch.randn(14336, 4096, device='cuda', dtype=torch.bfloat16)
f = torch.compile(lambda a, b: a @ b, mode='max-autotune-no-cudagraphs')
t = time.time(); y = f(a, w.t()); torch.cuda.synchronize(); print('bf16 ABT first call', time.time() - t)
# fp8
a8 = a.to(torch.float8_e4m3fn); w8 = w.to(torch.float8_e4m3fn); one = torch.tensor(1.0, device='cuda')
try:
    y = torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16); print('eager _scaled_mm OK', y.shape)
except Exception as e: print('eager _scaled_mm FAIL', repr(e)[:200])
try:
    g = torch.compile(lambda a, b: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float16), mode='max-autotune-no-cudagraphs')
    t = time.time(); y = g(a8, w8.t()); torch.cuda.synchronize(); print('compiled _scaled_mm OK', time.time() - t)
except Exception as e: print('compiled _scaled_mm FAIL', repr(e)[:300])
