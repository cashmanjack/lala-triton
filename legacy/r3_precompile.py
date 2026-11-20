"""Compile-only warmup of harness template kernels for one shape (static constexpr M, N, K). argv: shape cfgs_json [dynm]"""
import sys, json
sys.path.insert(0, '/home/min/a/cashman3/lala-triton-killtest')
import torch, triton, triton.language as tl
from kernels.inductor_tmpl import mm_tmpl, smm_tmpl, mm_tmpl_dynm, smm_tmpl_dynm, INDUCTOR_OPTS, ensure_short_ptr
shape, cfgs = sys.argv[1], json.loads(sys.argv[2])
dynm = len(sys.argv) > 3 and sys.argv[3] == 'dynm'
dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
tdt = torch.float8_e4m3fn if dt == 'fp8' else torch.bfloat16
a = torch.empty(64, device='cuda', dtype=tdt); b = torch.empty(64, device='cuda', dtype=tdt)
c = torch.empty(64, device='cuda', dtype=torch.float16 if dt == 'fp8' else torch.bfloat16)
s = torch.empty(1, device='cuda', dtype=torch.float32)
ensure_short_ptr()
sam, sak = K, 1
sbk, sbn = (1, K) if (lay == 'ABT' or dt == 'fp8') else (N, 1)
for cf in cfgs:
    kw = dict(stride_am=sam, stride_ak=sak, stride_bk=sbk, stride_bn=sbn, GROUP_M=cf['G'], EVEN_K=K % cf['BK'] == 0, ACC_TYPE=tl.float32,
              BLOCK_M=cf['BM'], BLOCK_N=cf['BN'], BLOCK_K=cf['BK'], num_warps=cf['num_warps'], num_stages=cf['num_stages'], grid=(1,), **INDUCTOR_OPTS)
    try:
        Ms = [M] if not dynm else cf.get('Ms', [M])
        for m in Ms:
            if dt == 'fp8':
                (smm_tmpl_dynm if dynm else smm_tmpl).warmup(a, b, s, s, c, m, N, K, USE_FAST_ACCUM=False, SCALING_ROWWISE=False, **kw)
            else:
                (mm_tmpl_dynm if dynm else mm_tmpl).warmup(a, b, c, m, N, K, ALLOW_TF32=False, **kw)
    except Exception as e:
        print('fail', cf, repr(e)[:120], file=sys.stderr)
