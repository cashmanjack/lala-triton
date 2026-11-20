"""Our Triton kernel: transcription of Inductor's mm / scaled_mm template (grouped tile order, EVEN_K, fp32 acc).
Used for calibration and the oracle harness (PREREGISTRATION R5-K). Layout: 'AB' = B is [K,N] row-major; 'ABT' = B stored [N,K] contiguous."""
import torch, triton, triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M); num_n = tl.cdiv(N, BLOCK_N)
    npg = GROUP_M * num_n
    gid = pid // npg
    first_m = gid * GROUP_M
    gsz = min(num_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % npg) % gsz)
    pid_n = (pid % npg) // gsz
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs); b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=rk[None, :] < K - k * BLOCK_K, other=0.)
            b = tl.load(b_ptrs, mask=rk[:, None] < K - k * BLOCK_K, other=0.)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * sak; b_ptrs += BLOCK_K * sbk
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc.to(c_ptr.dtype.element_ty), mask=(om[:, None] < M) & (on[None, :] < N))

def make_inputs(M, N, K, dtype, layout='ABT', seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    a = torch.randn(M, K, device='cuda', generator=g)
    b = torch.randn((N, K) if layout == 'ABT' else (K, N), device='cuda', generator=g)
    if dtype == 'bf16': a, b = a.bfloat16(), b.bfloat16(); out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    else: a, b = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn); out = torch.empty(M, N, device='cuda', dtype=torch.float16)
    if layout == 'ABT': b = b.t()          # [K,N] view with strides (1,K): K-major
    return a, b, out

def launch(a, b, c, cfg):
    M, K = a.shape; N = b.shape[1]
    bm, bn, bk = cfg['BM'], cfg['BN'], cfg['BK']
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    return _mm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                     BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=cfg['G'], EVEN_K=(K % bk == 0),
                     num_warps=cfg['num_warps'], num_stages=cfg['num_stages'])
