"""Double-read probe, run under ncu (or standalone). args: variant(seq|strided) size_MiB [size_MiB ...]. One launch per size.
Each of n_sm CTAs reads its own chunk (S/n_sm) twice in 4 KiB blocks; strided = prime-stride over 128 B segments."""
import sys, torch, triton, triton.language as tl

@triton.jit
def _probe(x_ptr, out_ptr, nseg_chunk, STRIDE: tl.constexpr, NREP: tl.constexpr = 2):
    pid = tl.program_id(0)
    base = pid * nseg_chunk                       # in 128 B segments (32 fp32)
    cols = tl.arange(0, 32)
    rows = tl.arange(0, 32)
    acc = tl.zeros((32, 32), dtype=tl.float32)
    nblk = nseg_chunk // 32
    for rep in range(NREP):
        for b in range(nblk):
            seg = b * 32 + rows
            if STRIDE > 1: seg = (seg * STRIDE) % (nblk * 32)      # STRIDE prime, coprime with nblk*32 (a power-of-two multiple)
            acc += tl.load(x_ptr + (base + seg)[:, None] * 32 + cols[None, :])
    tl.store(out_ptr + pid, tl.sum(acc))

def seg_chunk(size_mib, n_sm): return (size_mib * 2**20 // 4 // 32 // n_sm) // 32 * 32       # segments per CTA, multiple of 32
def seg_elems(size_mib, n_sm): return seg_chunk(size_mib, n_sm) * 32 * n_sm

def run(variant, size_mib, n_sm):
    x = torch.zeros(seg_elems(size_mib, n_sm), device='cuda', dtype=torch.float32)
    seg_chunk_ = seg_chunk(size_mib, n_sm)
    out = torch.empty(n_sm, device='cuda')
    _probe[(n_sm,)](x, out, seg_chunk_, STRIDE=1 if variant == 'seq' else 1031, num_warps=8)
    torch.cuda.synchronize(); return x.numel() * 4

if __name__ == '__main__':
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    for s in sys.argv[2:]: print('SIZE', s, run(sys.argv[1], int(s), n_sm), flush=True)
