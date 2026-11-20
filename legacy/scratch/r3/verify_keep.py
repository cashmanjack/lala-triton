"""2a verification, part (i): compile one (shape, config) via single-config patched Inductor and via the harness, in
fresh Triton caches; compare cubin resource usage and normalized SASS. argv: shape_key cfg_key outdir. GPU1."""
import os, sys, glob, json, re, subprocess, tempfile
R = '/home/min/a/cashman3/lala-triton-killtest'
shape, ck, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
base = tempfile.mkdtemp(prefix='vc_', dir='/var/tmp/cashman3_r3')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = f'{base}/ind'; os.environ['TRITON_CACHE_DIR'] = f'{base}/tri_ind'
sys.path.insert(0, R)
import torch, inductor_patch
import torch._inductor.config as ic
ic.max_autotune_gemm_backends = 'TRITON'
from common import make_inputs
from r3_worker_cfg import cfg_from_key
dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
cf = cfg_from_key(ck)
a, b, c = make_inputs(M, N, K, dt, layout=lay); one = torch.tensor(1.0, device='cuda')
inductor_patch.set_configs([(cf['BM'], cf['BN'], cf['BK'], cf['num_stages'], cf['num_warps'], cf['G'])])
f = (lambda x, y: torch._scaled_mm(x, y, scale_a=one, scale_b=one, out_dtype=torch.float16)) if dt == 'fp8' else (lambda x, y: x @ y)
y1 = torch.compile(f, mode='max-autotune-no-cudagraphs')(a, b); torch.cuda.synchronize()
# harness in its own cache
import triton
os.environ['TRITON_CACHE_DIR'] = f'{base}/tri_h'
from kernels.inductor_tmpl import launch
k = launch(a, b, c, cf, dt); torch.cuda.synchronize()
def info(cubin):
    ru = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '-res-usage', cubin], capture_output=True, text=True).stdout
    reg = int(re.search(r'REG:(\d+)', ru).group(1)); shm = int(re.search(r'SHARED:(\d+)', ru).group(1))
    sass = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '-sass', cubin], capture_output=True, text=True).stdout
    ins = [re.sub(r'/\*[0-9a-f]+\*/', '', l).split(';')[0].strip() for l in sass.splitlines() if re.match(r'\s+/\*[0-9a-f]{4}\*/', l)]
    ins = [re.sub(r'c\[0x0\]\[0x[0-9a-f]+\]', 'c[]', i) for i in ins]   # param offsets differ (arg count)
    return reg, shm, ins
ci = [p for p in glob.glob(f'{base}/tri_ind/**/*.cubin', recursive=True) if 'triton_tem' in os.path.basename(p)]
ch = [p for p in glob.glob(f'{base}/tri_h/**/*.cubin', recursive=True) if 'tmpl' in os.path.basename(p)]
ri, rh = info(ci[0]), info(ch[0])
import difflib
nd = sum(1 for l in difflib.ndiff(ri[2], rh[2]) if l[0] in '+-')
err = (y1.float() - c.float()).abs().max().item()
res = dict(shape=shape, cfg=ck, ind_reg=ri[0], ind_shared=ri[1], h_reg=rh[0], h_shared=rh[1], h_nregs_meta=k.n_regs, h_spills=k.n_spills,
           ind_n_ins=len(ri[2]), h_n_ins=len(rh[2]), sass_diff_lines=nd, out_maxdiff=err, n_ind_cubins=len(ci))
print(json.dumps(res)); open(outdir, 'a').write(json.dumps(res) + '\n')
print(base)
