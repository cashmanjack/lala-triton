"""Round-3 persistent worker (runs in either env). Reads JSON commands on stdin, answers one '@@{json}' line per command
on stdout. Launchers are created per shape; every timed run uses sustained.make_graph/measure (round-2 Phase D protocol).
argv: gpu_index env_tag"""
import sys, os, json, math, time, traceback
R = '/home/min/a/cashman3/lala-triton-killtest'
gpu, ENV = sys.argv[1], sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.insert(0, R)
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from common import make_inputs
from sustained import make_graph, measure, buffer_sets
WIN = json.load(open(f'{R}/env/phaseD_window.json'))
ONE = torch.tensor(1.0, device='cuda')
S = dict(shape=None)          # per-shape state
_nvm = {}
_DYN = {}


def reply(d): print('@@' + json.dumps(d, default=str), flush=True)


_CUBLAS_VER = []


def cublas_version():
    if not _CUBLAS_VER:
        import ctypes
        torch.mm(torch.ones(16, 16, device='cuda', dtype=torch.bfloat16), torch.ones(16, 16, device='cuda', dtype=torch.bfloat16))
        libs = sorted({l.split()[-1] for l in open('/proc/self/maps').read().splitlines() if 'libcublas.so' in l})
        try:
            lib, v, out = ctypes.CDLL(libs[0]), ctypes.c_int(), []
            for i in range(3): lib.cublasGetProperty(i, ctypes.byref(v)); out.append(v.value)
            _CUBLAS_VER.append('.'.join(map(str, out)) + ' ' + libs[0])
        except Exception as e:
            _CUBLAS_VER.append('unknown ' + repr(e)[:80])
    return _CUBLAS_VER[0]


def flags():
    m = torch.backends.cuda.matmul
    return dict(allow_tf32=m.allow_tf32, rpr=m.allow_bf16_reduced_precision_reduction,
                blas=str(torch.backends.cuda.preferred_blas_library()), torch=torch.__version__, cuda=torch.version.cuda,
                cublas=cublas_version())


class Ctx:
    """Applies a launcher's global torch flags while it is called / captured."""
    def __init__(self, rpr=None, blas=None): self.rpr, self.blas = rpr, blas
    def __enter__(self):
        m = torch.backends.cuda.matmul
        self.old = (m.allow_bf16_reduced_precision_reduction, torch.backends.cuda.preferred_blas_library())
        if self.rpr is not None: m.allow_bf16_reduced_precision_reduction = self.rpr
        if self.blas is not None: torch.backends.cuda.preferred_blas_library(self.blas)
    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.old[0]
        torch.backends.cuda.preferred_blas_library(self.old[1])


def setup(shape):
    if S['shape'] == shape: return S
    S.clear(); _nvm.clear(); torch.cuda.empty_cache()
    dims, dt, lay = shape.split('_'); M, N, K = map(int, dims.split('x'))
    S.update(shape=shape, M=M, N=N, K=K, dt=dt, lay=lay, sets=buffer_sets(M, N, K, dt, lay, make_inputs), ref=None, fns={})
    a, b, _ = S['sets'][0]
    S['input_sig'] = [float(a.float().double().sum()), float(b.float().double().sum()), float(a.float().abs().max()), float(b.float().abs().max())]
    return S


# ---------------------------------------------------------------- launchers
def base_call(a, b, c):
    if S['dt'] == 'fp8':
        return torch._scaled_mm(a, b, scale_a=ONE, scale_b=ONE, out_dtype=torch.float16)
    return torch.mm(a, b)


def nvmath_fns():
    """cuBLASLt via nvmath: heuristic top-8 algorithms, best by warm timing; one planned Matmul per buffer set."""
    import nvmath
    from nvmath.linalg.advanced import Matmul, MatmulPlanPreferences
    opts, kw = {}, {}
    if S['dt'] == 'fp8':
        kw['quantization_scales'] = {'a': 1.0, 'b': 1.0}
        opts = {'result_type': nvmath.CudaDataType.CUDA_R_16F}
    mms, algs = [], []
    for a, b, c in S['sets']:
        mm = Matmul(a, b, options=opts, stream=torch.cuda.current_stream(), **kw)
        algs.append(mm.plan(preferences=MatmulPlanPreferences(limit=8), stream=torch.cuda.current_stream()))
        mms.append(mm)
    n = min(len(x) for x in algs)
    # pick the algorithm by graph-replay timing over all buffer sets (DRAM-cold rotation, no Python launch overhead)
    ts = []
    st = torch.cuda.Stream()
    for j in range(n):
        try:
            with torch.cuda.stream(st):
                for m, al in zip(mms, algs): m.execute(algorithm=al[j], stream=st)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=st):
                    for _ in range(max(1, 24 // len(mms))):
                        for m, al in zip(mms, algs): m.execute(algorithm=al[j], stream=st)
            reps = []
            for _ in range(5):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); reps.append(e0.elapsed_time(e1))
            ts.append((sorted(reps)[2] / (max(1, 24 // len(mms)) * len(mms)), j)); del g
        except Exception as e:
            ts.append((float('inf'), j))
    best = min(ts)[1]
    _nvm['info'] = dict(n_algos=n, algo_ms=[round(t, 5) for t, _ in ts], best_idx=best)
    return [lambda m=m, al=al[best]: m.execute(algorithm=al, stream=torch.cuda.current_stream()) for m, al in zip(mms, algs)]


def launcher(lid):
    """Returns (list of zero-arg callables over buffer sets, Ctx, output getter for set 0)."""
    if lid in S['fns']: return S['fns'][lid]
    sets = S['sets']
    if lid in ('cublas', 'cublas_rpr0', 'cublaslt'):
        ctx = Ctx(rpr=False if lid == 'cublas_rpr0' else None, blas='cublaslt' if lid == 'cublaslt' else None)
        fns = [lambda a=a, b=b, c=c: base_call(a, b, c) for a, b, c in sets]
        out = lambda: fns[0]()
    elif lid == 'nvmath':
        ctx = Ctx(); fns = nvmath_fns(); out = lambda: fns[0]()
    elif lid.startswith('ours:'):                      # round-2 tutorial kernel
        from kernels.matmul_grouped import matmul
        cf = cfg_from_key(lid[5:]); ctx = Ctx()
        fns = [lambda a=a, b=b, c=c: matmul(a, b, c, cf['BM'], cf['BN'], cf['BK'], cf['G'], cf['num_warps'], cf['num_stages']) for a, b, c in sets]
        out = lambda: (fns[0](), sets[0][2])[1]
    elif lid.startswith('tmpl:'):                      # Inductor template transcription (harness)
        from kernels.inductor_tmpl import launch
        cf = cfg_from_key(lid[5:]); ctx = Ctx()
        fns = [lambda a=a, b=b, c=c: launch(a, b, c, cf, S['dt']) for a, b, c in sets]
        out = lambda: (fns[0](), sets[0][2])[1]
    elif lid.startswith('tmplD:'):                     # dynamic-M template transcription (Phase 2d)
        from kernels.inductor_tmpl import launch
        cf = cfg_from_key(lid[6:]); ctx = Ctx()
        fns = [lambda a=a, b=b, c=c: launch(a, b, c, cf, S['dt'], dynamic_m=True) for a, b, c in sets]
        out = lambda: (fns[0](), sets[0][2])[1]
    elif lid == 'ind_dyn':                             # Inductor dynamic-shape mode, first call at M=512 (tuning hint)
        import torch._inductor.config as ic
        ic.max_autotune_gemm_backends = 'ATEN,TRITON'
        dk = (S['dt'], S['N'], S['K'])
        if dk not in _DYN:
            f2 = torch.compile(lambda x, y: base_call(x, y, None), mode='max-autotune-no-cudagraphs', dynamic=True)
            a5, b5, _ = make_inputs(512, S['N'], S['K'], S['dt'], layout=S['lay'])
            torch._dynamo.mark_dynamic(a5, 0); f2(a5, b5); torch.cuda.synchronize(); del a5, b5
            _DYN[dk] = f2
        f2 = _DYN[dk]
        for a, b, c in sets: torch._dynamo.mark_dynamic(a, 0)
        ctx = Ctx(); fns = [lambda a=a, b=b, f2=f2: f2(a, b) for a, b, c in sets]; out = lambda: fns[0]()
    elif lid == 'ind_stock':                           # stock Inductor max-autotune (cache dir set by driver env)
        import torch._inductor.config as ic
        ic.max_autotune_gemm_backends = 'ATEN,TRITON'
        f2 = torch.compile(lambda x, y: base_call(x, y, None), mode='max-autotune-no-cudagraphs')
        ctx = Ctx(); fns = [lambda a=a, b=b, f2=f2: f2(a, b) for a, b, c in sets]; out = lambda: fns[0]()
    else:
        raise ValueError(lid)
    S['fns'][lid] = (fns, ctx, out)
    return S['fns'][lid]


def cfg_from_key(k):
    t, g, w, s = k.split('_'); bm, bn, bk = map(int, t.split('x'))
    return dict(BM=bm, BN=bn, BK=bk, G=int(g[1:]), num_warps=int(w[1:]), num_stages=int(s[1:]))


# ---------------------------------------------------------------- commands
def correctness(lid):
    fns, ctx, out = launcher(lid)
    a, b, c = S['sets'][0]
    if S['ref'] is None:
        S['ref'] = a.double() @ b.double()
    ref = S['ref']
    with ctx:
        y = out(); fl = flags()
    torch.cuda.synchronize()
    err = (y.double() - ref).abs()
    ma, mb, K = float(a.float().abs().max()), float(b.float().abs().max()), S['K']
    u = 2.0 ** -8 if y.dtype == torch.bfloat16 else 2.0 ** -11
    env = u * ref.abs() + K * 2.0 ** -24 * ma * mb
    R = float((err / env).max())
    res = dict(out_dtype=str(y.dtype), maxabs=float(err.max()), e_max=float(err.max()) / (ma * mb * math.sqrt(K)),
               e_F=float(torch.linalg.norm(y.double() - ref) / torch.linalg.norm(ref)), R=R, pass_=int(R <= 2),
               refmax=float(ref.abs().max()), **{f'flag_{k}': v for k, v in fl.items()})
    # where is the worst element relative to its envelope, and its ulp
    i = int((err / env).argmax()); rv = float(ref.flatten()[i]); res['worst_ref'] = rv; res['worst_err'] = float(err.flatten()[i])
    del err, env
    # kernel names from one eager call
    try:
        from torch.profiler import profile, ProfilerActivity
        with ctx:
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                fns[0](); torch.cuda.synchronize()
        names = sorted({e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA and 'memset' not in e.name.lower() and 'memcpy' not in e.name.lower()})
        res['kernels'] = '|'.join(names)[:600]
    except Exception as e:
        res['kernels'] = 'ERR ' + repr(e)[:200]
    if lid == 'nvmath': res['nvmath'] = _nvm.get('info')
    return res


def timed(lid, rep):
    if S['ref'] is not None: S['ref'] = None; torch.cuda.empty_cache()
    fns, ctx, _ = launcher(lid)
    with ctx:
        g, L, est = make_graph(fns)
        fl = flags()
    res, _ = measure(g, L, int(gpu), warmup_s=WIN['warmup_s'], window_s=WIN['window_s'], cooldown_s=WIN['cooldown_s'])
    del g; torch.cuda.empty_cache()
    res.update(L=L, est_ms=est, nsets=len(S['sets']), **{f'flag_{k}': v for k, v in fl.items()})
    return res


for line in sys.stdin:
    line = line.strip()
    if not line: continue
    msg = json.loads(line)
    try:
        if msg['cmd'] == 'bye': reply(dict(ok=1)); break
        setup(msg['shape'])
        if msg['cmd'] == 'setup': reply(dict(ok=1, input_sig=S['input_sig'], nsets=len(S['sets']), env=ENV, **flags()))
        elif msg['cmd'] == 'correct': reply(dict(ok=1, **correctness(msg['launcher'])))
        elif msg['cmd'] == 'time': reply(dict(ok=1, **timed(msg['launcher'], msg['rep'])))
        elif msg['cmd'] == 'drop':
            S['fns'].pop(msg['launcher'], None); reply(dict(ok=1))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        reply(dict(ok=0, error=repr(e)[:500]))
