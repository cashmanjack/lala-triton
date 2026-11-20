"""One-call Inductor integration: lala_triton.inductor.enable(profile, k=1) / disable().
Patches mm_configs/mm_options (bf16/fp16 mm, addmm) and scaled_mm_configs/scaled_mm_options (fp8 _scaled_mm) in torch 2.6's
torch._inductor.kernel.{mm,mm_scaled} to yield our top-k configs verbatim (no size-based shrinking; GROUP_M exposed).
Limitation: Inductor's config hook doesn't see B's layout; bf16 uses `bf16_layout` (default 'AB'). fp8 is always 'ABT' (scaled_mm needs column-major B)."""
import torch, triton
import torch._inductor.lowering  # noqa: F401  (import order avoids a circular import in torch 2.6)
import torch._inductor.kernel.mm as _mm
import torch._inductor.kernel.mm_scaled as _ms
from torch._inductor.virtualized import V
from .select import select
from .profile import load as _load

_ORIG = dict(mm_configs=_mm.mm_configs, mm_options=_mm.mm_options, scaled_mm_configs=_ms.scaled_mm_configs, scaled_mm_options=_ms.scaled_mm_options)
STATE = dict(on=False, log=[], select_s=0.0)

def _hint(x):
    return int(V.graph.sizevars.size_hint(x, fallback=torch._inductor.config.unbacked_symint_fallback))

def _gen(prof, dtype, layout, k):
    def configs(m, n, kk, *args, **kw):
        M, N, K = _hint(m), _hint(n), _hint(kk)
        import time as _t; _t0 = _t.time()
        sel = select(M, N, K, dtype, layout, prof, k=k)
        STATE['select_s'] += _t.time() - _t0
        STATE['log'].append((M, N, K, dtype, sel[0]))
        for c in sel:
            yield triton.Config(dict(BLOCK_M=c['BLOCK_M'], BLOCK_N=c['BLOCK_N'], BLOCK_K=c['BLOCK_K'], GROUP_M_OVERRIDE=c['GROUP_M']),
                                num_stages=c['num_stages'], num_warps=c['num_warps'])
    return configs

def _wrap(opt_fn):
    def options(config, *args, **kw):
        kwargs = dict(config.kwargs); g = kwargs.pop('GROUP_M_OVERRIDE', None)
        out = opt_fn(triton.Config(kwargs, num_stages=config.num_stages, num_warps=config.num_warps), *args, **kw)
        if g is not None: out['GROUP_M'] = g
        return out
    return options

def enable(profile, k=1, bf16_layout='AB'):
    prof = _load(profile) if isinstance(profile, str) else profile
    # Inductor's FX-graph cache key does not include our patch: a cache hit would skip the config hook entirely (silently stock configs).
    torch._inductor.config.fx_graph_cache = False
    _mm.mm_configs, _mm.mm_options = _gen(prof, 'bf16', bf16_layout, k), _wrap(_ORIG['mm_options'])
    _ms.scaled_mm_configs, _ms.scaled_mm_options = _gen(prof, 'fp8', 'ABT', k), _wrap(_ORIG['scaled_mm_options'])
    STATE['on'] = True

def disable():
    _mm.mm_configs, _mm.mm_options = _ORIG['mm_configs'], _ORIG['mm_options']
    _ms.scaled_mm_configs, _ms.scaled_mm_options = _ORIG['scaled_mm_configs'], _ORIG['scaled_mm_options']
    STATE['on'] = False
