"""Expose GROUP_M (and an explicit config list) for Inductor's Triton mm / scaled_mm templates in torch 2.6 without forking.
Usage: import inductor_patch; inductor_patch.set_configs([(BM, BN, BK, stages, warps, G), ...]) before torch.compile.
set_configs(None) restores stock behaviour. The configs are used verbatim (no size-based shrinking)."""
import triton
import torch._inductor.lowering  # noqa: F401  (import order: avoids a circular import in torch 2.6)
import torch._inductor.kernel.mm as _mm
import torch._inductor.kernel.mm_scaled as _ms

_orig = dict(mm_configs=_mm.mm_configs, mm_options=_mm.mm_options,
             scaled_mm_configs=_ms.scaled_mm_configs, scaled_mm_options=_ms.scaled_mm_options)
_CFGS = None


def _configs(m, n, k, *args, **kw):
    for bm, bn, bk, s, w, g in _CFGS:
        yield triton.Config(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M_OVERRIDE=g), num_stages=s, num_warps=w)


def _wrap(opt_fn):
    def options(config, *args, **kw):
        kwargs = dict(config.kwargs); g = kwargs.pop('GROUP_M_OVERRIDE', None)
        out = opt_fn(triton.Config(kwargs, num_stages=config.num_stages, num_warps=config.num_warps), *args, **kw)
        if g is not None: out['GROUP_M'] = g
        return out
    return options


def set_configs(cfgs):
    global _CFGS
    _CFGS = cfgs
    if cfgs is None:
        _mm.mm_configs, _mm.mm_options = _orig['mm_configs'], _orig['mm_options']
        _ms.scaled_mm_configs, _ms.scaled_mm_options = _orig['scaled_mm_configs'], _orig['scaled_mm_options']
    else:
        _mm.mm_configs, _mm.mm_options = _configs, _wrap(_orig['mm_options'])
        _ms.scaled_mm_configs, _ms.scaled_mm_options = _configs, _wrap(_orig['scaled_mm_options'])
