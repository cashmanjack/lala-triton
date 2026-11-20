"""Sustained-protocol timing of one (shape, config) with our kernel (R5-C: warmup 0.5 s / window 1.5 s / cooldown 1 s by default)."""
import torch
from .kernels import launch
from ._sustained import make_graph, measure, buffer_sets

class Shape:
    def __init__(self, M, N, K, dtype, layout):
        self.M, self.N, self.K, self.dtype, self.layout = M, N, K, dtype, layout
        self.sets = buffer_sets(M, N, K, dtype, layout)
    def close(self): del self.sets; torch.cuda.empty_cache()

def time_cfg(sh, cfg, gpu, warmup_s=0.5, window_s=1.5, cooldown_s=1.0):
    kern = launch(*sh.sets[0], cfg); torch.cuda.synchronize()        # compile + metadata
    meta = dict(n_regs=kern.n_regs, n_spills=kern.n_spills, shared=kern.metadata.shared)
    if meta['n_spills'] > 0 or meta['shared'] > 99 * 1024: return None, meta
    fns = [lambda a=a, b=b, c=c: launch(a, b, c, cfg) for a, b, c in sh.sets]
    g, L, est = make_graph(fns)
    res, _ = measure(g, L, gpu, warmup_s=warmup_s, window_s=window_s, cooldown_s=cooldown_s)
    del g
    return res, meta
