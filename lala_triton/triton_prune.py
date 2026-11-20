"""prune_configs_by hook for plain @triton.autotune kernels.
    @triton.autotune(configs=[...], key=['M','N','K'], prune_configs_by=lala_triton.triton_prune.prune_configs_by(profile, 'bf16', k=5))
Ranks the kernel's own configs with the model and keeps the top k (k=1 -> no benchmarking). Names map your kernel's constexpr names
(default: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M or GROUP_M; M, N, K are read from the kernel args)."""
from . import model as M3
from .profile import load as _load

def prune_configs_by(profile, dtype='bf16', layout='AB', k=5, names=None):
    prof = _load(profile) if isinstance(profile, str) else profile
    nm = dict(BM='BLOCK_M', BN='BLOCK_N', BK='BLOCK_K', G=('GROUP_SIZE_M', 'GROUP_M'), M='M', N='N', K='K'); nm.update(names or {})
    def pick(d, key):
        v = nm[key]
        for x in (v if isinstance(v, tuple) else (v,)):
            if x in d: return d[x]
        return None
    def early_config_prune(configs, named_args, **kw):
        M, N, K = (int(named_args[nm[x]]) for x in 'MNK')
        scored = []
        for c in configs:
            kw_ = c.all_kwargs() if hasattr(c, 'all_kwargs') else dict(c.kwargs)
            bm, bn, bk = (pick(kw_, x) for x in ('BM', 'BN', 'BK')); g = pick(kw_, 'G') or 8
            if None in (bm, bn, bk): scored.append((float('inf'), c)); continue
            cfg = dict(BM=bm, BN=bn, BK=bk, G=g, num_warps=c.num_warps, num_stages=c.num_stages)
            scored.append((M3.predict(M, N, K, cfg, dtype, prof, layout), c))
        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored[:k]]
    return dict(early_config_prune=early_config_prune)
