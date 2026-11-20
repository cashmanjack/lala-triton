"""Per-layer GEMM shapes (N, K) of dense decoder models; M is the token count. TP=1 shapes (each single GEMM fits on a 48 GB GPU)."""
MODELS = {
    'llama3-8b':  dict(hidden=4096, inter=14336, heads=32, kv=8, hd=128),
    'llama3-70b': dict(hidden=8192, inter=28672, heads=64, kv=8, hd=128),
}
def gemm_shapes(model, tp=1):
    c = MODELS[model]; h, i = c['hidden'], c['inter']
    qkv = (c['heads'] + 2 * c['kv']) * c['hd']
    return {'qkv': (qkv // tp, h), 'o': (h, h // tp), 'gate_up': (2 * i // tp, h), 'down': (h, i // tp)}
