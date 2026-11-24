"""Per-layer GEMM shapes (N, K) of dense decoder models; M is the token count. TP=1 shapes (each single GEMM fits on a 48 GB GPU)."""
MODELS = {
    'llama3-8b':  dict(hidden=4096, inter=14336, heads=32, kv=8, hd=128),
    'qwen2.5-7b': dict(hidden=3584, inter=18944, heads=28, kv=4, hd=128),
    'llama3-70b': dict(hidden=8192, inter=28672, heads=64, kv=8, hd=128),
    'gemma2-9b':  dict(hidden=3584, inter=14336, heads=16, kv=8, hd=256),   # Amendment 5d fresh set; o-proj K = 16*256 = 4096 != hidden
    # Round 6 fresh set; values from the published HF config.json (mistralai/Mistral-Nemo-Base-2407, microsoft/Phi-3-medium-4k-instruct), fetched 2026-11-23.
    'mistral-nemo-12b': dict(hidden=5120, inter=14336, heads=32, kv=8, hd=128),    # head_dim given explicitly in config; o-proj K = 32*128 = 4096 != hidden
    'phi3-medium':      dict(hidden=5120, inter=17920, heads=40, kv=10, hd=128),   # no head_dim key: 5120/40 = 128; fused qkv_proj / gate_up_proj in the HF model
}
def gemm_shapes(model, tp=1):
    c = MODELS[model]; h, i = c['hidden'], c['inter']
    qkv = (c['heads'] + 2 * c['kv']) * c['hd']
    return {'qkv': (qkv // tp, h), 'o': (h, c['heads'] * c['hd'] // tp), 'gate_up': (2 * i // tp, h), 'down': (h, i // tp)}
