LLM = [(m, n, k) for m in (128, 512, 2048, 8192) for n, k in [(4096, 4096), (14336, 4096), (4096, 14336), (28672, 8192), (8192, 28672)]]
SHAPES = [(*s, 'bf16', 'AB') for s in LLM + [(2048, 2048, 65536), (4096, 4096, 4096)]] + [(*s, 'fp8', 'ABT') for s in LLM] + \
         [(*s, 'bf16', 'ABT') for s in LLM]
def key(M, N, K, dt, lay): return f'{M}x{N}x{K}_{dt}_{lay}'
def r1key(M, N, K, dt): return f'{M}x{N}x{K}_{dt}'
