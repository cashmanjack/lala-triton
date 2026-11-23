"""Amendment 5d offline diagnostic (no GPU, no timing): top-1 picks of plain vs cap-only vs cap+pad on the fresh Gemma 2 9B shapes."""
import sys, os, json; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lala_triton import model as M3
from lala_triton.select import select_scored
from lala_triton.profile import load
from lala_triton.models import gemm_shapes
prof = load('profiles/rtx6000ada.json'); rows = []
def cap_only(M, N, K, dt):
    sp = [t for t in __import__('lala_triton.select', fromlist=['_space'])._space(prof, dt, 'ABT') if t['BM'] <= M3.bm_cap(M)]
    return M3.rank(M, N, K, dt, 'ABT', prof, sp, Gs=(8, 16, 32, 4, 1), k=1)[0][1]
k = lambda c: f"{c['BM']}x{c['BN']}x{c['BK']}_w{c['num_warps']}_s{c['num_stages']}_G{c['G']}"
for dt in ('bf16', 'fp8'):
    for g, (N, K) in gemm_shapes('gemma2-9b').items():
        for M in (1, 8, 32, 128, 512, 2048, 8192):
            a = select_scored(M, N, K, dt, 'ABT', prof, k=1, padcap=False)[0][1]; b = cap_only(M, N, K, dt); c = select_scored(M, N, K, dt, 'ABT', prof, k=1, padcap=True)[0][1]
            rows.append(dict(dtype=dt, gemm=g, M=M, plain=k(a), cap_only=k(b), cap_pad=k(c), plain_BM=a['BM'], cap_changes=k(a) != k(b), pad_changes=k(b) != k(c)))
import pandas as pd; d = pd.DataFrame(rows); d.to_csv('results/gemma_5d_offline_picks.csv', index=False)
print(d.groupby('dtype')[['cap_changes', 'pad_changes']].sum()); print('plain BM>cap(M):'); print(d[d.plain_BM > d.M.map(M3.bm_cap)][['dtype', 'gemm', 'M', 'plain', 'cap_pad']].to_string())
print('pad changes:'); print(d[d.pad_changes][['dtype', 'gemm', 'M', 'cap_only', 'cap_pad']].to_string())
