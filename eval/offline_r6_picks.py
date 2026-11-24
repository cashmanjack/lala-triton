"""Round 6: OFFLINE (no timing) picks and predicted times of every selector variant on given models. Written BEFORE the validation timing (predictions-before-measurements).
python3 eval/offline_r6_picks.py --profile profiles/rtx6000ada_r6.json --models mistral-nemo-12b,phi3-medium --out results/r6/predictions_fresh.csv
Also asserts (reports) that 5e/5f never change a pick at M>64 versus 5d (ours1pc)."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lala_triton import model as M3, select as S, profile as P, decode as D
from lala_triton.models import gemm_shapes
ap = argparse.ArgumentParser(); ap.add_argument('--profile', required=True); ap.add_argument('--models', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--ms', default='1,8,32,128,512,2048,8192'); ap.add_argument('--eps', type=float, default=0.03); a = ap.parse_args()
prof = P.load(a.profile); VAR = {'ours1': (False, False, False), 'ours1pc': (True, False, False), 'ours1e': (True, True, False), 'ours1f': (True, False, True), 'ours1ef': (True, True, True)}
rows, diff_large = [], 0
for m in a.models.split(','):
    for g, (N, K) in gemm_shapes(m).items():
        for dt in ('bf16', 'fp8'):
            for M in map(int, a.ms.split(',')):
                picks = {}
                for v, (pad, e, f) in VAR.items():
                    M3.PAD, M3.DEC5E, M3.DEC5F = pad, e, f; D.FALLBACK.clear()
                    t, c = S.select_scored(M, N, K, dt, 'ABT', prof, k=1)[0]; picks[v] = c
                    rows.append(dict(model=m, gemm=g, dtype=dt, M=M, N=N, K=K, method=v, pred_ms=t * 1e3, tile=f"{c['BM']}x{c['BN']}x{c['BK']}_w{c['num_warps']}_s{c['num_stages']}", n_tied_group='', n_returned='',
                                     table_ok=D.FALLBACK['ok'] + D.FALLBACK['clamped_hi_active'], table_fallback=D.FALLBACK['out_of_range'] + D.FALLBACK['no_table']))
                if M > 64: diff_large += sum(picks[v] != picks['ours1pc'] for v in ('ours1e', 'ours1f', 'ours1ef'))
                M3.PAD, M3.DEC5E, M3.DEC5F = True, True, False
                pk = S.select_auto_scored(M, N, K, dt, 'ABT', prof, eps=a.eps)
                rows.append(dict(model=m, gemm=g, dtype=dt, M=M, N=N, K=K, method='ours_auto', pred_ms=pk[0][0] * 1e3, tile=';'.join(f"{c['BM']}x{c['BN']}x{c['BK']}_w{c['num_warps']}_s{c['num_stages']}" for _, c in pk),
                                 n_tied_group=S.AUTO_STATS[-1][4], n_returned=len(pk), table_ok='', table_fallback=''))
M3.PAD = M3.DEC5E = M3.DEC5F = False
with open(a.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
n = sum(r['M'] <= 64 for r in rows if r['method'] == 'ours1f'); fb = sum(r['table_fallback'] > 0 for r in rows if r['method'] == 'ours1f' and r['M'] <= 64)
au = [r for r in rows if r['method'] == 'ours_auto']
print(f'{len(rows)} rows; picks differing from ours1pc at M>64 for 5e/5f/5ef: {diff_large}; 5f shapes M<=64: {n}, with >=1 fallback in the ranking: {fb}')
print('ours_auto: shapes', len(au), 'benchmarking (n_returned>1):', sum(r['n_returned'] > 1 for r in au), ' M<=32:', sum(r['n_returned'] > 1 for r in au if r['M'] <= 32), 'of', sum(r['M'] <= 32 for r in au))
