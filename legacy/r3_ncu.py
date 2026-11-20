"""2b unit test: 50 random (Phase D shape, E config) template kernels; M3 A+B read bytes vs ncu dram__bytes_read.sum
(--cache-control all, GPU1). Subset + predictions written first, then ncu per shape. Resumable."""
import sys, os, json, subprocess, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
OUT = f'{R}/runs/r3_ncu'; os.makedirs(OUT, exist_ok=True)
SUB = f'{R}/predictions/r3_ncu_subset.csv'
if not os.path.exists(SUB):
    p = pd.read_csv(f'{R}/predictions/r3_pred.csv'); p = p[p.phase == 'D']
    p.sample(50, random_state=0).reset_index(drop=True).to_csv(SUB, index_label='idx')
sub = pd.read_csv(SUB)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', TRITON_CACHE_DIR='/var/tmp/cashman3_r3/tc_ncu')
for shape, g in sub.groupby('shape'):
    out = f'{OUT}/{shape}'
    if os.path.exists(out + '.done'): continue
    cmd = ['/usr/local/cuda/bin/ncu', '--csv', '--metrics', 'dram__bytes_read.sum,dram__bytes_write.sum', '--cache-control', 'all',
           '-k', 'regex:mm_tmpl', '/usr/bin/python3', f'{R}/r3_ncu_target.py', shape, ','.join(map(str, g.idx)), out + '.manifest.json']
    with open(out + '.csv', 'w') as fo, open(out + '.err', 'w') as fe:
        rc = subprocess.run(cmd, stdout=fo, stderr=fe, env=env).returncode
    if rc == 0: open(out + '.done', 'w').close()
    print(shape, rc, flush=True)
# parse
UNIT = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9}
import csv
recs = []
for shape, g in sub.groupby('shape'):
    base = f'{OUT}/{shape}'; idx = json.load(open(base + '.manifest.json'))
    rows = [r for r in csv.reader(open(base + '.csv')) if len(r) > 12 and r[0].isdigit()]
    ids = sorted(set(int(r[0]) for r in rows)); assert len(ids) == len(idx), (shape, len(ids), len(idx))
    for kid, pi in zip(ids, idx):
        m = {r[12]: float(r[14].replace(',', '')) * UNIT.get(r[13], 1) for r in rows if int(r[0]) == kid}
        recs.append(dict(idx=pi, meas_read=m['dram__bytes_read.sum'], meas_write=m['dram__bytes_write.sum']))
j = sub.merge(pd.DataFrame(recs), on='idx')
j['err'] = (j.m3_read - j.meas_read) / j.meas_read
j.to_csv(f'{R}/runs/r3_ncu_joined.csv', index=False)
med = j.err.abs().median()
print(f'n={len(j)} median |err|={med:.4f} p90={j.err.abs().quantile(.9):.3f} max={j.err.abs().max():.3f} PASS={med <= 0.10}')
json.dump(dict(n=len(j), median_abs_err=med, p90=j.err.abs().quantile(.9), max=j.err.abs().max(), pass_=bool(med <= 0.10)),
          open(f'{R}/runs/r3_ncu_verdict.json', 'w'), indent=1)
