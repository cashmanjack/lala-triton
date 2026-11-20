"""2a verification (i): 12 random (shape, E config) pairs (seed 0, 4 per family) -> runs/r3_verify_compile.jsonl"""
import subprocess, json, os, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
OUT = f'{R}/runs/r3_verify_compile.jsonl'
p = pd.read_csv(f'{R}/predictions/r3_pred.csv'); p = p[p.phase == 'D']
sub = pd.concat([g.sample(4, random_state=0) for _, g in p.groupby(['dtype', 'layout'])])
done = {(json.loads(l)['shape'], json.loads(l)['cfg']) for l in open(OUT)} if os.path.exists(OUT) else set()
for r in sub.itertuples():
    if (r.shape, r.cfg) in done: continue
    subprocess.run(['/usr/bin/python3', f'{R}/r3_verify_compile.py', r.shape, r.cfg, OUT], stderr=subprocess.DEVNULL)
rs = [json.loads(l) for l in open(OUT)]
ok = all(x['ind_reg'] == x['h_reg'] and x['sass_diff_lines'] == 0 and x['out_maxdiff'] == 0 for x in rs)
print(len(rs), 'identical SASS/regs/output' if ok else 'MISMATCH', [ (x['shape'], x['cfg'], x['sass_diff_lines'], x['out_maxdiff']) for x in rs])
