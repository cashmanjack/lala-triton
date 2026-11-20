import json, pandas as pd
B = pd.read_csv('/home/min/a/cashman3/lala-triton-killtest/runs/phaseB.csv')
med = float(B.settle_s.median())
w = dict(warmup_s=0.5, window_s=max(1.5, 2 * med), cooldown_s=1.0, phaseB_median_settle_s=med, n=len(B))
json.dump(w, open('/home/min/a/cashman3/lala-triton-killtest/env/phaseD_window.json', 'w'), indent=1); print(w)
