import csv, json, glob, numpy as np, pandas as pd
R = '/home/min/a/cashman3/lala-triton-killtest'
from ncu_parse import UNIT
recs = []
for man in glob.glob(f'{R}/runs/ncu_phaseH/*.manifest.json'):
    base = man[:-len('.manifest.json')]; M, N, K = map(int, base.split('/')[-1].split('_'))
    items = json.load(open(man)); rows = [r for r in csv.reader(open(base + '.csv')) if len(r) > 12 and r[0].isdigit()]
    ids = sorted(set(int(r[0]) for r in rows)); assert len(ids) == len(items), base
    for kid, (tile, G, v) in zip(ids, items):
        mm = {r[12]: float(r[14].replace(',', '')) * UNIT.get(r[13], 1) for r in rows if int(r[0]) == kid}
        recs.append(dict(M=M, N=N, K=K, tile=tile, G=G, variant=v, dram=mm['dram__bytes_read.sum'], l2hit=mm['lts__t_sector_op_read_hit_rate.pct']))
b = pd.DataFrame(recs)
t = pd.read_csv(f'{R}/runs/phaseH_time.csv').groupby(['M', 'N', 'K', 'tile', 'G', 'variant']).ms_per_gemm.mean().reset_index()
d = b.merge(t, on=['M', 'N', 'K', 'tile', 'G', 'variant'], how='left')
cl = pd.read_csv(f'{R}/predictions/phaseH_cliff.csv')[['M', 'N', 'K', 'tile', 'G', 'near', 'compulsory', 'pred_read']]
d = d.merge(cl, on=['M', 'N', 'K', 'tile', 'G'])
w = d.pivot_table(index=['M', 'N', 'K', 'tile', 'G', 'near', 'compulsory', 'pred_read'], columns='variant', values=['dram', 'ms_per_gemm']).reset_index()
w.columns = ['_'.join(c).strip('_') for c in w.columns]
w['rec'] = np.where(w.near == 'F3', 'Blast', 'Alast')   # model-recommended: keep the operand whose reuse is at the cliff
w['bytes_rec_over_none'] = [getattr(r, f'dram_{r.rec}') / r.dram_none for r in w.itertuples()]
w['bytes_best_over_none'] = w[['dram_Alast', 'dram_Blast']].min(axis=1) / w.dram_none
w['time_rec_over_none'] = [getattr(r, f'ms_per_gemm_{r.rec}') / r.ms_per_gemm_none for r in w.itertuples()]
w['excess_none'] = w.dram_none / w.compulsory; w['excess_rec'] = [getattr(r, f'dram_{r.rec}') / r.compulsory for r in w.itertuples()]
pd.set_option('display.width', 250)
print(w[['M', 'N', 'K', 'tile', 'G', 'near', 'excess_none', 'excess_rec', 'bytes_rec_over_none', 'bytes_best_over_none', 'time_rec_over_none']].round(3).to_string())
print('\nmedian bytes(rec)/bytes(none) =', round(w.bytes_rec_over_none.median(), 3), '; configs where rec cuts bytes >=20%:', (w.bytes_rec_over_none <= 0.8).sum(), '/', len(w))
print('median time(rec)/time(none) =', round(w.time_rec_over_none.median(), 3), '; faster by >=5%:', (w.time_rec_over_none <= 0.95).sum(), ' slower by >=5%:', (w.time_rec_over_none >= 1.05).sum())
print('by near-type:'); print(w.groupby('near')[['bytes_rec_over_none', 'time_rec_over_none']].median().round(3))
w.to_csv(f'{R}/runs/phaseH_summary.csv', index=False)
