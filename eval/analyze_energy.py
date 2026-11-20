"""Kernel energy per GEMM (NVML total-energy counter over the timed window / GEMMs) for ours top-1 vs stock max-autotune on the 8B validation shapes.
Flags shapes where the energy ratio departs from the runtime ratio by >5% (on a power-capped card energy ~ power cap x time, so ratios should track)."""
import glob, json, numpy as np, pandas as pd
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results/energy_val_s*.csv'))])
k = ['gemm', 'dtype', 'M']
g = d.groupby(k + ['method'])[['ms', 'energy_J', 'energy_W', 'sm_clk']].median().unstack('method')
r = pd.DataFrame({'t_ratio': g['ms']['ours1'] / g['ms']['ind_auto'], 'E_ratio': g['energy_J']['ours1'] / g['energy_J']['ind_auto'],
                  'W_ours': g['energy_W']['ours1'], 'W_stock': g['energy_W']['ind_auto'], 'E_ours_mJ': g['energy_J']['ours1'] * 1e3, 'E_stock_mJ': g['energy_J']['ind_auto'] * 1e3})
r['E_over_t'] = r.E_ratio / r.t_ratio; r['tracks'] = (r.E_over_t - 1).abs() <= 0.05
gm = lambda x: float(np.exp(np.log(x).mean()))
out = {dt: dict(n=len(x), geomean_time_ratio=gm(x.t_ratio), geomean_energy_ratio=gm(x.E_ratio), median_power_W_ours=float(x.W_ours.median()), median_power_W_stock=float(x.W_stock.median()),
                n_energy_not_tracking_time=int((~x.tracks).sum()), worst_energy_ratio=float(x.E_ratio.max()), worst_energy_shape=str(x.E_ratio.idxmax()))
       for dt, x in r.groupby(level='dtype')}
print(json.dumps(out, indent=1)); print(r[~r.tracks].round(3).to_string())
r.to_csv('results/energy_validation_per_shape.csv'); json.dump(out, open('results/energy_validation_summary.json', 'w'), indent=1)
