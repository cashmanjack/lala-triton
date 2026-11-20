"""Driver: python3 run_ncu.py pred_csv tag gpu. One ncu process per (sweep, shape); skips completed ones."""
import sys, os, subprocess, pandas as pd
pred, tag, gpu = sys.argv[1], sys.argv[2], sys.argv[3]
R = '/home/min/a/cashman3/lala-triton-killtest'
df = pd.read_csv(pred)
keys = ['sweep', 'M', 'N', 'K'] + (['dtype'] if 'dtype' in df else [])
os.makedirs(f'{R}/runs/ncu_{tag}', exist_ok=True)
METRICS = 'dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_hit_rate.pct'
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TRITON_CACHE_DIR=f'{R}/.triton_cache')
for k, _ in df.groupby(keys, sort=False):
    name = '_'.join(map(str, k)); out = f'{R}/runs/ncu_{tag}/{name}'
    if os.path.exists(out + '.done'): continue
    cmd = ['/usr/local/cuda/bin/ncu', '--csv', '--metrics', METRICS, '--cache-control', 'all', '-k', 'regex:matmul_kernel',
           '/usr/bin/python3', f'{R}/run_ncu_target.py', pred, str(k[0]), str(k[1]), str(k[2]), str(k[3]), out + '.manifest.json']
    if len(k) > 4: cmd.append(str(k[4]))
    with open(out + '.csv', 'w') as fo, open(out + '.err', 'w') as fe:
        rc = subprocess.run(cmd, stdout=fo, stderr=fe, env=env).returncode
    if rc == 0: open(out + '.done', 'w').close()
    print(name, rc, flush=True)
