cd /home/min/a/cashman3/lala-triton-killtest
export CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/min/a/cashman3/lala-triton-killtest/.triton_cache
for s in $(python3 -c "import pandas as pd; d=pd.read_csv('predictions/phaseH_cliff.csv'); print(' '.join(f'{a}_{b}_{c}' for a,b,c in d[['M','N','K']].drop_duplicates().values))"); do
  IFS=_ read M N K <<< "$s"; [ -f runs/ncu_phaseH/$s.done ] && continue
  /usr/local/cuda/bin/ncu --csv --metrics dram__bytes_read.sum,lts__t_sector_op_read_hit_rate.pct --cache-control all -k regex:matmul_kernel_ev \
     /usr/bin/python3 phaseH.py ncu_target $M $N $K > runs/ncu_phaseH/$s.csv 2> runs/ncu_phaseH/$s.err && touch runs/ncu_phaseH/$s.done; echo $s
done
echo NCU H DONE
