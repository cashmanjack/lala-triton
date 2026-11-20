#!/bin/bash
# GPU0 chain: wait for Step 1, then 2c final evaluation shape by shape as GPU1 tuning finishes. Resumable.
cd /home/min/a/cashman3/lala-triton-killtest
until grep -q "STEP1 DONE" logs/r3/step1.log; do sleep 30; done
echo "[$(date)] step1 done"
SHAPES=$(python3 -c "
import sys; sys.path.insert(0,'.')
from phaseD_shapes import SHAPES, key
print(' '.join(key(*s) for s in SHAPES))")
for S in $SHAPES; do
  until grep -q "^tuned $S " logs/r3/tune.log 2>/dev/null; do sleep 30; done
  python3 r3_eval2c.py $S >> logs/r3/eval2c.log 2>&1
done
echo "[$(date)] EVAL2C CHAIN DONE"
