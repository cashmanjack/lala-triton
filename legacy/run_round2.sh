#!/bin/bash
# Master chain for round 2 (GPU0). Resumable: re-running skips finished work.
cd /home/min/a/cashman3/lala-triton-killtest
export TRITON_CACHE_DIR=/home/min/a/cashman3/lala-triton-killtest/.triton_cache
while ! grep -q "PHASE B DONE" logs/phaseB.log; do
  if ! pgrep -f "^python3 phaseB.py" >/dev/null; then echo "phaseB not running and not done; restarting"; nohup python3 phaseB.py >> logs/phaseB.log 2>&1 & fi
  sleep 60
done
echo "[$(date)] phase B done"
python3 phaseB_window.py
[ -f runs/phaseC_verdict.json ] || python3 phaseC.py > logs/phaseC.log 2>&1
CHOICE=$(python3 -c "import json;print(json.load(open('runs/phaseC_verdict.json'))['choice'])")
echo "[$(date)] model choice $CHOICE"; [ -n "$CHOICE" ] || { echo "no model choice; abort"; exit 1; }
[ -f predictions/phaseDE_pred.csv ] || python3 phaseD_predict.py > logs/phaseD_predict.log 2>&1 || { echo "predict failed; abort"; exit 1; }
for S in $(python3 -c "
import sys; sys.path.insert(0,'.')
from phaseD_shapes import SHAPES, key
print(' '.join(key(*s) for s in SHAPES))"); do
  [ -f runs/phaseD_tune/$S.json ] || python3 phaseD_tune.py $S $CHOICE 0 >> logs/phaseD_tune.log 2>&1
  if ! grep -q "^$S," runs/phaseD_eval.csv 2>/dev/null || [ $(grep -c "^$S," runs/phaseD_eval.csv) -lt 15 ]; then
    python3 phaseD_eval.py $S 0 >> logs/phaseD_eval.log 2>&1
  fi
  echo "[$(date)] D $S"
done
echo "[$(date)] phase D done"
python3 phaseE.py >> logs/phaseE.log 2>&1
echo "[$(date)] ALL DONE"
