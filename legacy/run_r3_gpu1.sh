#!/bin/bash
# GPU1 chain for round 3 (after Amendment 3b): calibration -> predictions -> ncu unit test -> compile verification
# -> 2c tuning/screening -> 2d. Resumable.
cd /home/min/a/cashman3/lala-triton-killtest
grep -q "CALIB DONE" logs/r3/calib.log 2>/dev/null || python3 r3_calib.py > logs/r3/calib.log 2>&1
[ -f predictions/r3_pred.csv ] || python3 r3_predict.py > logs/r3/predict.log 2>&1
[ -f runs/r3_ncu_verdict.json ] || python3 r3_ncu.py > logs/r3/ncu.log 2>&1
tail -1 logs/r3/ncu.log
python3 r3_verify_run.py > logs/r3/verify_compile.log 2>&1; tail -1 logs/r3/verify_compile.log
python3 r3_tune.py >> logs/r3/tune.log 2>&1
grep -q "TUNE DONE" logs/r3/tune.log || { echo "tune incomplete; stop"; exit 1; }
echo "[$(date)] GPU1 2c tuning done"
python3 r3_phaseE.py >> logs/r3/phaseE.log 2>&1
echo "[$(date)] GPU1 phase E done"
