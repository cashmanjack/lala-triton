"""Repeat the timing-based L2 knee probe (the same code as calibrate.stage_micro) under a fresh name. python3 eval/probe_repeat.py <gpu> <name>"""
import os, sys, json, types
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lala_triton import calibrate
r = calibrate.stage_micro(types.SimpleNamespace(), sys.argv[2])
print(sys.argv[2], 'gpu', sys.argv[1], 'knee', r['timing_knee_MiB'], 'hbm', r['hbm_bw'] / 1e9, 'l2', r['l2_bw'] / 1e12)
