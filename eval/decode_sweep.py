"""Round 6 Step 1 sweep (thin wrapper around lala_triton.decode.sweep). python3 eval/decode_sweep.py --gpu 0 --shard 0/2 --out results/r6/decode_sweep_s0.csv"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--shard', default='0/1'); ap.add_argument('--out', required=True)
a = ap.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
from lala_triton import decode
decode.sweep(a.out, shard=tuple(map(int, a.shard.split('/'))), log=lambda *x: print(*x, flush=True))
