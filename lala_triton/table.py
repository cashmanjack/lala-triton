"""python -m lala_triton.table --model llama3-8b --profile profiles/rtx6000ada.json --dtype bf16 --out table.json [--vllm-dir DIR]
Per-M config table (M = 1..8192) for every GEMM in a model. --out: {gemm: {"N":..,"K":..,"configs": {M: {BLOCK_SIZE_M,...}}}};
--vllm-dir additionally writes vLLM-layout files `N=<N>,K=<K>,device_name=<gpu>[,dtype=fp8_w8a8].json` mapping M -> config."""
import argparse, json, os
from .models import gemm_shapes
from .profile import load
from .select import select

MS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
def vllm_cfg(c): return dict(BLOCK_SIZE_M=c['BLOCK_M'], BLOCK_SIZE_N=c['BLOCK_N'], BLOCK_SIZE_K=c['BLOCK_K'], GROUP_SIZE_M=c['GROUP_M'],
                             num_warps=c['num_warps'], num_stages=c['num_stages'])

def build(model, profile, dtype, layout=None, tp=1):
    prof = load(profile) if isinstance(profile, str) else profile
    layout = layout or ('ABT' if dtype == 'fp8' else 'ABT')      # LLM weights are stored [N,K] (nn.Linear): B = W.T
    out = {}
    for name, (N, K) in gemm_shapes(model, tp).items():
        out[name] = dict(N=N, K=K, configs={str(M): vllm_cfg(select(M, N, K, dtype, layout, prof, k=1)[0]) for M in MS})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--profile', required=True)
    ap.add_argument('--dtype', default='bf16'); ap.add_argument('--layout'); ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--out', required=True); ap.add_argument('--vllm-dir')
    a = ap.parse_args()
    prof = load(a.profile); t = build(a.model, prof, a.dtype, a.layout, a.tp)
    json.dump(dict(model=a.model, dtype=a.dtype, gpu=prof.get('gpu'), generator='lala_triton (reimplementation)', gemms=t), open(a.out, 'w'), indent=1)
    if a.vllm_dir:
        os.makedirs(a.vllm_dir, exist_ok=True)
        dn = (prof.get('gpu') or 'gpu').replace(' ', '_')
        for g in t.values():
            suf = ',dtype=fp8_w8a8' if a.dtype == 'fp8' else ''
            json.dump(g['configs'], open(f"{a.vllm_dir}/N={g['N']},K={g['K']},device_name={dn}{suf}.json", 'w'), indent=1)
    print('wrote', a.out)
if __name__ == '__main__': main()
