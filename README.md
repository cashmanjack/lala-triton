# lala_triton (REIMPLEMENTATION, round 5)

Picks Triton GEMM configs from a traffic model (M3: L2-aware DRAM bytes) plus a sustained-load time model, with no or little benchmarking.
**This is a reimplementation**: the round 1-3 model code and data were lost; formulas come from `legacy/triton-locality-killtest.md` Section 2, legacy/REPORT*.md and
`PREREGISTRATION.md` (Amendments 5/5a/5b/5c/5d). It has not yet been checked against ncu DRAM bytes (counters are blocked on the dev machine; gate R5-0 pending).

## Install
    pip install -e . --no-deps          # torch 2.6 + triton 3.2 assumed; set LALA_TRITON_ROOT to relocate runs/ and profiles/
    python3 -m pytest -p no:anyio tests # 186 model unit tests, no GPU needed
(If pip reports the package as `UNKNOWN-0.0.0`, your setuptools is older than 61: `pip install -U setuptools`. The editable install still imports fine.)

## Calibrate a new GPU (about 45 min on 2 GPUs; one GPU: use --shard 0/1, about 80 min)
    nohup python3 -m lala_triton.calibrate --out profiles/<gpu>.json --gpu 0 --shard 0/2 > c0.log &
    nohup python3 -m lala_triton.calibrate --out profiles/<gpu>.json --gpu 1 --shard 1/2 > c1.log &
    # when both print "calibration shard done":
    python3 -m lala_triton.calibrate --out profiles/<gpu>.json --shard 0/2 --build-only
Measures HBM and L2 bandwidth, a timing-based L2 knee (sanity check only), and sustained eff for ~1100 tiles on 8192^3 (3 families), then fits the
Little's-law latency per dtype on small-M calibration shapes. **L2_eff is a constant, not measured**: it defaults to 102 MiB (RTX 6000 Ada, round 1).
For a new GPU pass `--l2-eff-mib X` (nominal L2 x ~1.0 is a reasonable start: the RTX 6000 Ada has 96 MiB nominal, as `torch.cuda.get_device_properties(0).L2_cache_size` reports, and uses 102 MiB; an L40S is the same AD102 die with 96 MiB of L2, so use `--l2-eff-mib 102` there too, and confirm with that same call) and, if ncu counters work, verify with a double-read (`lala_triton/_l2probe.py` under ncu, `dram__bytes_read.sum`).
Needs NVML (pynvml) for clocks; power-capped GPUs only make sense with the sustained protocol, which is why eff is measured that way.

## Use
    import lala_triton.inductor as ti
    ti.enable('profiles/rtx6000ada.json', k=1, bf16_layout='ABT')   # nn.Linear weights are [N,K] -> 'ABT'
    # set torch._inductor.config.max_autotune_gemm_backends = 'TRITON' so Inductor keeps our pick instead of benchmarking ATen against it
    f = torch.compile(model, mode='max-autotune-no-cudagraphs'); ...; ti.disable()
k=1: no benchmarking; k=5: Inductor times our 5 picks. Plain Triton: `@triton.autotune(..., prune_configs_by=lala_triton.triton_prune.prune_configs_by(profile, 'bf16', 'ABT', k=5))`.
Direct: `lala_triton.select.select(M, N, K, 'bf16', 'ABT', profile, k=5)` returns Inductor-format dicts.
Per-M tables: `python3 -m lala_triton.table --model llama3-8b --profile profiles/rtx6000ada.json --dtype bf16 --out table.json --vllm-dir tables/`.
Evaluation: `python3 eval/run_eval.py --profile ... --models llama3-8b,llama3-70b --dtypes bf16,fp8 --out results/` (resumable; see its docstring).

## Rerun the fresh Gemma 2 9B set on another GPU (Amendment 5d validation; e.g. Dongyoon's L40S)
    python3 eval/run_eval.py --profile profiles/<gpu>.json --models gemma2-9b --dtypes bf16,fp8 --ms 1,8,32,128,512,2048,8192 \
        --methods ind_auto,ours1,ours1pc,ours1tb,ours5d,tritonblas,oracle --oracle-ms 1,8,32 --tag fresh_gemma_<gpu>_s0 --out results --gpu 0
    python3 eval/analyze_5d.py fresh_gemma_<gpu>      # reads results/fresh_gemma_<gpu>_s*.csv, prints the (i)-(iii) verdicts for ours1 and ours1pc
56 shapes (28 per dtype), 3 interleaved reps each, about 1 h per GPU on the RTX 6000 Ada (the M <= 32 shapes dominate: ~4 min each because of the oracle search). Resumable. Two GPUs: run the same command twice with `--gpu i --shard i/2 --tag fresh_gemma_<gpu>_s<i>` for i = 0, 1.
Method names in `--methods`: `ind_auto` stock Inductor max-autotune (ATEN+TRITON); `ours1` plain top-1; `ours1pc` top-1 with **Amendment 5d** (BLOCK_M cap + padded-row charge); `ours1tb` top-1 with the 5b tie-break; **`ours5d` is the Amendment 5c distinct-tile top-5, not 5d** (the name predates 5d); `tritonblas` the tritonBLAS-style baseline; `oracle` the restricted per-M oracle (only at the M values in `--oracle-ms`).
Turning 5d on outside this harness: environment variable `LALA_TRITON_PADCAP=1` (default off), or `select_scored(..., padcap=True)` for one call. `ours1pc` sets it for you; do not set the env var for this run (it would also change `ours1`, `ours1tb`, `ours5d` and the oracle pools).
A profile made by the existing calibrate.py works unchanged (5d did not change the profile format or calibration); sanity check: `python3 -c "import lala_triton.profile as p; q=p.load('profiles/<gpu>.json'); print(q['n_sm'], q['l2_eff']/2**20, {k: sum(t.startswith('16x') for t in m) for k, m in q['meta'].items()})"` should print non-zero BM=16 tile counts for `bf16|ABT` and `fp8|ABT` (5d raises an error if none exist).

## Repository layout
`lala_triton/` package, `tests/`, `eval/` harness (run_eval, tuning_time, e2e_layer, analyze_*, plot_*), `profiles/` (RTX 6000 Ada profile), `results/` + `plots/` (summary CSVs/plots only),
`PREREGISTRATION.md` (thresholds written before data), `REPORT_5.md` (results and honest caveats), `legacy/` (round 1-3 one-off scripts and reports, kept for provenance; not maintained, and they hardcode the original machine's paths).
Opt-in tie-break (Amendment 5b, off by default because it failed on fp8): `LALA_TRITON_TIEBREAK=1`. Opt-in distinct-tile top-k (Amendment 5c, off by default because it failed on fp8): `LALA_TRITON_DISTINCT=1`. Opt-in BLOCK_M cap + padded-row charge (Amendment 5d, off by default until validated): `LALA_TRITON_PADCAP=1`. Ablation switch: `LALA_TRITON_TRAFFIC=naive`.
A tritonBLAS-style baseline (my transcription of arXiv 2512.04226, not the authors' code) is in `lala_triton/baselines/tritonblas.py`; run it with `run_eval.py --methods tritonblas`.

## Known limitations
- `enable()` turns off `torch._inductor.config.fx_graph_cache` (its cache key ignores our patch; a hit would silently use stock configs). Re-enable it yourself after `disable()` if you want it.
- The Inductor config hook can't see B's layout: bf16 uses `bf16_layout` (default 'AB'); fp8 is always 'ABT'. fp16 is treated as bf16.
- Triton 3.2 / torch 2.6 only (hooks patch `torch._inductor.kernel.mm{,_scaled}`). Triton fp8 GEMMs on sm_89 accumulate at reduced precision (round 3; unrelated to selection).
- eff is calibrated on one shape (8192^3, G=8); small-M accuracy rests on the fitted latency term (see REPORT_5.md for its measured error).
- The kernel used for calibration is our transcription of the Inductor template, not Inductor's own compiled kernel (round-3 check not repeated).
- Calibration ran on a power-capped GPU; results on uncapped/HBM parts are unknown, and L2 partitioning (A100/H100) is untested.
