# Round 5: turn the model into a tool Dongyoon can use

Context: `~/lala-triton-killtest/` holds three rounds of results (`REPORT.md`, `REPORT_2.md`, `REPORT_3.md`, plus `REPORT_4.md` if round 4 ran).
- The traffic model (M3) is accurate.
- The time model is the weak link:
  - It fails at small M (M ≤ 32), where it picks tiny tiles.
  - fp8 top 1 is 1.46× slower than Inductor, because eff was calibrated in short bursts.
- Inside Inductor's templates, M3 top 5 ties Inductor max autotune on kernel runtime, with 2 to 7× less tuning time.

**Goal for this round:** a small installable package that:
- picks Triton GEMM configs with no (or a little) benchmarking;
- can be calibrated on a new GPU in under an hour;
- plugs into Inductor and plain Triton.

A teammate (Dongyoon) will use it on other GPUs to evaluate real LLM workloads. Presentation is Dec 3, so favor working and honest over ambitious.

**Rules:**
- Explain changes and results in plain language.
- Don't commit to any public repo.
- Run long jobs under `nohup` with logs.
- Keep CSVs resumable.
- Write thresholds into `PREREGISTRATION.md` (Amendment 5) before collecting new data.

Budget: about 1.5 days. Steps in priority order. If you run over, drop from the bottom and say so.

---

## Step 1: Package (~3 h)

Create `tritonsel/` as an installable package (`pip install -e .`) with:

| Module | Contents |
|---|---|
| `tritonsel/model.py` | M3 traffic model, the time model, and ranking. Pure Python, no GPU needed. |
| `tritonsel/profile.py` | GPU profile load/save. A profile is JSON: name, SMs, L2_eff, HBM_BW, L2_BW, per-family eff tables, and the small-M terms from Step 2. |
| `tritonsel/calibrate.py` | `python -m tritonsel.calibrate --out profiles/<gpu>.json`. Runs the existing microbenchmarks: L2_eff double read, bandwidths, and sustained-load eff on the calibration shapes. Must finish in under 1 h on one GPU and print progress. |
| `tritonsel/select.py` | `select(M, N, K, dtype, layout, profile, k=1)` returns ranked configs in Inductor template format. |
| `tritonsel/inductor.py` | One call, `tritonsel.inductor.enable(profile, k=1)`, that patches Inductor's mm and scaled_mm config generation to use our top k. Reuse round 3's `inductor_patch.py`. Has a `disable()`. |
| `tritonsel/triton_prune.py` | A `prune_configs_by` hook for plain `@triton.autotune` kernels. |
| `tritonsel/table.py` | `python -m tritonsel.table --model llama3-8b --profile ... --out table.json`. Writes a per-M config table (M = 1 to 8192) for every GEMM shape in a model, in a vLLM-style JSON layout. |

Also include:
- A `README.md` for Dongyoon: install, calibrate, use with `torch.compile`, generate tables, and known limitations.
- Unit tests for the model (reuse round 1's).
- A smoke test that compiles and runs one GEMM through each integration.
- Profiles for RTX 6000 Ada (`profiles/rtx6000ada.json`), built from existing calibration data where possible.

## Step 2: Fix the two known failures (~4 h)

1. **Sustained eff.** Recalibrate eff per template family and layout under sustained load, using the round-2 sustained protocol on the calibration shapes only. The round-3 data on disk can seed it.
2. **Small M.**
   - Add an attainable-bandwidth term: bytes in flight per SM (occupancy × stages × tile bytes per stage) against what is needed to saturate DRAM (Little's law). Fit the one latency constant on calibration shapes only.
   - Add a split-K option only if Inductor's template supports it on this torch version. Check first and say so.
3. **Validation.** Validate on held-out shapes, not the calibration set. Pre-register:
   - top 1 runtime ratio vs Inductor max autotune ≤ 1.03 (geomean) for bf16 and fp8;
   - no single shape worse than 1.15;
   - M ≤ 32 shapes within 1.10 of the per-M oracle.

   Report honestly whether each holds.

## Step 3: tritonBLAS-style baseline (~2 h)

Implement tritonBLAS's published selection logic (cache hit-rate term, wave-tail term, square cache tile) as `tritonsel/baselines/tritonblas.py`, using the same template space and profile. Add it to the comparison, and note any part of their model you had to approximate.

## Step 4: Evaluation harness for Dongyoon (~3 h)

Create `eval/run_eval.py --profile ... --models llama3-8b,llama3-70b --dtypes bf16,fp8 --out results/`.

**Shapes:** Llama 3 8B and 70B GEMMs (QKV, O, gate/up, down; per-GPU shapes for 70B as one tensor-parallel shard if it doesn't fit), M ∈ {1, 8, 32, 128, 512, 2048, 8192}.

**Methods:**
- ours top 1 and top 5
- Inductor max autotune
- Inductor default (no autotune)
- cuBLAS via `torch.matmul`, with `allow_bf16_reduced_precision_reduction=False`
- nvMatmulHeuristics, if installed
- the tritonBLAS-style baseline
- an oracle over the extended space, on a subset

**Per run, record:**
- kernel runtime (sustained protocol, clocks recorded)
- tuning time, split into selection, compile and benchmarking
- predicted vs measured runtime
- predicted vs ncu DRAM bytes, on a subset if ncu is available

**End-to-end:** a Llama 3 8B decoder layer under `torch.compile` with (a) max autotune, (b) ours, (c) default. Measure:
- prefill latency at 512 and 2048 tokens;
- decode latency per token at batch 1, 8, 32;
- total compile/tuning time.

**Outputs:**
- `results/summary.md` with tables
- plots: runtime ratio per shape, tuning time, and dynamic M curves

## Step 5: Report (~30 min)

Write `REPORT_5.md` covering:
- what works;
- the numbers vs each baseline;
- what Dongyoon needs to do on a new GPU;
- known limitations;
- one honest headline sentence for the Dec 3 talk.
