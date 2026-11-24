# Round 5: `lala_triton`, a reimplemented selector packaged for Dongyoon

> **Naming note (2026-11-20):** the Python package was renamed `tritonsel` -> `lala_triton` (repo `lala-triton`) (commands in this report use the new names); env vars `TRITONSEL_*` -> `LALA_TRITON_*`. 


Run 2026-11-19 on the two RTX 6000 Ada GPUs (300 W cap; every timed run sat at the cap), torch 2.6.0+cu124, Triton 3.2.0.
Thresholds: `PREREGISTRATION.md` **Amendment 5 / 5a**, written before any round-5 data. All timings use the sustained protocol (DRAM-cold rotating buffers, CUDA-graph replay, NVML clocks recorded, 3 interleaved reps, median).

## Read this first: what this is and is not
1. **It is a reimplementation.** The round 1-3 model code, calibration data, results CSVs and Amendments 3/3b/4 were missing from the working directory, and there is no `REPORT_4.md`. I rebuilt M3 and the time model from `triton-locality-killtest.md` Section 2, REPORT 1-3 and the surviving preregistration text. It is not bit-identical to the old code, and nothing here is "the old model, reproduced".
2. **The ncu baseline gate (R5-0) has NOT run.** ncu is denied (`ERR_NVGPUCTRPERM`, driver 580.196.07, `RmProfilingAdminOnly: 1`; no sudo). So the reimplemented traffic model has **not** been checked against ncu DRAM bytes (the old 5.4% median error). Everything below that depends on the traffic model being right is unverified at that level; only end-to-end runtime results are measured. ECN was asked to enable counters.
3. **L2_eff is not measured.** It is the round-1 constant (102 MiB; no misses to 88 MiB, knee about 102 MiB), labelled as such in the profile. My timing-based knee check (secondary, no threshold) gives 55-58 MiB, which disagrees with 102 MiB. The probe is crude (a hit still costs L2 time), so I did not tune anything to it, but the disagreement is unresolved.
4. **One validation run was voided and redone.** The first run had a harness bug (dynamo's cache is keyed on a lambda's shared code object, so several Inductor methods silently ran the same kernel), giving a fake "everything ties" result. Kept as `results/validation_INVALID_*`; all numbers below are from the fixed harness (each method's CUDA graph is captured right after compiling).

## Verdicts (pre-registered R5-V, held-out = Llama 3 8B GEMMs, 32 shapes per dtype, none used in calibration)
| criterion | bf16 | fp8 |
|---|---|---|
| (i) geomean ours top-1 / Inductor max-autotune <= 1.03 | **0.958 PASS** | **1.026 PASS** |
| (ii) no shape worse than 1.15 | worst 1.083 (`gate_up`, M=32) **PASS** | worst 1.834 (`gate_up`, M=1) **FAIL** |
| (iii) M <= 32: within 1.10 of the per-M oracle on every shape | worst 1.082, 16/16 **PASS** | worst 1.812, 11/16 (69%) **FAIL** |

**fp8 fails (ii) and (iii).** The failures are 5 shapes, not diffuse noise: `gate_up` (N=28672, K=4096) at M = 1, 8, 16, 32 (1.83, 1.25, 1.32, 1.25 vs Inductor) and `qkv` at M=32 (1.21). The other 11 M <= 32 fp8 shapes tie the oracle within 3%.

**Documented fallback (benchmark top 5 for M <= 32), reported next to top-1:** it changes almost nothing. Fallback geomean vs Inductor: bf16 0.9585, fp8 1.0252; fp8 M <= 32 worst vs oracle still 1.81, 69% within 1.10. Reason: the model's top 5 are the same tile at G in {8,16,32,4,1}, so there is nothing different for the benchmark to try.

**Diagnosis (post hoc, on the shapes it failed on):** in the DRAM-bound small-M regime the model predicts many tiles at exactly the DRAM roofline, so the ranking is a tie broken by enumeration order. The first tied tile (128x256x64, 8 warps) wastes most of its work at M=1 and measured 0.250 ms vs the oracle's 0.138 ms (the model predicted 0.151 ms for it). A tie-break that prefers tiles with less wasted compute/bytes would very likely fix this, **but I did not apply it**: the held-out shapes are already seen, and the pre-registration allows refits only on calibration shapes. Doing it honestly needs a new amendment and fresh held-out shapes (e.g. Dongyoon's larger ones). Per your stop rule, this is the pre-registered failure to decide on.

## The numbers vs each baseline (bf16 prefers `ABT` weights, as in `nn.Linear`; sustained, medians)
| | bf16 | fp8 |
|---|---|---|
| ours top-1 / Inductor max-autotune, geomean | 0.958 | 1.026 |
| ours top-5 / Inductor max-autotune, geomean | 0.954 | 1.022 |
| ours top-1 / cuBLAS (`torch.mm`, reduced-precision reduction off), geomean | 0.890 | 0.945 |
| shapes within 3% of Inductor max-autotune | 97% | 66% |
| shapes >= 5% faster than Inductor max-autotune | 22% (7/32) | 19% (6/32) |
| predicted vs measured, ours top-1 (median abs log error; 189 rows) | 0.108 (about 11%), over-predicts | 0.133 (about 13%) |

- Where ours is >= 5% faster than stock max-autotune: bf16 `qkv`/`gate_up` M=128-512 (up to 0.71), `down` M=128; fp8 `o` M=128/8192, `qkv` M=128/8192, `down` M=128. These are shapes where Inductor's autotuner picks ATen/cuBLAS or a stock tile under burst timing and the sustained ranking differs.
- Inductor "default" (no autotune) is ATen, i.e. cuBLAS, so it is the same row as cuBLAS (ours/default 0.890 bf16, 0.946 fp8). nvMatmulHeuristics was **not run** (see Not done). The tritonBLAS-style baseline was run later (F7).
- Calibration facts (reimplemented profile `profiles/rtx6000ada.json`): HBM 797 GB/s, L2 about 4.0 TB/s, eff tables for 351 (bf16 AB), 347 (bf16 ABT), 383 (fp8 ABT) feasible tiles, measured on 8192^3 under the sustained protocol (eff 0.04-0.47; the power cap makes them far below 1). The Little's-law latency was fit on calibration small-M shapes only: bf16 2.96 us, fp8 2.41 us. It is a weak fit: RMS log error on those shapes 0.45 (bf16) and 0.56 (fp8), versus 0.69 / 0.78 without the term. So the small-M term helps but is far from exact.
- Split-K was **not added** (the template check was not done; stated rather than assumed).

## Tuning cost (first call in a fresh process with fresh Triton and Inductor caches; median seconds per GEMM; 48 runs)
| | select | compile | benchmark | total | vs stock max-autotune |
|---|---|---|---|---|---|
| bf16 ours top-1 | 0.03 | 2.3 | 0 | 2.3 | 0.36x |
| bf16 ours top-5 | 0.03 | 3.2 | 0.6 | 3.9 | 0.59x |
| bf16 stock max-autotune | 0 | 4.3 | 2.2 | 6.5 | 1x |
| fp8 ours top-1 | 0.03 | 2.3 | 0 | 2.3 | **0.095x** |
| fp8 ours top-5 | 0.03 | 3.4 | 0.6 | 4.1 | 0.165x |
| fp8 stock max-autotune | 0 | 13.4 | 11.6 | 24.8 | 1x |

Selection itself is about 30-40 ms; most of "ours" is Inductor's fixed compile cost.

## Llama 3 8B decoder layer under `torch.compile`, bf16 (one rep per cell, ms per layer, sustained)
RoPE and the KV-cache update are omitted; attention is SDPA with GQA; decode uses a 1024-token KV cache. These cells are **single reps**, and SM clocks differed between methods, so treat differences under a few percent as noise.

| case | stock max-autotune | ours top-1 | default |
|---|---|---|---|
| prefill 512 | 1.988 | 1.978 | 2.473 |
| prefill 2048 | 7.608 | 7.134 | 9.453 |
| decode b=1 | 0.524 | 0.525 | 0.529 |
| decode b=8 | 0.827 | 0.827 | 0.819 |
| decode b=32 | 1.936 | 1.985 | 1.995 |

Total compile/tuning time for the 5 shapes (fresh caches): stock max-autotune **93.8 s** (34.3 s benchmarking), ours top-1 **24.2 s** (0 benchmarking, 0.09 s selection), default 4.9 s. Layer latency: ours is about equal to max-autotune at decode and 512-token prefill, 6% lower at 2048-token prefill (clocks 1335 vs 983 MHz, single rep, so not claimed), 2.5% higher at decode b=32 (also within what I would call noise). Both beat default on prefill (ours 20-25% lower).

## What Dongyoon needs to do on a new GPU (see `README.md`; tested from a fresh copy of the package in a clean venv)
1. `pip install -e . --no-deps` (torch 2.6 / Triton 3.2); `python3 -m pytest -p no:anyio tests` (183 CPU tests).
2. Calibrate: `python3 -m lala_triton.calibrate --out profiles/<gpu>.json --gpu N --shard i/n` per GPU, then once with `--build-only`. It took 42 min on 2 GPUs here (about 80 min estimated on one). **Pass `--l2-eff-mib`** for the new GPU: the default is the RTX 6000 Ada round-1 constant and will be wrong elsewhere. If ncu counters work, check it with the double-read probe (`lala_triton/_l2probe.py`).
3. `lala_triton.inductor.enable(profile, k=1, bf16_layout='ABT')`, set `max_autotune_gemm_backends='TRITON'`, `torch.compile(..., mode='max-autotune-no-cudagraphs')`. (`enable()` turns off Inductor's FX-graph cache; otherwise a cache hit silently uses stock configs.)
4. `python3 eval/run_eval.py --profile ... --models llama3-8b,llama3-70b --dtypes bf16,fp8 --out results/`, `eval/tuning_time.py`, `eval/e2e_layer.py`, then `eval/make_summary.py`. Validation: `eval/analyze_validation.py`.
5. Smoke test of every integration: `python3 tests/smoke_integrations.py profiles/<gpu>.json`.
Verified in the clean copy: install, tests, table writer (vLLM-style files), quick calibration plus profile build, smoke test (Inductor and `prune_configs_by`), and a `run_eval` row.

## Known limitations
- Reimplementation; traffic model not ncu-validated (gate R5-0 pending); L2_eff is a constant, and my timing probe disagrees with it.
- **fp8 small-M `gate_up` (and `qkv` at M=32) fail the pre-registered checks** because of the tie-break described above. bf16 passes everything, but only on Llama 3 8B shapes on one GPU type.
- Small-M latency term fit is crude (RMS log error about 0.5 on calibration).
- eff is calibrated on one shape (8192^3, G=8); the calibration and oracle harness use our transcription of Inductor's template, not Inductor's compiled kernel (the round-3 equivalence check was not repeated). The Inductor numbers (ours, max-autotune, default) are all through real Inductor.
- The "oracle" is restricted (model top-20 U 100 random tiles, G=8, sustained), and exists only for M <= 32.
- Inductor's config hook cannot see B's layout (bf16 layout is a parameter); fp16 treated as bf16; Triton 3.2 / torch 2.6 only.
- Power-capped GPU, one device type: results may not transfer to uncapped or HBM parts. Triton fp8 accumulation on sm_89 is reduced precision (round 3), so fp8 comparisons are speed-only.

## Not done
Still not done: the ncu predicted-vs-measured subset (gate R5-0, blocked by counter permissions), nvMatmulHeuristics (not wired in), the extended-oracle comparison for M > 32, and the "disable" path of `inductor` and the table writer are untested on a second GPU. The `qkv` fp8 mid-M failures (F3, F6) are undiagnosed.
Done after the first version of this report (see the follow-ups): Llama 3 70B TP=8 shapes (F3), the tritonBLAS-style baseline (F7, my transcription, not the authors' code), dynamic-M plots (F4; `plots/r5_dynamic_M_*.png`), the naive-traffic ablation (F1), L2_eff sensitivity (F2), energy (F5), and the distinct-tile top-k amendment (F6). `results/summary.md` and the plots `plots/r5_ratio_per_shape.png`, `plots/r5_tuning_time.png` were generated earlier.

## One honest headline sentence for Dec 3
"On an RTX 6000 Ada, a zero-benchmark model-picked Triton config matches Inductor max-autotune on bf16 across three shape sets (Llama 3 8B 0.96x geomean, worst 1.08x; fresh Llama 3 70B TP=8 shards 0.97x, worst 1.10x; fresh Qwen 2.5 7B 0.96x, worst 1.05x) at about 0.36x its tuning time, and beats a tritonBLAS-style analytical baseline in bf16 by 10-20% (my transcription, not the authors' code); on fp8 it is only average-level (1.03x on 8B, 1.08x on 70B shards) at 0.1x the tuning time and still picks up to 1.8x slower on small-batch shapes, three pre-registered fixes (tie-break 5b, distinct top-k 5c, BLOCK_M cap 5d) each failed the fp8 criteria on fresh shapes and stay off (5d fixes the BM=128/256 picks but leaves BM=16 picks that tie on BN, F8), in fp8 the tritonBLAS-style baseline is comparable with different failures, and the traffic model underneath has not been re-validated against hardware counters."

---
# Follow-ups (requested 2026-11-20; updated as each result comes in)

### F1b. Interleaved rerun of the five worst bf16 naive shapes (same process, 5 reps, ours-M3 / ours-naive / stock paired): DONE
The 2.12x vs 1.08x comparison above came from two different runs, so I reran the worst shape and the next four with M3, naive and stock measured interleaved in the same run (`results/naive_interleaved*.csv`; ratios are medians over 5 reps).

| shape (bf16) | M3 / stock | naive / stock | naive / M3 |
|---|---|---|---|
| `gate_up` M=32 | 1.065 | **2.079** | 1.95 |
| `gate_up` M=16 | 1.001 | 1.102 | 1.10 |
| `o` M=512 | 0.993 | 1.025 | 1.03 |
| `o` M=32 | 1.028 | 1.032 | 1.00 |
| `down` M=512 | 1.023 | 1.017 | 0.99 |

The big gap holds up without clock differences: on `gate_up` M=32, naive picks 64x128x32 (1 stage) and is 1.95x slower than M3's 32x32x64 (1 stage), same process and same-run stock reference. On the other four shapes the difference is within 3% except `gate_up` M=16 (10%). So the L2/traffic model matters mainly where the naive formula mis-ranks a small-M shape, which is a few shapes, not a broad average gain. (Note: SM clocks within a run still differ between kernels, e.g. 1425 vs 2385 MHz on `down` M=512, because different kernels hit the power cap differently; that is part of the sustained behaviour being measured.)

## F2. L2_eff sensitivity, selection part (offline, no GPU): DONE
Same time model, L2_eff = 58 MiB vs 102 MiB, top-1 pick per shape.
- Llama 3 8B validation shapes (32 per dtype, 64 total): **0 of 64 top-1 picks change.**
- Wider grid (8B, 70B TP=1, 70B TP=8; 14 M values 1..8192; bf16+fp8; 336 shapes): **1 of 336 changes** (70B `gate_up` fp8 M=128).
- Meaning: on these shapes the choice of L2_eff in 58-102 MiB hardly matters to the ranking, so the unresolved 55-58 vs 102 MiB disagreement does not threaten the validation numbers. It could still matter for much larger shapes or a GPU with a different L2. (Timing of the changed pick not done; it is outside the validation set.)
- **Timing-probe repeat (3 more runs: 2 on GPU 0, 1 on GPU 1, fresh process each): the knee is stable.** Sequential 57.4 / 57.1 / 57.1 MiB (original 57.7), strided 54.6 / 54.6 / 54.1 MiB (original 54.6); HBM 797 GB/s and L2 4.0-4.2 TB/s the same on both GPUs. So 55-58 MiB is a reproducible property of this probe, not noise, and the disagreement with the round-1 102 MiB is real. What it does not settle: the probe measures time, and its curve rises smoothly (second-pass extra cost 0.30 at 48 MiB, 0.52 at 64, 0.78 at 88, 0.86 at 104 MiB) instead of showing a sharp knee, so a hit still costs measurable time and the "halfway" point is a convention. Round 1 counted DRAM misses (counter-based) and saw none up to 88 MiB. Both can be true; only the ncu probe (blocked) can tell. Per F2 above, the choice does not change selections on these shapes.

## F1. Ablation: naive per-tile traffic instead of M3 (time model otherwise identical): DONE
Naive traffic = M*N*K*s*(1/BM+1/BN) (no L2 reuse), floored at compulsory bytes (env `LALA_TRITON_TRAFFIC=naive`). Same 8B validation shapes, ours-top-1 through real Inductor, each ratio is vs stock max-autotune. Naive was timed in its own run (paired with stock max-autotune inside that run); the M3 numbers are the earlier validation run, so clocks differ between the two (script `eval/analyze_ablation.py`, `results/ablation_naive*.{json,csv}`).

| | bf16 M3 | bf16 naive | fp8 M3 | fp8 naive |
|---|---|---|---|---|
| geomean vs stock max-autotune | 0.958 | 0.971 | 1.026 | 1.010 |
| worst shape | 1.083 | **2.12** (`gate_up` M=32) | 1.834 | 1.824 (`gate_up` M=1) |
| shapes within 3% of stock | 97% | 88% | 66% | 69% |
| picks that differ from M3 | | 26 of 32 | | 21 of 32 |

Plain reading: **the L2 model buys robustness for bf16, not average speed.** Without it bf16 loses 1.3 points of geomean but gets one shape 2.1x slower; M3 keeps the worst case to 1.08. For fp8 the naive model is not worse (geomean 1.010 vs 1.026 is within what I would call run-to-run noise, and both have the same 1.8x failure on `gate_up` M=1, which is the tie problem, not a traffic problem). So I cannot claim M3 is responsible for the fp8 result. Caveats: single run per variant; different clocks between the two runs; the count of differing picks (26/21) is by string compare and is a bit higher than the offline count (22/21) so some of it may be formatting rather than a different tile.

## F3. fp8 tie-break (Amendment 5b): fresh-set result is a FAIL for fp8, so it stays OFF; bf16 unaffected
Fresh held-out set: Llama 3 70B TP=8 shards, 64 shapes (32 per dtype), 3 interleaved reps, each ratio vs stock max-autotune; oracle for M<=32 only (`results/fresh70b_tp8_*`, `eval/analyze_fresh.py`). Rule frozen in `PREREGISTRATION.md` Amendment 5b before the run.

| | bf16 plain | bf16 tie-break | fp8 plain | fp8 tie-break |
|---|---|---|---|---|
| (i) geomean vs stock (<= 1.03) | 0.967 PASS | 0.967 PASS | 1.078 **FAIL** | 1.051 **FAIL** |
| (ii) worst shape (<= 1.15) | 1.096 PASS | 1.095 PASS | 1.80 (`o` M=1) **FAIL** | 1.58 (`qkv` M=128) **FAIL** |
| (iii) M<=32 vs oracle (<= 1.10 on all) | 1.03 PASS | 1.05 PASS | 1.80 (11 of 16 ok) **FAIL** | 1.53 (13 of 16 ok) **FAIL** |
| picks changed by tie-break | | 11 of 32 | | 14 of 32 |

- **What the tie-break did:** on fp8 it helped where it was designed to: `gate_up` M=8/16/32 went from 1.07/1.07/1.23 to 1.00/1.00/1.07 (vs stock), `down` M=32 1.12 to 1.05, geomean 1.078 to 1.051, worst 1.80 to 1.58. On bf16 it is neutral (one near-tie shape slightly worse vs oracle, 1.03 to 1.05; all criteria still pass).
- **What it did not fix, i.e. why fp8 still fails:** (a) **M=1 on every GEMM**: `o` 1.53, `down` 1.27, `qkv` 1.32 (the oracle picks e.g. 16x32x128, 1 stage, 4 warps where we pick 16x32x128, 4 stages, 2 warps; so "more bytes in flight" is not the right preference at M=1: it picked a deeper pipeline and fewer warps than the oracle). (b) **`qkv` (N=1280) at M=128/2048/8192: 1.58/1.53/1.24**: these are not small-M and not DRAM ties; the model's pick is simply wrong there (likely wave quantization, since 1280 columns give few tiles). The tie-break does not touch these, and I have not diagnosed (b) further.
- **Decision rule applied:** Amendment 5b says the tie-break becomes default only if it passes (i)-(iii) on the fresh set; it does not for fp8, so it stays **off** (`LALA_TRITON_TIEBREAK=1` opts in). No second tweak on these shapes (needs Amendment 5c and another fresh set). bf16 passes everything on the fresh set with either selector, which is the stronger bf16 evidence: it now holds on two shape families (8B on 1 GPU shard, 70B TP=8).
- Net status of the headline: bf16 claim holds on 8B and on 70B TP=8; the fp8 claim does not (average 1.03 on 8B but 1.08 on 70B shards, worst 1.8x in both).


## F4. Dynamic-M plots (from existing data, no new timing): DONE
`eval/plot_dynamic_m.py` -> `plots/r5_dynamic_M_measured.png`, `plots/r5_dynamic_M_buckets_predicted.png`.
- **Measured (Llama 3 8B validation run):** time relative to the best measured method at each M (oracle where it exists, M<=32; otherwise the best of ours / stock max-autotune / cuBLAS, so the "best" is only as good as those three). Ours is flat at ~1.0 for bf16 at every M; stock max-autotune has spikes (up to 1.4x at `qkv`/`gate_up`/`down` around M=128-512); cuBLAS is up to 1.6x at large M. For fp8 the plot shows the known failure (`gate_up` at M<=32, up to 1.83x) and ours also spikes at M=512 (1.35x `qkv`, 1.17x `gate_up`).
- **Buckets are model-predicted, not timed** (a bucketed table cannot be timed from the data I already have: only 8 M values were measured). The prediction says a power-of-two bucket costs ~0% over exact per-M configs everywhere except `gate_up` at M of about 192 and 384 (up to 1.39x bf16, 1.25x fp8). Treat this as a hint to time those M values, not a result; much of the flatness may come from the model's many near-ties.

## F5a. Energy: kernel energy per GEMM, ours top-1 vs stock max-autotune: DONE
NVML total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, whole GPU) over the timed window divided by GEMMs in it; fresh run of both methods on the 64 Llama 3 8B validation shapes, 3 interleaved reps, median (`results/energy_val_s*.csv`, `eval/analyze_energy.py`, `results/energy_validation_per_shape.csv`). This run also replicates the runtime numbers: geomean time ratio 0.960 (bf16) and 1.025 (fp8) vs 0.958/1.026 in the first validation run.

| | bf16 | fp8 |
|---|---|---|
| geomean energy ratio ours / stock | **0.971** | **1.027** |
| geomean time ratio ours / stock | 0.960 | 1.025 |
| median power, ours / stock | 297 / 294 W | 296 / 297 W |
| worst-shape energy ratio | 1.10 (`qkv` M=8) | 1.71 (`gate_up` M=1; time ratio 1.83) |
| shapes where energy ratio differs from time ratio by >5% | 9 of 32 | 4 of 32 |

- As expected on a 300 W-capped card, power sits at 290-315 W for almost every kernel, so energy follows runtime: the geomeans agree to within about 1 point, and energy is at most ~10% off the runtime ratio on any shape.
- **Where it does not track (13 of 64 shapes beyond 5%):** all within -6% to +10% of the time ratio and in mixed directions (e.g. bf16 `down` M=16/32 time ratio 1.00 but energy 1.07; `down` M=8 and 512 energy 0.95 at equal time), i.e. the two methods run the same-speed kernels at slightly different power (268-313 W). My reading is that this is power-cap/clock jitter plus per-kernel power differences; I did not verify the cause. One notable case: fp8 `gate_up` M=1, where our slow pick draws less power than stock (291 vs 311 W), so energy is 1.71x rather than 1.83x. Nothing is cheaper in energy than in time by a meaningful margin, and no shape is much worse in energy than in time.

## F5b. Energy: GPU energy spent on tuning + compile, and decoder-layer energy: DONE
Same NVML counter, read around the first call in a fresh process with fresh caches (so it includes compile, benchmarking and the first run; **GPU only**: CPU energy during compile is not counted, which flatters every method that compiles a lot). `results/tuning_time_energy.csv` (64 runs: 16 per method) and `results/e2e_energy.json`. I also tried to subtract an idle baseline, but my idle readings were unreliable (28 to 135 W between processes, probably residual from the previous process), so the table gives **total** energy and mean power, not "energy above idle".

Per GEMM, median first call (Llama 3 8B shapes, M in {32, 2048}):
| | bf16 energy (J) | bf16 time (s) | fp8 energy (J) | fp8 time (s) |
|---|---|---|---|---|
| ours top-1 | 231 | 2.3 | 70 | 2.4 |
| ours top-5 | 308 | 3.8 | 129 | 4.0 |
| stock max-autotune | 401 | 6.5 | 771 | 25.0 |
| Inductor default | 225 | 2.1 | 53 | 1.9 |

So top-1 spends 0.58x (bf16) and **0.09x (fp8)** of stock max-autotune's tuning energy, about the same ratios as for time. Mean GPU power during tuning is only 54-139 W (the GPU is mostly waiting on CPU compile), so tuning energy is mostly time x low power, not heavy GPU work; that is why energy follows time here too.

Llama 3 8B decoder layer, bf16, five shapes, fresh caches per method (single run each; first-call energy summed over the 5 shapes):
| | tuning+compile time | GPU energy during tuning+compile | |
|---|---|---|---|
| stock max-autotune | 93.0 s | 17,055 J | 1x |
| ours top-1 | 23.1 s | 3,847 J | 0.23x |
| default | 4.5 s | 607 J | 0.04x |

Layer energy per forward pass, sustained (mJ; latency ms in brackets): prefill 512: stock 590 (2.02), ours 580 (1.97), default 723 (2.46); prefill 2048: stock 2439 (8.01), ours 2136 (7.11), default 2796 (9.41); decode b=1: 154 (0.523) / 159 (0.522) / 156; b=8: 240 / 250 / 251; b=32: 582 (1.96) / 581 (1.99) / 602. Energy tracks latency in every case (about 300 W throughout the prefill cases, 295-305 at 2048), the one place it does not is that the latency ordering and energy ordering swap by a few percent at decode b=1/8 (ours +3% energy at equal latency; within the power jitter seen in F5a). **The 2048-token prefill advantage (ours 11% faster, 12% lower energy than stock) reproduced in this second independent run** (the first run showed 6%), still one rep per run, and still with different SM clocks (1320 vs 960 MHz) which are part of what the sustained protocol measures but also mean I cannot separate "better tile" from "tile that draws less power per FLOP".


## F6. Amendment 5c: top-k = k distinct tiles, validated on fresh Qwen 2.5 7B shapes: **FAILS the pre-registered fp8 criteria; stays OFF**
Rule and thresholds were written and committed before any data on these shapes (PREREGISTRATION.md, Amendment 5c, commit `27c318e`). Fresh set: Qwen 2.5 7B linear layers (qkv/o/gate_up/down), M in {1, 8, 32, 128, 512, 2048, 8192}, 28 shapes per dtype, never used for calibration or for choosing the rule. Medians of 3 interleaved reps; ratios are vs real Inductor max-autotune (ATEN,TRITON) from the same run; the per-M oracle (M <= 32 only) is the restricted one (model top-20 + 5 distinct + 100 random tiles). Script: `eval/analyze_5c.py`; data `results/fresh_qwen_s{0,1}.csv`.

| fp8 (28 shapes) | geomean vs stock (i, <=1.03) | worst vs stock (ii, <=1.15) | M<=32 worst vs oracle (iii, <=1.10) | verdict |
|---|---|---|---|---|
| top-1 | 1.035 | 1.53 (gate_up M=1) | 1.53 | fail i, ii, iii |
| top-5 old (variants of one tile) | 1.020 | 1.53 (gate_up M=1) | 1.53 | pass i; fail ii, iii |
| **top-5 distinct (5c)** | **0.994** | **1.34 (qkv M=512)** | **1.26 (o M=1)** | pass i; **fail ii, iii** |

bf16 regression check (same criteria): top-1 0.963 / worst 1.04 / M<=32 worst 1.05; old top-5 0.958 / 1.04 / 1.04; distinct top-5 0.955 / 1.05 / 1.04. All pass, so 5c does not hurt bf16.

**What 5c did.** It fixed the specific failure it was written for: fp8 gate_up M=1 goes from 1.53x slower than stock to 1.00x (the old top-5 stayed at 1.53 because all five candidates were the same tile). gate_up M=32 also improves 1.17 -> 1.04 (the old top-5 got this one too). Overall fp8 geomean is 0.994, i.e. slightly faster than stock.
**What still fails** (two shapes, so the pre-registered "all shapes" criteria fail): (a) fp8 `o` M=1 is 1.24x vs stock and 1.26x vs the oracle, so a tile exists that none of our five distinct candidates is close to; (b) fp8 `qkv` M=512 is 1.34x (top-1: 1.52). M=512 is outside the oracle range, so I do not know how fast the best tile is there. This is the same `qkv` fp8 mid-M weakness seen on the 70B shards (F3) and it remains undiagnosed.
**Decision (per the pre-registered rule):** `LALA_TRITON_DISTINCT` stays default OFF. Not tuned further on these shapes; any change needs Amendment 5d and another fresh set. The result is nevertheless the best fp8 outcome so far (worst-case 1.81 -> 1.34 across sets, though these are different shapes, so this is not a like-for-like improvement claim).
**Caveats.** One fresh model, one GPU type; 28 shapes per dtype and 3 reps; shape 'o' M=1 is a single shape deciding criterion (iii). Benchmarking cost (median first call, one shape per process, fresh caches): fp8 stock 23.6 s, top-1 0.33 s, old top-5 1.63 s, distinct top-5 1.63 s; bf16 stock 4.7 s, top-1 0.32 s, top-5 1.56 s, distinct 1.98 s. Candidates that are distinct tiles cost slightly more to compile than G-variants in bf16 and the same in fp8.

## F7. tritonBLAS-style analytical baseline vs ours vs stock max-autotune
**What this is, and is not.** `lala_triton/baselines/tritonblas.py` is *my transcription* of the tritonBLAS analytical model (arXiv 2512.04226: L_MT, waves, hit rate, cache-tile selection, L_mem/L_tile/L_total). The paper leaves many details unspecified; every gap-fill is listed in the module docstring. It is **not the authors' code**, it picks from our profile's hardware numbers, and its tile is launched through our harness kernel (the `oracle` launcher), not real Inductor. Treat it as "a tritonBLAS-style selector", and note a faithful implementation might do better. It is a baseline for comparison, not a verdict on tritonBLAS itself. Same shape grid as before (Llama 3 8B, validation shapes already seen; Qwen 2.5 7B), 28 shapes per model/dtype, 3 reps medians, sustained protocol, interleaved with ours top-1 and stock max-autotune; `results/tblas_s{0,1}.csv`, `results/tblas_per_shape.csv`. Ratios are vs stock (lower is better).

| model, dtype | tritonBLAS-style geomean / worst | ours top-1 geomean / worst | tritonBLAS-style / ours (geomean) | M<=32 geomean: tB / ours |
|---|---|---|---|---|
| Llama 3 8B bf16 | 1.051 / 1.46 (o M=128) | 0.953 / 1.09 | 1.103 | 1.133 / 1.011 |
| Llama 3 8B fp8 | 1.027 / 1.30 (o M=128) | 1.023 / 1.83 | 1.003 | 1.055 / 1.112 |
| Qwen 2.5 7B bf16 | 1.156 / 2.01 (gate_up M=32) | 0.964 / 1.05 | 1.199 | 1.301 / 1.008 |
| Qwen 2.5 7B fp8 | 1.115 / 1.94 (gate_up M=32) | 1.042 / 1.53 | 1.070 | 1.163 / 1.075 |

Shapes where ours is >5% faster than the tritonBLAS-style pick / where it is >5% slower: Llama bf16 14 / 1, Llama fp8 12 / 6, Qwen bf16 19 / 0, Qwen fp8 14 / 5.

**Reading.** In bf16 our selector beats this baseline clearly and almost everywhere (it is 10-20% faster in geomean, and the baseline's worst cases are 1.5-2.0x vs our 1.05-1.09). In fp8 the two are close on average (ours is ahead on Qwen, tied on Llama), but they fail on different shapes: the baseline is much better than ours on the fp8 shapes where ours has its worst failures (e.g. Llama fp8 worst 1.30 vs our 1.83), and worse elsewhere (gate_up M=32 at 1.94). So in fp8 the baseline's worst case is not clearly worse than ours; the honest claim is "ours is better in bf16, comparable in fp8, with complementary failures", not "ours beats tritonBLAS". The M<=32 fp8 geomean even favours the baseline on Llama (1.055 vs 1.112). Caveats: my reimplementation, our harness kernel for launching it, models and shapes partly seen during development (Llama), single GPU.

## F8. Amendment 5d: BLOCK_M cap + padded-row charge, validated on fresh Gemma 2 9B shapes: **FAILS the pre-registered fp8 criteria (ii) and (iii); stays OFF. bf16 passes and is unchanged.**
Rule and thresholds were written and committed before any timing on these shapes (PREREGISTRATION.md Amendment 5d, commit `e87ff47`). Motivation: Dongyoon's L40S run (relayed to me; I have not seen the data and used none of it) where the fp8 small-M tie picked BLOCK_M=128 at M=1 and 8, 7.5x slower on `o` M=1. Fresh set: Gemma 2 9B linear layers (qkv (8192,3584), o (3584,4096), gate_up (28672,3584), down (3584,14336)), M in {1, 8, 32, 128, 512, 2048, 8192}, 28 shapes per dtype, 56 total; Mistral 7B was again not usable (same GEMMs as Llama 3 8B). Medians of 3 interleaved reps on both GPUs, ratios vs real Inductor max-autotune from the same run, per-M oracle for M <= 32 (12 shapes per dtype; pool = plain top-20, 5b top-5, 5c distinct-5, 5d top-20 and 100 random tiles). All 56 shapes completed with no launcher or measurement failures. Script `eval/analyze_5d.py`; data `results/fresh_gemma_s{0,1}.csv`, `results/fresh_gemma_per_shape.csv`, `results/fresh_gemma_summary.json`; untimed pick diagnostic `eval/offline_5d_picks.py`.

| | bf16 plain | bf16 5d | fp8 plain | fp8 5d |
|---|---|---|---|---|
| (i) geomean vs stock (<= 1.03) | 0.970 PASS | 0.969 PASS | 1.056 FAIL | 1.019 PASS |
| (ii) worst shape (<= 1.15) | 1.055 PASS | 1.054 PASS | 1.92 (`gate_up` M=1) FAIL | **1.54 (`gate_up` M=1) FAIL** |
| (iii) M <= 32 vs oracle (<= 1.10 on all) | 1.059 PASS | 1.058 PASS | 1.89, 7 of 12 ok FAIL | **1.52, 9 of 12 ok FAIL** |
| picks changed by 5d | | 0 of 28 | | 5 of 28 |

Decision (pre-registered): 5d needs both dtypes to pass (i)-(iii); fp8 fails (ii) and (iii), so `LALA_TRITON_PADCAP` stays default OFF. No second tweak on these shapes (a further change needs Amendment 5e and another fresh set).

- **What 5d did.** It changed exactly the 5 fp8 picks whose plain BM exceeded the cap, and all 5 got faster vs stock: `gate_up` M=1/8/32 from 1.92/1.33/1.27 to 1.54/1.02/1.14 (plain picked 128x256x64, 8 warps at every one of them), `qkv` M=1/8 from 1.50/1.02 to 1.02/1.00. fp8 geomean 1.056 to 1.019, so (i) now passes. On the 23 fp8 shapes with an unchanged pick the 5d and plain ratios agree to within about 2 points (largest `o` M=2048, 0.948 vs 0.968, same pick), which is my read of the run-to-run noise. bf16 is untouched (0 of 28 picks change, geomean 0.970 vs 0.969, worst 1.055 vs 1.054), so there is no bf16 regression.
- **Why fp8 still fails: the three remaining bad shapes already have BM=16, so the cap cannot touch them.** `gate_up` M=1 (1.54 vs stock, 1.52 vs oracle), `o` M=1 (1.24 / 1.25), `gate_up` M=32 (1.14 / 1.13, passes (ii) by 0.01 and fails (iii)). In all three the oracle picks a wider tile (`gate_up` M=1: 16x128x64, 4 warps, 2 stages; `o` M=1: 16x128x128, 4 warps, 4 stages; `gate_up` M=32: 32x32x128, 2 warps, 4 stages) while we pick 16x32x128. For `gate_up` M=1 the model scores our pick and the oracle's pick at exactly the same 0.1329 ms, i.e. the same enumeration-order DRAM-roofline tie as 5b, now along BN instead of BM; measured 0.181 ms for ours vs 0.119 for the oracle (the model under-predicts our pick by 27%). So the tie problem is wider than BLOCK_M, and I did not try to fix it (no 5e).
- **Padded-row charge (part 2) did nothing here.** The offline diagnostic (before any timing) found cap-only and cap+pad picks identical on all 56 shapes, as the amendment said to expect (all M are powers of two). Nothing in this result supports part 2 either way; it matters only for non-power-of-two M, which I did not test.
- **Side note.** bf16 `qkv` M=1 has no recorded Inductor pick for either selector (Inductor lowers that shape without a template); the two ratios are 1.00005 and 1.00003, i.e. the same kernel. My first pick-difference count was 1 because of a NaN comparison; the script now counts 0.
- **Caveats.** One fresh model, one GPU type (both GPUs are the same part), 28 shapes per dtype, 3 reps; criteria (ii) and (iii) are single-shape maxima, so they are sensitive to the few small-M shapes. Three of the four Gemma shapes share K=3584 with Qwen 2.5 7B (different N), so this set is fresh by shape but not unrelated by dimension. The oracle is restricted and larger than in 5c (extra capped top-20), so it is no easier to match. fp8 has now failed pre-registered criteria on four held-out sets (8B, 70B TP=8, Qwen, Gemma) with three different fixes; bf16 has passed all five.

## F9. Round 6: small-M (decode) selection: 5e, 5f, 5g on fresh Mistral NeMo 12B and Phi-3 medium (2026-11-23/24)
Pre-registered in `PREREGISTRATION.md` (Amendments 5e/5f/5g) and committed before any validation timing; predictions in `results/r6/predictions_fresh.csv`. 112 shapes (2 models x 4 GEMMs x 7 M x bf16/fp8), all methods interleaved in one run on both RTX 6000 Ada GPUs, sustained protocol, 3 reps, medians, capture-after-compile harness. Oracle (M = 1, 8, 32, 128; pool also contains the 5e/5f top-20) timed on 24 shapes per dtype at M <= 32 for the criteria. Raw: `results/r6/r6_fresh_s*.csv`, `*_oraclepool.csv`, `r6_summary.json`, `r6_per_shape.csv`. **Calibration used SHORT diagnosis timing (0.15 s warmup / 0.3 s window / no cooldown), not the sustained protocol; validation used the full sustained protocol.**

### 1. What Step 1 showed (calibration shapes only)
- The current model predicts the *same* time for most BM=16 tiles (bf16 M=1 12288x3072: 0.0987 ms for all of them; measured 0.087 to 0.32 ms). The tie is broken by enumeration order.
- The thing that actually separates fast from slow tiles is **BK, i.e. the number of K-loop iterations** (share of within-shape log-time variance: 0.54 bf16, 0.72 fp8), then bytes in flight per block (about 0.4). Number of active blocks, warps and BN explain almost nothing on their own (R^2 <= 0.03). Small BK (16, 32) with few stages is latency-bound: per-iteration latency is divided by pipeline depth, and num_stages 1 and 2 behave the same (depth = stages-1, saturating at about 2-3).
- The current model is not bad on average on the calibration shapes (median pick 1.03x the measured best) but has large misses on a few shapes (up to 2.1x bf16, 1.9x fp8).
- A first, fuller form of 5e (achieved bandwidth from active blocks x bytes in flight, 3 parameters fitted to pooled RMS error) got a better RMS (0.73 to 0.39) but picked *worse* tiles (median 1.17x); a pooled-RMS fit rewards the slow tiles. It was dropped; the registered 5e is a 2-parameter K-loop latency floor fitted for ranking.

### 2. Verdicts on the fresh models (pre-registered criteria, per dtype; ratio vs stock max-autotune; (iii) = every M <= 32 shape within 1.10 of the oracle)
| method | bf16 geomean / worst | bf16 (i) (ii) (iii) | fp8 geomean / worst | fp8 (i) (ii) (iii) |
|---|---|---|---|---|
| `ours1` (round-5) | 0.950 / 1.06 | pass pass pass | 1.116 / 1.71 | FAIL FAIL FAIL |
| `ours1pc` (5d) | 0.950 / 1.06 | pass pass pass | 1.080 / 1.51 | FAIL FAIL FAIL |
| **`ours1e` (5e)** | 0.951 / 1.06 | pass pass pass, (iv) ok | 1.058 / 1.51 | FAIL FAIL FAIL (1.31 at `gate_up` M=1) |
| **`ours1f` (5f, fallback current)** | 0.955 / 1.15 | pass pass **FAIL** (1.13, Phi-3 `o` M=32) | 1.049 / 1.51 | FAIL FAIL pass (1.04) |
| **`ours1ef` (5f + 5e fallback)** | 0.955 / 1.15 | pass pass **FAIL** (1.13) | 1.049 / 1.51 | FAIL FAIL pass |
| `ours5d` (5c top-5) | 0.942 / 1.06 | pass pass pass | 1.018 / 1.36 | pass FAIL pass |
| **`ours_auto` (5g = 5d+5e+benchmark when unsure)** | 0.950 / 1.06 | pass pass pass, (iv) ok | 1.026 / 1.37 | pass **FAIL** pass |
| `tritonblas`-style | 0.992 / 1.19 | pass FAIL pass | 1.026 / 1.20 | pass FAIL pass |
| `cublas` | 1.156 / 1.75 | FAIL FAIL | 1.132 / 1.41 | FAIL FAIL |

**By the registered decision rule none of 5e, 5f, 5g becomes a default: each fails at least one criterion for fp8, and 5f also fails bf16 (iii).** (5g's (iv) passes: +/-0.001 vs `ours1pc`, one bf16 shape >1.05x.)

**What the failures are (so the fail is not over- or under-read).** Every worst shape in the fp8 rows above is at **M = 512 (`o`, `down`, `qkv`, up to 1.51x)**, with the same ~1.5x for every `ours1*` variant. This is mid-M, outside the decode regime that 5e/5f/5g are about, and it is identical across 5d/5e/5f by construction (their picks equal 5d's at M >= 128; 0 of 96 picks differ offline, and 0 of 96 on the fresh shapes). It is a separate problem, not caused or fixed by this round (the picked tile is BM=64, BN=64, 2 stages; tritonBLAS-style picks 128x256x128 there and is only 1.04x). **Post-hoc split (not a pre-registered criterion; read as description only):**
| decode, M <= 32, geomean / worst vs stock | `ours1` | `ours1pc` | `ours1e` | `ours1f`/`ours1ef` | `ours5d` | `ours_auto` |
|---|---|---|---|---|---|---|
| bf16 | 1.002 / 1.06 | 1.001 / 1.06 | 1.003 / 1.06 | 1.014 / 1.15 | 1.001 / 1.04 | **0.998 / 1.01** |
| fp8 | 1.166 / 1.71 | 1.082 / 1.32 | 1.031 / 1.32 | 1.010 / 1.05 | 1.010 / 1.03 | **1.009 / 1.08** |
At decode, 5g and 5f essentially close the fp8 gap (from 1.17 / 1.71 down to about 1.01 / 1.05-1.08). 5e alone helps only partly (1.082 to 1.031) and still leaves `gate_up` M=1 at 1.31x; the remaining `ours1e` miss is a tie the K-loop floor does not break. 5f's (iii) bf16 fail comes from the table choosing a poor tile for Phi-3 `o` M=32 (an N,K that is far from the four calibration sizes).

**Completeness and criterion (v).** Both shards finished: 56 shapes each (112 total), 11 methods, 3 reps for every shape/method, no missing or non-positive timings. Criterion (v) (median Kendall tau vs `ours1pc`) is garbled in the pre-registration text ("at least ours1pc's...") and is diagnostic only; it is reported in section 3 and is not part of any pass/fail verdict above. Step 1 conclusions come from SHORT diagnosis timing, not the sustained protocol.

### 3. Ranking quality (Kendall tau between predicted and measured time over the oracle's candidate pool, M <= 32, median over shapes; about 140-150 timed candidates per shape)
| | current (5d) | 5e | 5f | 5e+5f |
|---|---|---|---|---|
| bf16 tau | 0.696 | 0.686 | 0.690 | 0.686 |
| fp8 tau | 0.751 | 0.750 | **0.787** | 0.754 |
| median number of pool candidates predicted within 2% of the model's best | 51 | **22** | 27 | 27 |
So on the fresh pool **5e roughly halves the ties but does not improve overall rank correlation** (the calibration gain in tau on near-best tiles, 0.59 to 0.69, did not carry over to the whole pool here; different metric and different shapes); 5f improves fp8 tau slightly. Ties remain large, which is why the decisive improvement at decode comes from 5g's benchmarking rather than from a better analytic model. Honest reading: **5e and 5f did not make the analytical model a reliable ranker at small M.**

### 4. Costs
- **5f calibration:** about 37 min per GPU when sharded over two GPUs (3,540 timings each; about 75 min on one GPU), slightly under the 45-min-per-GPU target for two GPUs and over it for one. Profile 132 KB to 254 KB (485 tile entries; indent-2 JSON). Command: `python3 -m lala_triton.calibrate --decode-table --out profiles/<gpu>.json --shard i/n`, then `--build-only`. The table has only 4 calibration (N,K) sizes (thin); on seen models 12 of 96 decode shapes needed fallbacks, none of the 48 fresh decode shapes did.
- **5g:** benchmarks (returns up to 5 tiles) on **79% of shapes overall and 100% of M <= 32** (offline prediction was 80% / 100%: the model is always "unsure" at decode, so this is not near-zero tuning there). Even so, geomean first-call time is **0.25 s vs 9.2 s for stock max-autotune (0.027x)**, passing the registered tuning-cost criterion (vi) <= 0.75x; `ours1pc`/`ours1e` need 0.06-0.13 s, 5c top-5 1.3 s. Benchmark fraction and tuning time are for the RTX 6000 Ada with warm Triton compile cache effects as in earlier rounds.

### 5. Recommended configuration for Dongyoon's next run
Run every method (README command) on the L40S: it is the same AD102 die, so this is the closest test of whether decode picks transfer. The pre-registered rule promotes none, so the defaults stay as they are. For a decode (M <= 32) deployment candidate use **5g (`ours_auto` = 5d + 5e + benchmark up to 5 distinct tiles when unsure)**: best or tied-best at decode in both dtypes (bf16 0.998 / worst 1.01; fp8 1.009 / worst 1.08) at about 3% of stock tuning time, no bf16 regression. 5f is optional (it adds calibration and failed bf16 (iii)); 5e alone is not enough for fp8 at M=1. Do not read any of this as fixing fp8 overall: M = 512 fails for every `ours*` method.

### 6. Honest assessment
Does this support "analytical model for large M plus a measured decode table matches autotuning with near-zero tuning"? **Not as stated.** (a) At decode, the *measured table* (5f) did not beat the simpler analytic floor everywhere and failed a bf16 criterion; what matched stock was **benchmarking 5 tiles (5g)**, i.e. a small, not zero, tuning cost, triggered on every decode shape. (b) The claim for large M is not supported on fp8: M=512 `o`/`down`/`qkv` are 1.3-1.5x slower than stock for our top-1 and 1.1-1.4x for 5g. (c) What the evidence does support: on bf16 the selector matches or beats stock (geomean about 0.95) with near-zero tuning, and a 5-candidate benchmark at about 3% of stock tuning time matches stock at decode on both dtypes. Missing: other GPUs (Dongyoon's L40S is pending), Hopper, the real tritonBLAS code (ours is a transcription), the mid-M fp8 problem, an end-to-end serving result, a 5f table with more than four calibration sizes, and sustained-protocol (not short) timing for the calibration sweep. Single-GPU-model, power-capped, one run per shape; noise between repeated shapes is about 2 points.
