# Round 3: cuBLAS sanity check + model inside Inductor's templates

Run 2026-10-19/20 on the same two RTX 6000 Ada GPUs. Both are capped at 300 W, and every timed run sat at the cap.

Thresholds are in `PREREGISTRATION.md` **Amendment 3**, written before any round-3 data. **Amendment 3b** is a harness-fidelity fix, written before any 2c/2d comparison data; see "Threats". All comparisons use the round-2 Phase D sustained protocol: DRAM-cold, CUDA-graph replay, interleaved, 3 paired reps, with the SM clock recorded for every run.

## Summary

**1. Is the cuBLAS comparison real? Yes, but smaller than round 2 said, and part of it is clock.**
- **Correctness.** Against an fp64 reference, both sides are within the bf16 output-rounding envelope once cuBLAS runs with `allow_bf16_reduced_precision_reduction=False`.
  - With torch's default (True), cuBLAS exceeds the envelope on 31 of 42 bf16 shapes, by up to 88×. The cause is the "sliced" (in-CTA split-K) kernels, which reduce partial sums in bf16.
  - Round 2's "max error ≈ 6" was this effect showing through a loose tolerance. It is not a cuBLAS bug.
  - Turning the flag off fixes it and makes cuBLAS about 8% *faster*.
- **Speed.** Against the best *correct* cuBLAS variant per shape, our round-2 Triton picks are **1.19× faster in geomean** (95% CI 1.13–1.26). That is faster on 40/42 shapes and ≥5% faster on 30/42.
  - The best cuBLAS variant was taken over 6: torch 2.6, the cublasLt backend, torch 2.14.1 with cuBLAS 13.1, nvmath-python cuBLASLt heuristic top-8, and each with the flag off.
  - **H9 passes.** Round 2's 28% was inflated by comparing against the default (incorrect, slower) cuBLAS setting.
- **Kernel and clock.** Every cuBLAS version, including cuBLAS 13.1, dispatches **Ampere-generation kernels** (`ampere_bf16_s1688gemm…`, `ampere_s16816gemm…`) on this Ada GPU. They run at a lower sustained clock: median 865–890 MHz versus 990 MHz for our kernels, at the same 300 W.
  - Per clock cycle, our advantage is only **1.07×**. Roughly 60% of the gap is power-to-clock behaviour under the cap, not better code.
- **fp8 is not evaluable under the pre-registered criterion.** See point 4.

**2. With Inductor's own templates, does the model still win on bf16? No. Does it now win on fp8? It ties, but doesn't win by the pre-registered bar.**
- **bf16.** Model top-5 inside `mm_template` is 1.01× S1's time (geomean): within noise on 83% of shapes, ≥5% faster on only 9.5%. **H11 fails.**
  - Round 2's 9% bf16 win over Inductor came from the tutorial kernel's larger tile space (e.g. 128×256) and its kernel code, not from a better selector.
- **fp8.** The kernel-family gap from round 2 is gone: ours equals S1 (0.999×) at 0.14× of S1's tuning time. But it is within noise on only 75% of shapes (80% required), so **H10 fails narrowly**.

**3. Where does the gain come from? Mostly nowhere.** Median per-shape ratios:

| ratio | bf16 | fp8 |
|---|---|---|
| search space (G ∈ {1…32} added) S2/S3 | 1.000 | 1.017 |
| model vs Inductor S1/S4 | 0.999 | 0.990 |
| choosing G S5/S4 | 1.000 | 1.000 |

- **G tuning is not where the value is on this GPU.** M3 picked G=8 as top-1 on all 62 shapes, and its top-5 lists were usually identical with or without G.
- The full extended-space oracle (S3) is 5–6% faster than S1, and in 17/62 shapes S3 beats the G=8 oracle by ≥5%. Neither our model nor Inductor's autotuner finds those configs.

**4. Two new findings (not hypotheses).**
- **Power cap re-orders configs.** Under the cap, configs that move less data on chip run at higher clocks: for example 256×128×128 at 1,140 MHz versus 128×128×32 at 895 MHz on the same shape. Short do_bench bursts don't see this. That's why Inductor's own autotune pick (S1) is 2.7% slower than the best of its own list under sustained load, and why our burst-calibrated `eff` mis-ranks fp8 tiles.
- **fp8 accuracy.** Triton's fp8 GEMMs on sm_89, Inductor's template included, accumulate along K at reduced precision, with envelope ratio R = 43–107. cuBLAS `_scaled_mm` (R = 1.7–6.2) evidently promotes to fp32. Inductor's `USE_FAST_ACCUM=False` does not change the Triton result.
  - The tutorial kernel and the template produce identical errors. That is consistent with Triton folding `acc += dot(a, b)` into `dot(a, b, acc)`, but I haven't confirmed it in the compiled IR.
  - No fp8 GEMM passes our fp32-accumulation envelope (R ≤ 2), so fp8 H9 has no eligible shapes. fp8 cuBLAS numbers below are descriptive (marked "S6np").

## Verdicts
| | Pre-registered criterion | Result | Verdict |
|---|---|---|---|
| **H9** bf16 "beats cuBLAS" | geomean t_bestcuBLAS / t_ours ≥ 1.05 on shapes where both pass the fp64 envelope (R ≤ 2) | **1.19** (CI 1.13–1.26), n=42. A@B 1.18, A@B.T 1.20. Per cycle 1.07 | **PASS** |
| H9 fp8 (secondary) | same | No shape where ours passes (fp8 Triton R 43–107); cuBLAS passes on 4/20 | **N/A** |
| **H10** fp8, templates | ours top-5 vs S1: within noise ≥80%, ≥5% faster ≥20%, wall ≤ 1/3 | 75% / 20% / 0.14 | **FAIL** (narrowly, on the first criterion) |
| **H11** bf16, templates | same | 83% / 9.5% / 0.51 | **FAIL** |
| **H12** dynamic M | closes ≥50% of the bucket→oracle gap on ≥2/3 (N, K) | bf16: −9.1, −10.2, −15.2. fp8: −7.8, −15.3, −16.1 (ours is *worse* than buckets on every pair) | **FAIL** (bf16 and fp8) |
| 2a harness (i) | regs/shared/spills identical to Inductor-compiled kernel, 12 random configs | 12/12 identical registers and bit-identical output. SASS identical on 8/8 bf16. fp8 differs only by one extra scale-pointer argument | PASS |
| 2a harness (ii) | median \|log(harness / Inductor-logged)\| ≤ 0.05 | 0.030 (537 configs, 62 shapes) | PASS |
| 2b unit test | median \|M3 bytes − ncu DRAM bytes\| ≤ 10% | 0.08% (n=50; p90 11%) | PASS |

**Why H12 failed.** It fails for two separate, diagnosable reasons, visible in `plots/r3_dynamic_M.png`:
1. **M ≤ 32.** The model picks the tiniest template tile (16×32×16, one stage), which is 4–6× slower than the oracle. M3 predicts every config at the DRAM roofline in this regime and has no term for whether a tile can *sustain* bandwidth.
   - Round 2's config grid never contained BK=16 or single-stage tiles, so this blind spot never showed.
2. **fp8 at large M.** The burst-calibrated eff prefers BK=32 tiles, which lose under the power cap (point 4 above).
- Descriptively, restricted to M ≥ 64, bf16 closes 23–36% of the gap. fp8 is still worse than buckets.

## Gain decomposition (per family; per-shape values in `runs/r3_2c_per_shape.csv`, `plots/r3_gain_decomposition.png`)
| | bf16 A@B (22) | bf16 A@B.T (20) | fp8 (20) |
|---|---|---|---|
| S2/S1: oracle within the stock list vs Inductor's pick (geomean) | 0.97 (bf16 overall) | | 0.97 |
| S3/S1: oracle over the extended space vs Inductor (geomean) | 0.950 | 0.944 | 0.938 |
| S4 top-5 / S1 (geomean) | 1.024 | 0.996 | 0.999 |
| S4 top-1 / S1 (geomean) | 1.010 | 0.988 | 1.457 |
| median search-space gain S2/S3 | 1.005 | 1.000 | 1.017 |
| median model gain S1/S4 | 0.999 | 0.997 | 0.990 |
| median G contribution S5/S4 | 1.000 | 1.000 | 1.000 |
| shapes with S3 ≥ 5% faster than S2 | 6 | 4 | 7 |
| S4 / S6 (best cuBLAS; fp8 non-passing) | 0.908 | 0.935 | 0.900 |
| S1 / S6 | 0.887 | 0.939 | 0.901 |

Both Inductor-family methods are ~9% faster than the best cuBLAS. **Inductor (S1) already beats cuBLAS by about as much as our round-2 kernel did.**

## Tuning cost (median per shape, cold compile, fresh process and cache, GPU1)
| method | bf16: configs / wall | fp8: configs / wall |
|---|---|---|
| S1 stock max-autotune (ATEN+TRITON) | 20 / 7.2 s | 98 / 29.1 s |
| S4 top-1 (patched Inductor, 1 config, no benchmark) | 1 / 2.0 s | 1 / 1.9 s |
| S4 top-3 | 3 / 3.2 s | 3 / 3.3 s |
| S4 top-5 | 5 / 3.7 s | 5 / 4.0 s |
| S2/S3 oracle screening (harness do_bench) | median 91 configs per shape | |

About 2 s of every patched-Inductor run is fixed dynamo/Inductor overhead. That is why bf16 top-5 can't reach 1/3 of S1's 7 s, even though it benchmarks 4× fewer configs.

## Step 1 details (`runs/r3_step1_per_shape.csv`, `runs/r3_h9_verdict.json`)
- **Versions.**
  - Main env: torch 2.6.0+cu124, cuBLAS 12.4.5.
  - New env: torch 2.14.1+cu130 (latest on PyPI), cuBLAS 13.1.1, nvmath-python 1.0.0. It's a venv on `/var/tmp`, because the home quota was full; the main env is untouched.
- **bf16 time relative to V1 (torch 2.6 default)**, geomean over 42 shapes:

  | variant | time / V1 |
  |---|---|
  | cublasLt backend | 1.00 |
  | torch 2.14 | 1.00 |
  | torch 2.14 + Lt | 1.00 |
  | nvmath heuristic top-8 best | 0.94 |
  | flag off (V1r) | 0.92 |
  | ours | 0.72 |

- **Best cuBLAS variant per shape:** V1r 27, nvmath 8, others 7. The newer torch and cuBLAS gave nothing: same kernels, same speed.
- **Error envelope on the M=128, K=14336 shape** (refmax 574):
  - bf16 has 8 significant bits, so the spacing between representable values (one ulp) is 4 at 512–1024. Correct rounding errs by at most half of that, i.e. 2.
  - Ours and cuBLAS with the flag off: max error 1.93, R = 0.97 (exactly rounding).
  - cuBLAS with the flag on: max error 6.5. The worst element has |ref| = 0.016 and error 1.65, i.e. partial-sum rounding at the magnitude of the partial sums. R = 65.

## Plots
- `plots/r3_speedup_vs_S1_S6.png`: per-shape speedup of ours (top-5) over S1 and S6
- `plots/r3_gain_decomposition.png`: S2/S3, S1/S4, S5/S4 per shape
- `plots/r3_dynamic_M.png`: time / per-M oracle vs M, bf16 and fp8, for static, buckets, ours, and Inductor dynamic

## Threats to validity
- **Triton process-state quirk (Amendment 3b).**
  - Triton 3.2 sets LLVM's process-global `nvptx-short-ptr` during the first PTX lowering in a process. So the first kernel compiled in any process differs from later ones (e.g. 154 vs 164 registers) with identical TTIR.
  - Inductor's final S1 kernels were in the "first" state on 59/59 template picks. The harness is pinned to the other state.
  - Measured effect: ratio 0.97–1.03 with mixed sign on 6 configs. A per-shape S1-vs-harness difference of ±3% is therefore possible noise in H10/H11.
- **The oracle is "best of top-3 by screening"**, re-timed sustained. It is occasionally beaten by another method (ratios below 1 in the 2d plot). S3 ≤ S2 by construction, so min-of-more-noisy-candidates bias favours S3 by ~1%.
- **Clock is part of every number.** Sustained clocks differ by up to 40% between configs at identical power. Results on an uncapped or HBM GPU could differ qualitatively.
- **Inductor's dynamic mode** (`dynamic=True`) makes all dims symbolic, which turns off EVEN_K. Dynamo still specializes and re-autotunes at M=1. That is Inductor's real behaviour, not a harness choice.
- **fp8 S6 is not a correct baseline.** It fails the envelope on 16/20 shapes, as does every Triton fp8 kernel.
- **The fp8 harness** passes two scale pointers where Inductor deduplicated them into one (≤ 8 SASS instructions in the epilogue).

## What surprised me
1. **cuBLAS's default bf16 setting is the less accurate *and* slower one** on this GPU, and every cuBLAS up to 13.1 still ships Ampere kernels for Ada bf16.
2. **Stock Inductor already beats the best cuBLAS by ~9% on bf16 and fp8.** Our round-2 "win over cuBLAS" was mostly "Triton vs Ampere-kernel cuBLAS".
3. **The power cap matters more than locality.** It re-ranks configs by on-chip data movement (clock varies 765–1,145 MHz across configs of one shape), which dwarfs anything G does.

## Single best next step
Replace the burst-calibrated `eff` with a **clock-as-response model calibrated under sustained load**: predicted time = cycles ÷ predicted sustained clock, with clock a function of on-chip traffic (SMEM/L2 bytes per FLOP). Add a **minimum-pipelining / bandwidth-attainment term** for small tiles.

Both failure modes in this round (fp8 tile ranking, and small-M dynamic shapes) trace to exactly those two gaps. The data to fit it is already on disk in `runs/r3_E_eval.csv` and `runs/r3_2c_eval.csv`, which record clock and power per config. It must be validated on fresh shapes.
