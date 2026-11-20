# Round 2: power coupling, real baselines, dynamic M

Run 2026-10-16 on GPU0 (RTX 6000 Ada, 300 W cap). GPU1 was used only for ncu. All thresholds are in `PREREGISTRATION.md` Amendments 2 and 2b, written before any round-2 data. Every phase ran (A–H), and none was dropped.

## Summary
**Is the power coupling real?** Yes in direction, small in size. On this card the 300 W cap is hit by *every* GEMM we ran, even M=128. Lowering DRAM traffic raises the sustained clock: median ρ(clock, bytes) = −0.66 across G-sensitive shapes, and once runtime is converted to clock cycles the G effect disappears. But measured properly (sustained, DRAM-cold), the G effect is only about 2% at the median, not the ≥5% round 1 suggested. Round 1's short benchmarks overstated it.

**The power-aware time model (M4) as specified can't be fit.** Power is pinned at the cap in every run, so a regression with power as the response has nothing to explain. The card adjusts its clock, not its power.

**Does this beat what people use today?**
- **bf16: yes.** M3-guided picks for the vendored tutorial kernel are faster than Inductor max-autotune on 69% of 42 shapes, with a geomean speedup of 9% (0.914× the time). They are faster than cuBLAS on 39 of 42 (geomean 0.72× the time), at about 0.29× Inductor's tuning time for top-5 and about 0.06× for top-1.
- **fp8: no.** Inductor's own fp8 Triton template is about 24% faster than the *best* config of our kernel, so model-guided selection can't compensate.
- **Overall:** H7 as pre-registered fails (61% of shapes ≥ Inductor within noise; tuning-time ratio 0.28 vs ≤ 0.2 required).
- **Dynamic M:** with no benchmarking at all, the model closes 58–102% of the gap between bucketed tuning and a per-M oracle on all three (N, K) pairs.

## Verdicts
| | Pre-registered criterion | Result | Verdict |
|---|---|---|---|
| **H5** power coupling | median ρ(clk, bytes) ≤ −0.5 and median explained fraction ≥ 0.5 (locked-clock arm impossible: no permission) | ρ = **−0.66** (95% CI −0.94…+0.46). Explained fraction **1.12** (CI 0.69–1.66, n=9; 2 shapes excluded for no sustained speedup). Restricting to the 7 shapes where G actually changes bytes ≥1.3×: ρ = −0.71. Cycles(G8)/cycles(bestG) median 0.99 | **PASS** (wide CI; small effect) |
| **H6** energy | ≥5% less energy at bestG on ≥50% of shapes and Wilcoxon p < 0.05 | 4/11 shapes (36%). Median energy ratio 0.968, p = 2e-5 | **FAIL** (consistent but small, ~3%) |
| **H3'** M4 vs "M1 with G=8" | mean regret lower by > 0.092 at k=1 and k=3 | Power fit degenerate (R² = 0.01, P≈299 W everywhere). M4 k=1/3 regret 1.352/1.159, vs M1-G8 1.308/1.153 and M3 1.096/1.056 | **FAIL**, so M3 was used for Phases D/E |
| **H7** vs Inductor | Ours-top5 ≥ B2 within noise on ≥70%, ≥5% faster on ≥20%, tuning ≤ 1/5 of B2's | 61% / **47%** / 0.28. bf16 only: 88% / 69% / 0.29. fp8 only: 5% / 0% | **FAIL** (not killed: B2 within noise of our-kernel oracle on 48% < 80%) |
| **H8** dynamic M | closes ≥50% of the bucket→oracle gap on ≥2 of 3 (N, K) | 58%, 73%, 102% | **PASS** |
| **Soft knee** (Phase G) | worst near-cliff error ↓ ≥2× and median ≤ 7% | worst 93.5% → 74.2% (1.26×); median 5.7% | **FAIL** |
| **NVMMH** (Phase F) | descriptive | Installs, supports RTX 6000 Ada, has a Triton target. Traffic error 121–175% vs M3's 0.1–5%. **No L2 capacity knee**. Always prefers G=8. Its picks are slower than ours on 45 of 62 shapes and faster on 6 | see `RELATED_WORK.md` |
| **Eviction hints** (Phase H) | descriptive | No change in bytes or time on 36 cliff configs: Triton 3.2 drops `eviction_policy` when a load is pipelined into `cp.async` (any num_stages ≥ 2); hints reach the PTX only at num_stages=1 | hints never reach the hardware |

## Tuning cost
Phase D, median per shape, wall-clock including cold compile:

| method | configs benchmarked | wall |
|---|---|---|
| B2 Inductor max-autotune | 20 | 8.9 s |
| B3 tutorial autotune list | 16 | 9.8 s |
| Ours top-1 / top-3 / top-5 | 0 (compile only) / 3 / 5 | 0.54 / 1.58 / 2.52 s |
| NVMMH top-1 / top-5 | 0 / 5 | 0.49 / 1.89 s |

Phase E, per (N, K):
- **Static:** one exhaustive 1,265-config sweep, ≈25 s warm (≈330 s with cold compile).
- **Buckets:** 4 sweeps, ≈150 s warm.
- **Ours:** 0 configs benchmarked, about 2 µs of model evaluation per config.
- **Inductor dynamic:** tunes once at the hint M (≈7–8 s), then gets 1.21–1.25× the oracle geomean.

## Plots (`plots/`)
- `r2_clock_vs_bytes.png`: sustained clock vs DRAM bytes per G-sensitive shape
- `r2_time_energy_vs_G.png`: time and energy vs G, unlocked only (locking not permitted)
- `r2_regret_vs_k_M4.png`: regret vs k for M3, M4, M1 and M1-G8 on round-1 data
- `r2_speedup_vs_B1_B2.png`: per-shape speedup over cuBLAS and Inductor
- `r2_dynamic_M.png`: time / per-M oracle across M for static, bucket, ours and Inductor

## What surprised me
1. **The cap binds for every GEMM, even tiny ones.** Every sustained run sits at 297–301 W with the SW power cap active in 100% of samples. Clocks range 700–2100 MHz depending on how much work is DRAM movement. Power stays constant and clock is what varies, so the spec's M4 (power as the response) is unidentifiable here. A clock-as-response model is the natural fix, but it was not pre-registered, so not attempted.
2. **Round 1 overstated G effects.** With sustained, DRAM-cold timing, round 1's best G is still best on only 6 of 11 "G-sensitive" shapes, and the median best-G vs G=8 gain is about 2%. The largest surviving effects are on big shapes where G swings traffic 1.6–11× (8192×14336×4096 bf16 12%, fp8 9%; 8192×28672×8192 bf16 7%), plus 128×4096×4096 fp8 at 8%.
3. **Our bf16 wins over Inductor and cuBLAS come mostly from tile choice, not G.** Inductor fixes GROUP_M=8, but our top-1 beats it by 8% geomean. On this card, cuBLAS via torch 2.6 is slow for bf16. The one shape profiled ran an `ampere_bf16_s1688gemm` kernel, i.e. an Ampere-generation kernel. It also fails our tolerance on two K=14336 shapes (max error about 6).
4. **NVMMH ignores L2 capacity** in its byte predictions, and its Python metric wrapper segfaults.
5. **Triton's eviction hints vanish under pipelining**, so the LADM-style insertion-policy idea is not expressible in pipelined Triton GEMMs on sm_89.

## Threats to validity
- **No clock locking.** H5 rests on correlation, not intervention. The ρ confidence interval crosses zero, and 4 of the 11 shapes have essentially constant bytes across G.
- **One GPU, power-capped.** Results may not transfer to uncapped or HBM parts.
- **Different benchmarking methods.** The H3' and Phase C evaluations reuse round-1 *short* do_bench timings. Phase D/E used sustained or do_bench timing respectively, as pre-registered.
- **Possible pseudo-oracle bias.** For the A@B.T bf16 shapes, B4 is a pseudo-oracle, the best over configs that the methods themselves proposed.
- **Noisy per-M oracle in Phase E.** It was chosen with single-rep 10 ms benchmarks over about 200 candidates, and the model sometimes beats it after re-timing. The buckets are as specified; tuning at M=576 transfers poorly across 129–1024.
- **Unfair kernel family on fp8.** Inductor uses its own fp8 template, which outperforms the tutorial kernel's whole config space.
- **The NVMMH mapping is ours.** Its configs often want split-k, which our kernel lacks, so it is handicapped.
- **M3 reuses round-1 eff.** It was not recalibrated for the A@B.T layout.

## Single best next step
**Make the selector's kernel family competitive on fp8.** Point M3 at Inductor's own Triton mm / scaled_mm templates, and let the model choose GROUP_M, which Inductor never varies.

For bf16 the result is already practically useful: zero-benchmark top-1 is about 8% faster than Inductor max-autotune at 1/16 of its tuning time.

If the power story is worth pursuing, re-run H5 with clock locking on a machine where it is permitted. That is the only clean test of causation.
