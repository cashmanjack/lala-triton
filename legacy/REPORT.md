# Kill test: asymptotic L2 locality for Triton tile selection

Run 2026-10-15 on 2× RTX 6000 Ada (AD102, 142 SMs, 96 MiB L2, ~800 GB/s GDDR6, **power-capped at 300 W, so it runs at 0.8–1.1 GHz under GEMM load**), with Triton 3.2.0 and ncu 2024.1.1. All thresholds were fixed in `PREREGISTRATION.md` before the data was collected. Amendment 1 is labelled "~17:45", but the file mtime shows it was written at 16:42, before any Phase 2/3 data.

## Summary
The traffic model works. It predicts ncu-measured DRAM read bytes with a median error of 5% on the controlled sweeps and 0.1% on a random 10% subset of the full sweep, with Spearman ρ of 0.99 or better in both. The naive per-tile formula is off by about 10×. It also gets GROUP_M right: in the streaming regime, the traffic-optimal G measured 12–16, exactly the predicted `sqrt(W·BN/BM)`, while the tutorial default of 8 was never optimal. Where it fails is close to the capacity cliff. Large-G knees arrive early and smeared, and footprints within a few percent of L2 thrash when the model says they fit.

Locality does move runtime here. 11 of 42 test shapes show a G effect of at least 5% in every timing rep, with best-G vs G=8 gains of up to about 20–25%. Most of those shapes are ones the model labels compute-bound rather than near-ridge.

As a pruner, M3 is clearly the best of the four models. It gets within 5% of the oracle on 100% of held-out shapes using 10% of configs, its mean top-1 regret is 1.10 (versus 1.40 for the naive model and 1.55 for Triton's old model), and it beats M1 by more than noise on near-ridge shapes. But it only cuts the configs needed for "90% of shapes within 5%" from 62 to 39, which misses the "≤ half" criterion. Against a naive baseline that simply fixes G=8, its edge narrows.

**Nothing was killed.** H1 is a partial pass and H2, H3 and H4 pass, but the H2 and H3 margins are thin given the about 9% timing noise this power-capped card produces.

## Verdicts
| | Pre-registered criterion | Result | Verdict |
|---|---|---|---|
| **H1** traffic validity | median err ≤25%, ρ ≥0.85, every knee within 2× | err **5.4%** (M1: 1223%), ρ **0.992** (n=308). Knees: 15/17 within 2×; the G=32 R0→R1 knee is **3.4× early** (both tiles). Random Phase 3 subset: err **0.1%**, ρ **0.996** (n=5612; M1 ρ 0.79) | **PARTIAL** (2–4× knee miss: neither pass nor kill) |
| **H2** relevance | ≥20% of near-ridge shapes: best-G beats G=8 by ≥5% (> noise), or the best tile changes | **3/11 = 27%**. Only 2/11 hold in every rep. All 42 shapes: 21 hit by median; 11 have ≥5% in every rep | **PASS (marginal)** |
| **H3** pruning value | M3 regret ≤1.05 at k=10% on ≥90% of shapes, AND (a) ≤½ M1's k, or (b) lower near-ridge regret at k=1,3,5,10 by more than noise | 100% of shapes. (a) ✗: k₉₀ = 39 vs 62. (b) ✓: near-ridge regret M3 1.050/1.013/1.012/1.011 vs M1 1.437/1.135/1.124/1.108; every gap exceeds the 8.8% noise floor (k=10 only barely: 9.7%) | **PASS (via b)** |
| **H4** optimal G (nice to have) | measured best G within one 2× step of √(W·BN/BM) | 6/6 R2 sweeps: measured best 12 or 16 vs 11.9 / 16.9 (≤0.43 of a step) | **PASS** |

Regret at k (mean over 42 shapes; regret = best-in-top-k / oracle):

| model | k=1 | 3 | 5 | 10 | 20 | 5% | 10% | Kendall τ |
|---|---|---|---|---|---|---|---|---|
| M3 L2-aware | **1.096** | **1.056** | **1.037** | **1.031** | **1.027** | **1.003** | **1.000** | **0.69** |
| M1 naive per-tile | 1.396 | 1.139 | 1.111 | 1.089 | 1.058 | 1.017 | 1.012 | 0.64 |
| M2 old Triton perf model | 1.553 | 1.302 | 1.227 | 1.189 | 1.149 | 1.063 | 1.041 | 0.64 |
| M0 random | 2.290 | 1.588 | 1.419 | 1.283 | 1.189 | 1.091 | 1.059 | — |
| *M1 with G fixed to 8 (supplementary)* | 1.159 | 1.123 | 1.111 | 1.080 | | | | |

Plots:
- `plots/knee_sweep.png`
- `plots/g_sweep.png`
- `plots/phase2_scatter.png`
- `plots/phase3_scatter.png`
- `plots/regret_vs_k.png`

## What surprised me
1. **This L2 behaves like one clean cache.** The double-read microbenchmark shows 0% second-pass misses up to 88 MiB and a sharp knee at about 102 MiB, with no partition duplication. That is why the step-function model fits so well here. Expect this to be worse on A100/H100.
2. **Knees get early and smooth as G grows.** For G=1 the knee is sharp and on time. For G=16–32, B-reuse loss starts about 2–3.4× earlier and ramps gradually, even though the modelled footprint is well under L2. Some unmodelled reuse-distance spread is likely, from CTA stagger and a non-lockstep window, and it grows with G. I did not tune L2_eff to hide this.
3. **Hard thresholds are fragile at the cliff.** G=64 with F2 at 98% of L2_eff is predicted as "fits" but measured at 15× more traffic. A soft knee would fix most of the worst errors.
4. **G matters for "compute-bound" shapes.** Big G effects (10–25%) appear on M=512–8192 shapes with model intensity of 2–10× the ridge. The near-ridge filter, mostly M=128 shapes where Tm is about 1–4, is not where the effect lives. A plausible cause is that on a power-capped card DRAM traffic costs clock speed, but this was not tested.
5. **Compile is cheap.** Median compile is 0.3 s per config, so the "avoid compiling" argument is weak. The model costs 2.3 µs per config.
6. **The spec's SMEM formula is off by one stage.** Triton 3.2 on sm_89 uses `(num_stages-1)` buffers. That is exact for bf16, but fp8 needs 1.25–2.25× more.

## Threats to validity
- **Timing noise.** Under the power cap, the 10-retiming noise floor averages 9% (max−min/median), and per-rep G gains sometimes flip sign. The H2 and H3(b) margins sit close to this.
- **Selection bias.** Best-of-5-G versus G=8 is biased upward by noise. That is why the every-rep check is reported.
- **Which baseline is fair.** M1 is G-blind, so ties among G were broken randomly as pre-registered. That makes M1 weaker than a practitioner who would fix G=8. On near-ridge shapes, M3 restricted to G=8 does as well as full M3, so part of M3's advantage there is tile choice driven by its traffic estimate, not G choice.
- **eff calibration.** eff was calibrated at G=8 on 4 large shapes and shared by M1 and M3, so the ablation stays clean. It is still heavily power-state dependent (best sustained bf16 was 162 TFLOPS vs 364 published).
- **Timing split across GPUs.** Shapes were split across the two identical GPUs. Regret is computed within each shape.
- **Hardware coverage.** One GPU type was tested; the large, unpartitioned L2 is favourable to the model.
- **M2 constants.** M2 is vendored from Triton v3.0.0 with A100-era constants, so it is a weak strawman.
- **Approximations.** Partial waves are handled fractionally, not exactly. Dispatch order was assumed, not measured. The persistent-kernel check was not needed, since H1 was not killed on the window assumption.

## Most promising next step
**Benchmark against nvMatmulHeuristics**, NVIDIA's analytic GEMM picker. It supports Ada and claims L2 hit-rate prediction (see `RELATED_WORK.md`). It is the real competitor, and the novelty claim depends on beating it. At the same time, replace the hard regime threshold with a soft knee calibrated once from the Phase 1 miss curve, and re-run H1/H3 on an H100 or B200, where partitioned L2 is the open risk.

## Files
- `PREREGISTRATION.md`
- `env/`: inventory, L2/bandwidth/eff/ridge calibration
- `model/`: M1/M2/M3
- `tests/`: unit tests, 4/4 pass
- `predictions/`: written before the matching runs
- `runs/`: raw CSVs, ncu output, per-shape H2/H3 tables
- `analyze_*.py`: analysis scripts. `analyze_extra.py` is supplementary, not pre-registered.
