# Pre-registration: asymptotic L2 locality for Triton tile selection

> **Naming note (2026-11-20):** the Python package was renamed `tritonsel` -> `lala_triton` (repo `lala-triton`) after this was written; env vars `TRITONSEL_*` -> `LALA_TRITON_*`. Older text below that says `tritonsel` means the same code.


Written: 2026-10-15 ~16:50 local, **before** any calibration, ncu sweep, or timing data was collected.
Any later amendment is appended below with its own timestamp and a reason. Nothing above the amendment line changes.

## Platform (from Phase 0 inventory, env/inventory.json)
- 2x NVIDIA RTX 6000 Ada (AD102, sm_89), 142 SMs, 96 MiB nominal L2, 99 KiB max SMEM/block, GDDR6 384-bit at 10 Gbps (960 GB/s nominal).
- Triton 3.2.0, PyTorch 2.6.0+cu124, driver 580.178.04, ncu 2024.1.1. Counters permitted.
- Clocks can't be locked (no root), so clocks and temperature are logged per row.
- The spec was written with H100 in mind. On this part, W is about 142 (not 264) and the L2 is 96 MiB (not 50 MB). The hypotheses and thresholds below are unchanged from the spec; only the platform numbers differ.
- GPU0 is used for all timing, including eff calibration. GPU1 is used for ncu byte counts, which don't depend on timing. Exception: if Phase 3 timing exceeds budget, whole shapes may be moved to GPU1. Regret is computed within a shape, so this is safe, and it will be logged per row.
- Published peak (spec sheet): bf16 dense 364.2 TFLOPS, fp8 dense 728.5 TFLOPS. Ridge = peak / measured HBM_BW (Phase 1).

## Calibration rules (Phase 1)
- **L2_eff**: one kernel reads a buffer of size S twice (each CTA re-reads its own chunk), grid = all SMs, measured with `--cache-control all`. Second-pass miss fraction = (dram_read - S) / S.
  - Knee = the S at which the miss fraction crosses 0.5, by log-linear interpolation.
  - Two variants: sequential, and strided (prime-stride 128 B segments).
  - **L2_eff = min(knee_seq, knee_strided)**, fixed before Phase 2 and never refit.
- **HBM_BW** = streaming copy bandwidth (read + write bytes / time) on a 2 GiB buffer, median of do_bench. Read-only bandwidth is recorded but not used.
- **L2_BW** = repeated reads of an L2-resident 32 MiB buffer.
- **eff(config)** = ideal_compute_time / measured time.
  - ideal = 2MNK / (published peak × occ_share), with the same per-SM accounting as the time model.
  - G fixed at 8.
  - Median over the calibration shapes: (8192, 8192, 8192), (16384, 8192, 8192), (8192, 16384, 8192), (12288, 12288, 12288) bf16. For fp8: the same shapes in fp8.
  - These shapes are excluded from every test set.
- **Occupancy** `occ` comes from compiled metadata (shared, n_regs, num_warps) against SM limits (100 KiB SMEM/SM, 64K regs, 48 warps, 24 CTAs). Pre-compile SMEM estimates are logged against compiled values.

## Model definitions (fixed now)
- `W = min(P, 142 * occ)`.
- Window shape (generalised for the case where a window spans several groups):
  - `g = min(G, Tm)`, `c = min(Tn, ceil(W / g))`, `r = min(Tm, max(g, W / Tn))` if `c == Tn`, else `r = g`.
- Footprints: `F2 = (r*BM + c*BN) * K * s`, `F3 = (g*BM + N) * K * s`.
- Traffic:
  - A = `M*K*s` if (`F2 <= L2_eff` or `c == Tn`), else `M*K*s * Tn/c`.
  - B = `N*K*s` if `F3 <= L2_eff`, else `N*K*s * Tm/r`.
  - The spec's R0 condition is "roughly A+B fits", so also: if `(M+N)*K*s <= L2_eff`, both are compulsory.
- M1 (naive) uses identical time-model code; only its DRAM bytes = `M*N*K*s*(1/BM + 1/BN)`, i.e. no L2 reuse at all.
- M2 = `estimate_matmul_time` vendored from triton v3.0.0. It doesn't see GROUP_M, so its ties are broken by a fixed-seed random permutation.
- M0 = uniform random ranking, seed 0. Its regret is averaged over 100 seeds.

## H1: traffic model validity (Phase 2; ncu `dram__bytes_read.sum`)
- Data: the K-sweep (M=N=8192, K = 20 log-spaced values in [512, 32768], rounded to a multiple of 128, G in {1,4,8,16,32}) plus the G-sweep, for two configs: 128x128x64/4w/3s, and 128x256x64/8w/3s if it compiles.
- Error = |pred - meas| / meas on DRAM read bytes. Spearman ρ is computed over all (shape, config, G) points pooled.
- **Knee definition.** Normalise bytes by compulsory reads `(M+N)*K*s`. For each predicted regime transition, the measured knee is the smallest swept K at which the measured ratio exceeds the geometric mean of the two predicted regime plateau ratios. A transition is "absent" if this never happens within the sweep.
- **Pass**: median abs % error ≤ 25%, ρ ≥ 0.85, and every predicted knee inside the sweep range has a measured knee within 2x of predicted K\*.
- **Kill**: ρ < 0.7, or a predicted knee is absent, or one is off by > 4x.
- Anything in between is reported as "weak/partial".
- If H1 fails mainly because of the window/dispatch-order assumption (ordering looks right but magnitudes are wrong), the persistent-kernel variant is tested before declaring a kill, as the spec requires.

## H2: relevance (Phase 3)
- Near-ridge set: shapes whose M3-predicted intensity `2MNK / bytes` (best-M3 config) lies in [0.5, 2] × ridge.
- Noise floor per shape = (max - min) / median over 10 re-timings of one config.
- **Pass**: on ≥ 20% of near-ridge shapes, either:
  - the best G for the oracle-best tile beats G=8 for that tile by ≥ 5% (and by more than the noise floor), or
  - the oracle-best config's tile differs from the best tile among G=8 configs (with a time gap ≥ 5%).
- **Kill**: no near-ridge (or any realistic) shape shows a G-induced time difference > noise + 3%.
- If the near-ridge set is empty, H2 is evaluated on all test shapes and that is stated.

## H3: pruning value (Phase 3)
- Regret = best-in-top-k time / oracle time (spec writes the inverse; same meaning). k ∈ {1, 3, 5, 10, 20, 5%, 10%}.
- **Pass**: M3 has regret ≤ 1.05 at k = 10% of feasible configs on ≥ 90% of held-out shapes, **and** M3 beats M1 in one of two ways:
  - (a) the smallest k at which ≥ 90% of shapes have regret ≤ 1.05 is ≤ half of M1's, or
  - (b) the mean regret on the near-ridge subset is strictly lower at k = 1, 3, 5, 10 (at least 3 of the 4), with the difference exceeding the mean noise floor.
- **Kill**: M3 and M1 are within noise, i.e. neither (a) nor (b) holds.
- Kendall τ is also reported.

## H4 (not a kill criterion)
- In the R2 G-sweeps, the ncu-measured traffic-minimising G is within one power-of-2 step of `sqrt(W*BN/BM)`.

## Phase 3 protocol
- Configs: the spec grid, filtered by actual compile success, with no spills and compiled SMEM ≤ 99 KiB.
- Timing: do_bench(warmup=10, rep=50, return_mode="median"), ≥ 3 independent reps in randomized config order (seed logged).
- Correctness: max abs error vs fp32 torch reference ≤ 1e-2 × max|ref| + 1e-2 (bf16), or 5e-2 relative for fp8. Failures are discarded and logged.
- If the budget is exceeded, configs are subsampled uniformly at random (seed logged); shapes are never dropped.

---
## Amendments

### Amendment 1: 2026-10-15 ~17:45, before any Phase 2/3 data
1. **Ridge definition.** Both GPUs are hard power-capped at 300 W. Under sustained tensor load, SM clocks are 735–1050 MHz (nvidia-smi: SW Power Cap active) versus 2505 MHz boost, so the published 364 TFLOPS is unreachable.
   - The H2 near-ridge set now uses the **measured ridge** = (max sustained bf16 TFLOPS over all eff-calibration measurements) / HBM_BW. fp8 is handled the same way.
   - The published-peak ridge (455 FLOP/B bf16) is reported as a secondary analysis.
   - eff() still normalises by the published peak; it simply comes out around 0.3.
2. **SMEM.** Triton 3.2 on sm_89 allocates (num_stages-1) pipeline buffers, so the pre-compile SMEM estimate is `(S-1)*(BM+BN)*BK*s`, not `S*(...)`. This was verified against compiled metadata (exact for bf16). Feasibility uses compiled metadata, as already pre-registered.
3. **Model code.**
   - `g = min(G, Tm, W)`. The W term only matters when W < G, which never happens in the test sets; it makes the W=1 unit test well defined.
   - `knee_K` helper bug fixed before any data: the R0→R1 knee is `L2/((g*BM+N)*s)`.
4. **Phase 2 shapes**, chosen from the model before measurement:
   - R1-dominant: (16384,16384,6144), (12288,12288,7168), (6144,16384,8192).
   - R2-dominant: (8192,8192,32768), (16384,16384,24576), (12288,16384,32768).
   - Calibration shapes are avoided.

### Amendment 2: 2026-10-16, written before any round-2 data (follow-up spec: power coupling, SOTA baselines, dynamic M)
**Phase A facts.**
- Power limit is 300 W enforced, equal to the default and max. SM clock default is 2505 MHz, max 3105.
- The NVML energy counter works.
- **Clock locking is not permitted** (no sudo), so the locked-clock arm and H5(3) are skipped.
- All round-2 timing runs on **GPU0**; GPU1 is used only for ncu and non-timing work.

**Common timing rules.**
- "Sustained protocol" for a (shape, config):
  - Run 1 s of warmup, then ≥3 s of back-to-back launches.
  - Inputs rotate over `n = max(1, ceil(256 MiB / (A+B+C bytes)))` buffer sets, so operands are DRAM-cold as in serving. This replaces do_bench's flush, which would add memset power.
  - An NVML side thread samples every 20 ms: power, SM/mem clock, temperature, throttle bits 0x4 (SW power cap) and 0x8 (HW slowdown).
  - The energy counter is read before and after the timed window.
  - Recorded: time per GEMM (CUDA events over the window / iterations), median SM clock, mean power, energy per GEMM.
  - Configs are randomised within a shape, with a 2 s idle cooldown before each config, and the starting temperature is logged.
- **Noise floor.** For tie decisions, round-1 per-shape (max−min)/median is reused; the mean over 42 shapes is 0.092. For Phase B/D/E, a per-shape floor is also measured: the CV of the 3 reps. Differences are paired by rep. 95% CIs are percentile bootstrap (10,000 resamples) over shapes. "Better by more than noise" means the mean paired difference exceeds the noise floor named in the criterion.

**Phase B shape sets (fixed from round-1 data, `runs/r2_shape_classes.csv`).**
- **G-sensitive (11)**, meaning every round-1 rep showed a ≥5% G effect for the oracle tile:
  - fp8: 128x4096x4096, 2048x28672x8192, 512x14336x4096, 512x28672x8192, 512x8192x28672, 8192x14336x4096, 8192x4096x14336
  - bf16: 2048x4096x14336, 512x8192x28672, 8192x14336x4096, 8192x28672x8192
- **Controls (5)** = the 5 non-sensitive shapes with the smallest round-1 G range:
  - fp8: 128x4096x14336, 512x4096x4096, 128x14336x4096
  - bf16: 128x4096x14336, 128x14336x4096
- **Configs** = the round-1 oracle tile (warps/stages included) × G ∈ {1, 4, 8, 12, 16, 32}. 3 repetitions of the whole shape block.
- **Power-model calibration runs (for Phase C)**:
  - Round-1 calibration shapes 8192³ and 16384x8192x8192, bf16 and fp8.
  - Tiles: 128x128x64_w4_s3, 128x256x64_w8_s3, 64x64x64_w4_s3, 32x128x64_w4_s3 (fp8: BK=128 where feasible, else 64), each with G ∈ {1, 8, 32}.
  - Same sustained protocol, 1 rep.
- DRAM bytes per (shape, config) are taken from new ncu runs on GPU1, bytes only, cold L2 as in round 1.

**H5 (power coupling).**
- (1) For each G-sensitive shape: Spearman ρ between per-config mean sustained SM clock (6 G values) and ncu DRAM read+write bytes. Requirement: median over shapes ≤ −0.5.
- (2) Explained fraction = `log(clk_bestG/clk_G8) / log(t_G8/t_bestG)`. bestG is the **round-1 bestG** (independent data, which avoids selection bias), using sustained means over reps. Requirement: median over shapes ≥ 0.5. Shapes where bestG = 8, or where t_G8/t_bestG ≤ 1 (no sustained speedup), are reported and excluded from (2). Their count is reported.
- (3) Skipped: no clock locking.
- **PASS** = (1) and (2). **KILL** = median ρ > −0.2 or median explained fraction < 0.25. Anything else = PARTIAL.

**H6 (energy, descriptive).** Energy per GEMM at round-1 bestG vs G=8, paired by rep.
- PASS if ≥5% lower on ≥50% of G-sensitive shapes (using rep means), and a Wilcoxon signed-rank test on all (shape, rep) paired log-ratios has p < 0.05.

**Phase C: M4.**
- Power model: `P = P0 + α_dtype·FLOP/T + β·Bytes/T`, fit by OLS only on the power-calibration runs plus the Phase B control shapes. Bytes = ncu DRAM read+write for those runs.
- `T_power = (α·FLOP + β·Bytes)/(P_cap − P0)`, with `P_cap = 300 W`.
- `T_comp = M3 compute term with the clock de-throttled`: eff_nt(tile) = median over calibration shapes of `eff × 2505/sm_clk_logged` (round-1 per-row clock).
- `T_dram = M3 bytes/HBM_BW`; T_l2 is kept as in M3.
- `T = max(T_power, T_comp, T_dram, T_l2) + launch`, with bytes from the M3 traffic model.
- **H3'**:
  - Evaluated on the round-1 42-shape timing table (the medians already used).
  - PASS if mean regret of M4 is lower than "M1 with G=8" by more than 0.092 (round-1 mean noise floor) at both k=1 and k=3, and by ≥ 0.05 at k=1. A 95% bootstrap CI is reported.
  - Otherwise M4 is "no better", and Phases D/E use whichever of M3/M4 has the lower mean k=1 regret on the 42 round-1 shapes (decided by this rule, not by later data).

**Phase D: baselines (H7).**
- **Shapes**:
  - The 42 round-1 shapes in their round-1 layouts. bf16 is A@B with B row-major. fp8 was already K-major B, i.e. the A@B.T layout.
  - Plus the 20 bf16 LLM shapes in A@B.T layout (B stored [N,K] contiguous).
  - fp8 has no separate A@B variant: Triton fp8 on sm_89 used the K-major layout in round 1.
- **B1**: `torch.matmul` for bf16. For fp8, `torch._scaled_mm` with unit scales and fp16 output if supported on sm_89; otherwise B1-fp8 is skipped.
- **B2**:
  - Inductor `max-autotune-no-cudagraphs` with `max_autotune_gemm_backends="ATEN,TRITON"`, a fresh cache dir per shape, and a recorded wall time for the first call (tuning + compile).
  - The picked kernel is identified from the autotune log/profiler.
  - fp8 B2 only if Inductor autotunes `_scaled_mm` with Triton on sm_89; otherwise skipped and stated.
- **B3**: triton.autotune over the vendored tutorial's 16-config CUDA list (our kernel, same layout); its pick is recorded.
- **B4**: the round-1 exhaustive oracle config (A@B layouts and fp8). For the new A@B.T bf16 shapes no exhaustive sweep is run. B4 there is a **pseudo-oracle** = the best of the union of all configs benchmarked by any method for that shape, labelled as such.
- **Ours**:
  - Model top-k (k=1,3,5), compiled and benchmarked with do_bench(warmup=5, rep=20); the best is picked. Tuning cost is recorded.
  - eff is reused from round 1 (not recalibrated for the B.T layout). This is a stated threat.
- **Evaluation**: every method's pick is re-timed on GPU0, interleaved in random order within a shape, 3 reps, with the sustained protocol. Phase D/E use a shortened window: warmup 0.5 s, measurement max(1.5 s, 2 × Phase B median clock-settling time), cooldown 1 s.
- **H7**:
  - PASS if Ours(top-5) is ≥ B2 within noise (t_ours/t_B2 ≤ 1 + max(0.03, per-shape rep CV)) on ≥70% of shapes, **and** ≥5% faster than B2 on ≥20% of shapes, **and** Ours' tuning wall time ≤ 1/5 of B2's (median over shapes).
  - KILL if B2 is within noise of B4 on ≥80% of shapes.
  - Comparison to cuBLAS is descriptive.

**Phase E: dynamic M (H8).**
- (N, K) pairs, all A@B.T bf16: (4096, 4096), (14336, 4096), (4096, 14336).
- M grid as in the spec, 21 values.
- Tuning measurements use do_bench(warmup=5, rep=10), 1 rep:
  - (a) Static = exhaustive tuning over all feasible bf16 configs at M=512.
  - (b) Buckets {1–16, 17–128, 129–1024, 1025+}, each exhaustively tuned at midpoints M = 8, 72, 576, 4608.
  - (c) Model top-1 per M, with no benchmarking.
  - (d) Per-M oracle over a **restricted set**: the union of model top-50 at that M, the (a)/(b) picks, and 150 uniformly random feasible configs (seed 0).
  - (e) Inductor with dynamic=True, if it works; otherwise skipped.
- Final times: every pick at every M is re-timed with do_bench(warmup=25, rep=100, L2 flushed), median of 3 interleaved reps.
- **H8** PASS if `(t_b − t_c)/(t_b − t_d) ≥ 0.5` (using geomean over M) on ≥2 of 3 pairs. If t_b ≤ t_d (no gap), that pair counts as N/A.

**Phase G: soft knee.**
- Replace the hard thresholds with the round-1 Phase-1 sequential miss curve, `miss(F/L2_eff)`, by log-linear interpolation (0 below the curve, 1 above).
  - A traffic = `A0·(1 + (Tn/c − 1)·miss(F2/L2))`.
  - B traffic = `B0·(1 + (Tm/r − 1)·miss(F3/L2))`.
  - The "A+B fits" override is kept.
- PASS if the worst-case abs error on Phase-2 + Phase-3-subset points with F2 or F3 within ±10% of L2_eff drops ≥2× vs hard M3, and overall median error ≤7%.
- Soft M3 is **not** substituted into Phases C/D/E: those were frozen before Phase G.

### Amendment 2b: 2026-10-16, before any nvMatmulHeuristics (NVMMH) metric or timing data
- NVMMH 0.1.0.27 (pip) is used with target TRITON, hardware descriptor RTX_6000_ADA, and flag PERF_MODEL_BASED_AUTO_TUNING, with its internal discovery set loaded implicitly.
  - Precision strings: bf16 = "TST"; fp8 = "QQHSH".
  - Layout strings: A@B = NN_ROW_MAJOR; A@B.T and fp8 = NT_ROW_MAJOR.
- Its Python wrapper segfaults in `estimateSiliconMetric`: it passes the target enum where the C API expects a backend handle. Metrics are therefore obtained by calling the documented C function `nvMatmulHeuristicsEstimateSiliconMetricEx` with a TRITON backend handle. Nothing else is changed.
- **Config mapping, ours → NVMMH** (for its byte and L2 predictions on our configs):
  - cta = (BM, BN, BK), loadStages = num_stages.
  - Warp tile = (BM/2, BN/2) for 4 warps and (BM/2, BN/4) for 8 warps; instruction tile 16x8x16 (fp8: 16x8x32).
  - G=1 → (swizzle 1, ctaOrder 0); G>1 → (swizzle G, ctaOrder 1, M-fastest).
- **Config mapping, NVMMH → ours** (for timing its picks):
  - BM, BN, BK = cta tile; num_stages = loadStages; num_warps = (BM/warpM)·(BN/warpN).
  - swizzle 1 → G=1; (ctaOrder 1, swizzle s) → G=s; (ctaOrder 0, swizzle > 1) is N-grouping, which our kernel can't express, so that pick is skipped in favour of the next-ranked one.
  - Configs that fail to compile or spill are skipped the same way.
  - split-k > 1 is skipped (our kernel has no split-k).
- **Phase F accuracy comparison** (descriptive): on the round-1 Phase-2 points and the Phase-3 ncu subset, NVMMH `GMEM_LOAD_BYTES` vs ncu `dram__bytes_read.sum`, and NVMMH `L2_HIT_RATE` vs ncu `lts__t_sector_op_read_hit_rate` (Phase 2) or M3's implied hit rate. Median abs % error and Spearman ρ are reported next to M3.
- **In Phase D**, method "NVMMH-k" (k = 1, 5) = its top mappable configs, benchmarked like Ours-k. It is added to the regret/speedup tables as an additional baseline. H7 is unchanged.

### Amendment 3 / 3b / 4 (rounds 3 and 4): NOT RECOVERED
The original text of Amendments 3 and 3b is lost (this file was recreated on 2026-11-19 from `PREREGISTRATION.md.bak_r2`,
which ends at Amendment 2b). Their thresholds survive only as quoted in `REPORT_3.md`. There is no `REPORT_4.md`,
so Amendment 4 (if round 4 ran) is unknown. Amendment numbering continues at 5 per the round-5 spec.

### Amendment 5: 2026-11-19, written BEFORE any round-5 data (package `tritonsel`, a REIMPLEMENTATION)
**Provenance.** The round-1..3 `model/`, `kernels/`, `env/`, `runs/`, `predictions/`, `common.py` and the unit tests are missing.
`tritonsel` reimplements M3 and the time model from the formulas in the "Model definitions" section above, `REPORT.md` and
`REPORT_3.md`. Nothing is claimed to be bit-identical to the round-1..3 code. Every number below is re-measured.
`triton-locality-killtest.md` (the original spec, "Section 2") is also missing; the "Model definitions" section above is the only
formula source. Gaps filled by me (not recovered): the compute-time term (see R5-T) and Inductor-template transcription (R5-K).

**R5-0 Baseline re-establishment (gate; runs before anything else).**
- Re-measure L2_eff by the Phase-1 double-read rule (above), HBM_BW, L2_BW. Never refit afterwards.
- ncu `dram__bytes_read.sum` (`--cache-control all`, GPU1) for 50 kernels drawn with `random.Random(5).sample` from the round-1
  design: bf16 A@B, M=N=8192, K = the 20 log-spaced values in [512, 32768] (multiples of 128), G in {1,4,8,16,32}, tiles
  128x128x64_w4_s3 and 128x256x64_w8_s3 (K-sweep, 200 points), plus the G-sweep shapes (16384,16384,6144), (12288,12288,7168),
  (6144,16384,8192), (8192,8192,32768), (16384,16384,24576), (12288,16384,32768) with G in {1,2,4,8,12,16,24,32,64}, tile 128x128x64_w4_s3
  (54 points); sample 30 from the K-sweep and 20 from the G-sweep.
- Predictions are written to `predictions/r5_baseline_pred.csv` before ncu runs.
- **Pass**: median |pred - meas| / meas <= 8.4% (round 1 was 5.4%; "a few percent" = +3 points) and Spearman rho >= 0.95.
- **Fail => stop and report to the user; no later step proceeds.**

**R5-K Kernel.** Our own Triton kernel transcribing Inductor's mm / scaled_mm template (GROUP_M tile order, EVEN_K, fp32 accumulate,
`tl.dot`). Round 3 found harness vs real Inductor-compiled kernels within a median 3% (`|log ratio|` 0.030). That check is repeated only
if time permits; otherwise it is a stated threat. All end-to-end/Inductor numbers use the real Inductor.

**R5-C Calibration (sustained protocol of Amendment 2, window shortened to warmup 0.5 s / window 1.5 s / cooldown 1 s).**
- Candidate tile space per family (bf16-AB, bf16-ABT, fp8-ABT): BM in {16,32,64,128,256}, BN in {32,64,128,256}, BK in {16,32,64,128},
  stages in {1,2,3,4,5}, warps in {2,4,8}, G fixed 8; feasible = compiles, 0 spills, SMEM <= 99 KiB, and BM*BN >= 16*32.
  (This is wider than Inductor's stock list, which lacks BK=16 and stage-1 tiles; round 3 showed that gap caused the small-M failure.)
- eff(tile, family) = ideal_time / measured_sustained_time on calibration shape **8192x8192x8192** (bf16 and fp8). Consistency check on
  12288^3 for 20 random tiles per family: median |eff_8192 / eff_12288 - 1| is reported (no threshold).
- Small-M latency constant (Little's law term, R5-T) is fit on calibration shapes only: M in {1, 4, 16, 32} x (N,K) in {(8192,8192),(16384,4096)},
  every feasible tile subset (random 40 per shape, seed 5), bf16-ABT and fp8-ABT. One scalar per dtype, least squares on log time.
- Calibration shapes never appear in the validation set.

**R5-T Time model.** T = max(T_comp, T_dram, T_l2) + launch, with
- T_comp = ceil(P/142) * 2*BM*BN*K / (peak_per_SM * eff), peak_per_SM = published peak / 142 (bf16 364.2, fp8 728.5 TFLOPS).
  (The busiest SM runs ceil(P/142) CTAs sharing the tensor pipe; this replaces an earlier draft that used occupancy waves. Written before any data.)
- T_dram = (A + B + M*N*out_bytes) / (HBM_BW * attain), attain = min(1, inflight / needed) (Little's law), with
  inflight = min(P, 142*occ) * max(stages-1, 1) * (BM+BN)*BK*s  (bytes in flight, whole GPU) and needed = lat * HBM_BW; `lat` is the one fitted constant.
- T_l2 = (A+B traffic over L2) / L2_BW. launch = 4 us (fixed).
Any change to this form after validation data is touched is an amendment AND the held-out result is reported with both versions.

**R5-V Validation (held-out).** Llama 3 8B GEMMs, per dtype in {bf16 (A@B.T), fp8}: (N,K) in {(6144,4096), (4096,4096), (28672,4096), (4096,14336)},
M in {1, 8, 16, 32, 128, 512, 2048, 8192} (32 shapes per dtype). Calibration shapes above are disjoint.
- Reference = real Inductor max-autotune-no-cudagraphs (ATEN,TRITON), pick re-timed by our sustained harness through the real compiled
  Inductor graph (CUDA graph replay, DRAM-cold buffers, 3 interleaved paired reps; per-shape ratio = median over reps).
- Ours top-1 = `tritonsel.inductor.enable(profile, k=1)` through real Inductor.
- Per-M oracle for M <= 32 = best over (model top-20 U Inductor's autotune list U 100 random feasible tiles, seed 5), our harness kernel, sustained, 2 reps. It is a restricted oracle.
- **Criteria (user-fixed, not tuned)**: (i) geomean ours-top1 / Inductor-max-autotune <= 1.03 for bf16 and for fp8 separately;
  (ii) no single shape > 1.15; (iii) for M <= 32 shapes, ours-top1 / per-M oracle <= 1.10 on every such shape (all must hold; fraction also reported).
- One refit of eff/latency constant on **calibration shapes only** is allowed after a failed validation; held-out shapes are never used for fitting.
  If a criterion fails, the failure is reported and work stops for the user's decision.

**R5-S Scope order (user-set, 2026-11-19).** Gate R5-0, calibration, Steps 1-2 (package, fixes), then Step 4 (eval harness, descriptive only),
then Step 3 (tritonBLAS-style baseline) only if time remains. Step 4 outcomes have no pass/fail thresholds.

### Amendment 5a: 2026-11-19, before any round-5 data (counters blocked; killtest spec found)
1. **ncu is unavailable** (`ERR_NVGPUCTRPERM`, `RmProfilingAdminOnly: 1`, driver 580.196.07; no sudo). ECN has been asked to enable counters.
   **R5-0 stands unchanged and is NOT substituted.** It is pending; no step that the user marked as dependent on it is declared validated until it runs.
2. **L2_eff** is taken as a documented constant from round 1 (`REPORT.md`: 0% second-pass misses up to 88 MiB, knee at about 102 MiB):
   L2_eff = 102 MiB, labelled `source: round1 REPORT.md` in the profile. Only the sequential knee is quoted there, so the "min(seq, strided)" rule is not re-applied.
3. **Secondary sanity check** (timing-based, never a replacement): double-read probe timing, knee where the double/single time-ratio crosses halfway
   between its plateaus. Reported next to the constant; no threshold. A disagreement is reported, not tuned away.
4. `triton-locality-killtest.md` is now on disk; its Section 2 is the model source. Differences from R5-T of Amendment 5, all stated:
   - Section 2 writes T_comp = ceil(P/W) * 2BMBNK / (per_SM_peak * occ_share * eff). R5-T's ceil(P/142) * 2BMBNK/(peak_sm*eff) is the same
     formula with the partial last wave charged exactly (full waves cost occ shares each, the last wave ceil(rem/142)); identical when P is a multiple of W.
   - Section 2's T_l2 = M*N*K*s*(1/BM+1/BN)/L2_BW is used as is (mathematically equal to what the code computes).
   - Regime/window definitions: Section 2 and the "Model definitions" section above agree except the amended window extension (c == Tn), which is kept.
   - The Little's-law attainable-bandwidth term is NOT in Section 2; it is the new Step-2 term from the round-5 spec (R5-T).
5. Unit tests (from Section 5): reads >= compulsory and == compulsory in R0; monotone in L2; M3 == M1 when W = 1 (modulo the stated M1 definition).

### Amendment 5b: 2026-11-20, before any data on the new shapes (fp8 small-M tie-break)
**Why.** R5-V failed for fp8 (worst 1.834, M<=32 within 1.10 on 11/16). Post-hoc diagnosis on those same shapes: in the DRAM-bound small-M regime
the model scores hundreds of tiles at exactly the DRAM roofline, so top-1 is decided by enumeration order. The R5-V 8B shapes are now SEEN and
are not held-out for this change.
**Rule (frozen text; code must match it exactly).** Let t* = lowest predicted time over all (tile, G) candidates. If the best candidate's binding
term is T_dram (t_dram >= t_comp and >= t_l2), define the tie set S = all candidates with predicted time <= 1.02 * t* **and** binding term T_dram.
Pick from S by: (1) smallest BM; (2) then largest bytes in flight inflight = min(P,142*occ)*max(stages-1,1)*(BM+BN)*BK*s; (3) then lowest predicted time;
(4) then enumeration order (G=8 first). If the best candidate is not DRAM-bound the rule does nothing (plain argmin). Top-k for k>1 takes the rule's
pick first, then the remaining candidates in plain predicted order. epsilon=0.02, the two sort keys and their order are fixed now and are not tuned.
The time model itself, eff tables, lat constants and L2_eff are unchanged.
**Fresh held-out set.** Llama 3 70B single-GPU shards at TP=8: (N,K) in {qkv (1280,8192), o (8192,1024), gate_up (7168,8192), down (8192,3584)},
M in {1,8,16,32,128,512,2048,8192}, bf16 (A@B.T) and fp8: 64 shapes. Never used for calibration or for choosing this rule.
(Mistral 7B was rejected as the fresh set: its GEMMs are identical to Llama 3 8B's.)
**Criteria** = R5-V (i)-(iii) verbatim, per dtype, vs real Inductor max-autotune and the same restricted per-M oracle, on the fresh set.
Reported side by side: plain argmin (Amendment 5 selector) and tie-break selector on the same run. The rule applies to both dtypes (it is not fp8-specific);
a bf16 regression on the fresh set counts against it. The 8B shapes are re-run only as a labelled non-held-out diagnostic.
**Decision rule.** The tie-break becomes the default only if it passes (i)-(iii) on the fresh set for the dtypes where it is enabled; otherwise it stays off
and the failure is reported. No second tweak of epsilon/keys on these shapes: a further change needs Amendment 5c and another fresh set.

### Amendment 5c: 2026-11-20, before any data on the new shapes (top-k = k distinct tiles)
**Why.** The documented fallback "benchmark top 5 for M <= 32" changed almost nothing in R5-V (fp8 worst stayed 1.81), because the model's top 5 were the same
tile at G in {8,16,32,4,1}: nothing different for Inductor to try. (Diagnosis post hoc, on seen shapes.)
**Rule (frozen).** For k > 1, `select`/Inductor top-k returns the k best **distinct tiles** (distinct (BM,BN,BK,num_warps,num_stages)), each at its best G
(lowest predicted time; exact ties resolve to the first G in (8,16,32,4,1)), ordered by predicted time. k = 1 is unchanged (so top-1 results are identical).
The 5b tie-break stays OFF for this validation (plain argmin ranking). Implemented behind `LALA_TRITON_DISTINCT=1` / `select_scored(..., distinct=True)`,
default OFF until validated.
**Fresh held-out set (never used for calibration, for the 5b rule, or for choosing this rule).** Qwen 2.5 7B linear layers
(hidden 3584, intermediate 18944, 28 heads, 4 KV heads, head_dim 128; biases ignored): (N,K) in {qkv (4608,3584), o (3584,3584), gate_up (37888,3584), down (3584,18944)},
M in {1, 8, 32, 128, 512, 2048, 8192}: 28 shapes per dtype. fp8 is the primary test; bf16 is run too and reported with the same criteria as a regression check.
(Mistral 7B was rejected: its linear shapes are identical to Llama 3 8B's, which are already seen.)
**Methods, same run, 3 interleaved reps, medians:** ours top-5 distinct (new), ours top-5 variants (the old rule), ours top-1, real Inductor max-autotune (ATEN,TRITON),
restricted per-M oracle for M <= 32 (model top-20 U 5 distinct U 100 random, seed 5, G=8, sustained).
**Criteria for the new top-5 (R5-V form):** (i) geomean new-top-5 / stock max-autotune <= 1.03; (ii) no shape worse than 1.15; (iii) for M <= 32, within 1.10 of the oracle on every shape.
Benchmarking cost of the 5 candidates is reported (tuning-time) but not part of the pass/fail. 
**Decision rule.** Becomes the default for k>1 only if (i)-(iii) all hold for fp8; otherwise stays off and the failure is reported. No re-tuning of k or the rule on these shapes
(a change needs Amendment 5d and another fresh set). The old top-5 and top-1 on the same shapes are reported side by side.

### Amendment 5d: 2026-11-23, before any data on the new shapes (BLOCK_M cap + padded-row charge)
**Why.** Dongyoon's L40S run (reported to me by the user; the data is not in this repo and is not used for any fit) shows the fp8 small-M roofline tie picking
BLOCK_M=128 at M=1 and M=8, 7.5x slower on `o` at M=1. Same mechanism as the 5b diagnosis: in the DRAM-bound small-M regime many tiles sit at exactly the roofline, so the
winner is decided by enumeration order, and a 128-row tile at M=1 loads and multiplies 127 padded rows. The 5b tie-break (OFF, failed fp8 on fresh shapes) only *prefers* small BM
inside a 2% tie set; 5d instead removes oversized tiles from the candidate set and makes the padding visible to the model. (The L40S is the same AD102 die as our RTX 6000 Ada, 142 SMs.)
**Rule (frozen text; code must match it exactly). One switch (`LALA_TRITON_PADCAP=1`, `select_scored(..., padcap=True)`), default OFF until validated, covers both parts:**
1. **BM cap.** Before ranking, drop every candidate tile with BM > max(16, next_pow2(M)). So BM=16 only for M in 1..16, BM <= 32 for M in 17..32, <= 128 for M=128; for M >= 256 the cap
   does not bind (largest calibrated BM is 256). Applies to all G, both dtypes, top-1 and top-k, and to the Inductor hook (M = the size hint). If no tile survives, raise.
2. **Padded-row charge.** M_pad = ceil(M/BM)*BM replaces M wherever the model charges A: (a) A0 = M_pad*K*s in `traffic()` (so the R0 compulsory test is A0+B0 and A's DRAM bytes use A0), and
   (b) the A term of T_l2: M_pad*K*s*ceil(N/BN). **T_comp is unchanged**: it is already ceil(P/142)*2*BM*BN*K per tile, i.e. whole padded tiles (I checked the code before writing this; a unit test asserts the
   flag does not change T_comp). Output bytes M*N*so, the footprints F2/F3 (already whole tiles) and everything else are unchanged. Stated approximation: Inductor's mm template wraps row indices modulo M, so padded
   rows re-read real rows (cache hits, not extra DRAM); charging them as DRAM bytes is therefore conservative. It is kept because it is the user-specified model.
Unchanged: time model otherwise, eff tables, lat constants, L2_eff, the profile file (`profiles/rtx6000ada.json`, sha256 prefix 48bce33124722fbd; no refit), 5b tie-break and 5c distinct top-k (both OFF).
**What to expect, stated before data so a pass is not over-read.** For every M in the grid (all powers of two) the cap alone removes all padding for M >= 16, and at M=1 and 8 every surviving tile has BM=16, so part 2 can only re-rank
BM=16 tiles (through the BN-dependent L2 A term and a constant DRAM offset). On this set 5d is therefore mostly the cap; part 2 matters for non-power-of-two M. An offline, untimed diagnostic (picks of plain vs cap-only vs cap+pad on the fresh shapes)
is reported, computed after this amendment is committed. Also: fp8 has failed (ii)/(iii) on three earlier sets partly for reasons the cap does not touch (e.g. at M=1 the oracle prefers fewer stages and more warps than our BM=16 pick;
`qkv` mid-M), so a fp8 fail is a live outcome.
**Fresh held-out set.** Gemma 2 9B linear layers (hidden 3584, intermediate 14336, 16 heads, 8 KV heads, head_dim 256; biases ignored; o-proj K = 16*256 = 4096, not hidden):
(N,K) in {qkv (8192,3584), o (3584,4096), gate_up (28672,3584), down (3584,14336)}, M in {1, 8, 32, 128, 512, 2048, 8192}, bf16 (A@B.T) and fp8: 28 shapes per dtype, 56 total.
Never used for calibration or for choosing this rule. Mistral 7B was again rejected (its GEMMs equal Llama 3 8B's). No (N,K) here appears in the calibration set, R5-V (Llama 3 8B), 5b (70B TP=8) or 5c (Qwen 2.5 7B);
three of four share K=3584 with Qwen 2.5 7B but with different N, so the set is fresh by shape, not unrelated by dimension.
**Methods, same run, 3 interleaved reps, medians, sustained window 0.5/1.5/1.0 s, 2 GPUs:** ours top-1 plain (Amendment 5 selector), ours top-1 with 5d, real Inductor max-autotune (ATEN,TRITON),
restricted per-M oracle for M <= 32 (24 shapes): plain top-20 U 5b top-5 U 5c distinct-5 U 5d top-20 U 100 random uncapped tiles (seed 5, G=8), sustained. (The pool is larger than in 5c, so this oracle is no easier to match.)
**Criteria (R5-V verbatim, per dtype, for ours top-1 with 5d vs stock max-autotune and the oracle):** (i) geomean <= 1.03; (ii) no single shape > 1.15; (iii) every M <= 32 shape within 1.10 of the oracle.
Plain top-1 is reported side by side on the same run with the same criteria (not a criterion for the decision).
**Decision rule.** 5d becomes the default only if (i)-(iii) all hold for **both** dtypes (the cap is dtype-agnostic; a bf16 regression counts against it). Otherwise it stays OFF and the failure is reported.
No second tweak on these shapes: a further change needs Amendment 5e and another fresh set.

### Amendments 5e, 5f, 5g: 2026-11-23, written before ANY validation timing (small-M / decode selection, Round 6)
**Why.** Remaining fp8 failures (and some bf16 picks) at M = 1..64 come from the model predicting an identical DRAM-roofline time for most BM=16 tiles, so the winner is decided by enumeration order
(5b diagnosis; 5d only removed oversized BM). **Step 1 evidence (calibration shapes only; never validation shapes).** I timed every feasible tile with BM <= min(64, 5d cap) (G=8, ABT) on synthetic (N,K) in {2048x2048, 6144x6144, 12288x3072, 3072x12288}, M in {1,2,4,8,16,32,64},
bf16 and fp8: 6952 timings, both GPUs, `results/r6/decode_sweep_s*.csv`. **These used SHORT diagnosis timing (0.15 s warmup / 0.3 s window / no cooldown, DRAM-cold rotating buffers, CUDA-graph replay), NOT the full sustained protocol** (tile ranking, not absolute speed, is what they are used for).
Findings that fix the forms below: the current model predicts the same time for most BM=16 tiles (e.g. bf16 M=1 12288x3072: 0.0987 ms for all; measured 0.087 to 0.32 ms); BK (K-loop iterations) is the strongest single factor (R^2 of within-shape log time 0.54 bf16, 0.72 fp8), bytes in flight next (~0.4);
warps, BN and block count explain almost nothing alone; per-iteration latency is divided by pipeline depth, and num_stages=1 and 2 behave the same, so depth = max(stages-1, 1), saturating near 2-3.
A first analytical form of 5e (full achieved-bandwidth model with 3 params fitted to pooled RMS log error) improved RMS (0.73 -> 0.39) but made picks WORSE (median pick/best 1.17 vs 1.03) and was **dropped**; the form below was fitted for ranking instead.

**Common to 5e/5f/5g.** Each acts only for M <= 64 (5g also acts at larger M, see below). Each keeps 5d (BM cap + padded-row charge) ON. Plain model otherwise unchanged. Profile: `profiles/rtx6000ada.json` (sha256 prefix 48bce33124722fbd, unchanged) plus the 5f table in `profiles/rtx6000ada_r6.json` (prefix bcdbecf12b854466).
Opt-in switches, all default OFF: `LALA_TRITON_DECODE5E=1`, `LALA_TRITON_DECODE5F=1`, `LALA_TRITON_AUTO_EPS=0.03`.

#### 5e: K-loop latency floor
**Rule (frozen).** For M <= 64: T = max(T_current, waves * ceil(K/BK) * t_iter / min(max(stages-1, 1), d_sat)), waves = ceil(blocks / (SMs * occ)), blocks = ceil(M/BM)*ceil(N/BN). (Deviation from the first sketch, which used min(stages, d_sat): the data show stages 1 and 2 are identical, so depth is stages-1.)
**Fitted parameters (2 per dtype, calibration only, grid search; objective = mean log of expected pick/best over the model's tie group, minus 0.1 * median Kendall tau on near-best tiles (measured <= 1.3x best)):**
bf16 t_iter = 4.4 us, d_sat = 3; fp8 t_iter = 4.6 us, d_sat = 3 (`lala_triton/params_5e.json`, `eval/fit_5e.py`). Note t_iter is a ranking constant, not a latency: with it the 5e time is a poor ABSOLUTE predictor (RMS log error got worse than current), and is used only to rank.
**Calibration result (28 shapes/dtype; the same data it was fitted on, so optimistic):** median pick/best bf16 1.026 -> 1.009, fp8 1.021 -> 1.017; mean 1.035 -> 1.019 and 1.073 -> 1.048; worst 1.17 -> 1.11 and 1.42 -> 1.28; median Kendall tau on near-best tiles 0.59 -> 0.69 and 0.52 -> 0.62. It beat the current model on median pick and tau, so it is registered.
Offline check, no timing: 5e/5f/5e+5f change 0 of 96 picks at M >= 128 versus 5d on four seen models' shapes (and 0 of 96 on the fresh shapes), so large-M behaviour is unchanged by construction.

#### 5f: measured decode table
**Rule (frozen).** `python3 -m lala_triton.calibrate --decode-table --out <profile> --shard i/n` (then `--build-only`) times every feasible tile (BM <= min(64, 5d cap)) on the 4 calibration (N,K) x 7 M grid above (short diagnosis timing), and stores per (dtype, tile)
log(BW) = a + b*log(active) + c*log(kit), BW = DRAM bytes / time, active = min(blocks, SMs*occ), kit = ceil(K/BK), ridge-fitted (lambda 0.3 on b, c), with the measured range. For M <= 64 and a tile in the table the time is bytes/BW.
Active above the table's range is clamped; active below range - 0.4 (log) or kit outside range +- 0.4 (log) or an unseen tile -> fall back and count it (`lala_triton.decode.FALLBACK`). Fallback is the current model (**ours1f**) or 5e (**ours1ef**).
**Calibration check (leave-one-(N,K)-out over the four calibration sizes, 50% of predictions fell back because every held-out size is at an edge of the other three):**
5f(+5e fallback) vs 5e: bf16 median pick 1.008 vs 1.009, worst 1.22 vs 1.11, tau 0.62 vs 0.69; fp8 median 1.009 vs 1.017, worst 1.34 vs 1.28, tau 0.60 vs 0.62. I.e. NOT clearly better than 5e on calibration; it is registered as a test arm because the spec asks for it, with that stated up front.
Cost recorded: calibration about 37 min per GPU on two GPUs (about 75 min on one), profile 132 KB -> 254 KB (indent-2 JSON, 485 tile entries).

#### 5g: benchmark only when unsure
**Rule (frozen).** On top of 5d + 5e (the model that did best on calibration). Rank all tiles; take distinct tiles at their best G. If no other distinct tile is predicted within eps = 3% of the best -> top-1, zero benchmarking.
Otherwise return up to 5 distinct tiles from the tied group (greedy max-min diversity over (BM,BN,BK,warps,stages); first pick = model best; ties -> better prediction), which Inductor benchmarks. Applies at every M. eps is NOT tuned.
**Expected, stated before data (offline, seen models only):** at M <= 32 the tied group has >= 2 tiles for every shape at any eps in 0.3%..3% (100% of shapes), so 5g will benchmark every decode shape; over the whole M grid the benchmark fraction is 0.48 (0.3%), 0.59 (1%), 0.77 (2%), 0.80 (3%).
So the question 5g answers is cost vs. stock autotuning (5 candidates instead of the stock list) and quality, not "near-zero tuning".

#### Fresh validation set (never used before in this project; values from the published Hugging Face config.json, fetched 2026-11-23, not from memory)
- **Mistral NeMo 12B** (mistralai/Mistral-Nemo-Base-2407): hidden 5120, intermediate 14336, 32 heads, 8 KV heads, head_dim 128 (explicit in config). (N,K): qkv (6144,5120), o (5120,4096) [K = 32*128, not hidden], gate_up (28672,5120), down (5120,14336).
- **Phi-3 medium** (microsoft/Phi-3-medium-4k-instruct): hidden 5120, intermediate 17920, 40 heads, 10 KV heads, no head_dim key so 5120/40 = 128. (N,K): qkv (7680,5120), o (5120,5120), gate_up (35840,5120), down (5120,17920).
M in {1, 8, 32, 128, 512, 2048, 8192}; bf16 (A@B.T) and fp8: 2 models x 4 GEMMs x 7 M x 2 dtypes = 112 shapes. No (N,K) here is a calibration size; the shared dimensions with seen models (28672 with Gemma gate_up, 5120/14336 with Llama/Gemma) are paired with different partners.
Offline predictions of every selector variant, written before any timing: `results/r6/predictions_fresh.csv` (`eval/offline_r6_picks.py`).
**Methods (all in the SAME run, interleaved, 3 reps, medians, sustained window 0.5/1.5/1.0 s, capture-after-compile harness):** ind_auto, cublas, ours1, ours1pc (5d), ours1e (5d+5e), ours1f (5d+5f, falls back to current), ours1ef (5d+5e+5f), ours5d (5c distinct top-5), ours_auto (5d+5e+5g), tritonblas, oracle at M in {1, 8, 32, 128}.
ours1e/ours1f/ours1ef are measured at M <= 64 only (M=1, 8, 32 here); at M >= 128 their picks equal ours1pc's (checked offline above). The restricted oracle pool is the previous one (plain top-20, 5b, 5c, 5d top-20, 100 random tiles seed 5, G=8) PLUS the 5e, 5f and 5e+5f top-20, so it cannot be easier than before. Every timed pool candidate is logged with its predicted time under each model (`*_oraclepool.csv`) for the Kendall tau diagnostic.
**Criteria, per dtype, for EACH of 5e, 5f(+5e fallback), 5g vs stock max-autotune and the oracle (R5-V verbatim):** (i) geomean <= 1.03; (ii) no shape > 1.15; (iii) every M <= 32 shape within 1.10 of the oracle.
**Added criteria:** (iv) no regression vs 5d: bf16 geomean vs ind_auto no more than 0.01 worse than ours1pc's and no bf16 shape where the method is > 1.05 x ours1pc with a different pick; (v) median Kendall tau (predicted vs measured over the oracle pool, M <= 32, per shape) at least ours1pc's... reported as a diagnostic, not pass/fail.
For 5g additionally: benchmark fraction and tuning time (first-call seconds, geomean over shapes) are reported against stock max-autotune; **registered threshold (vi): geomean 5g first-call time <= 0.75 x stock's**. A failing (vi) with passing (i)-(iii) is reported as "quality yes, cost no".
**Decision rule.** Each of 5e, 5f, 5g becomes a recommended default only if (i)-(iv) hold for BOTH dtypes (5g also (vi)). Otherwise it stays OFF and the failure is reported. If both pass, the simplest (5e, then 5g) is recommended. No re-tuning on these shapes.
**Honest limits registered up front:** one GPU model (RTX 6000 Ada, power-capped), 4 calibration (N,K) sizes for 5f (thin), diagnosis-grade timing for 5e/5f fitting, no Hopper, no real tritonBLAS code, no end-to-end serving test.
