# Asymptotic L2 Locality for Triton Tile Selection: Kill Test Spec

Owner: Jack. Executor: Claude Code on the GPU server. Target budget: about one day of wall-clock, with early exits.

Keep this spec and all results in a private working directory (e.g. `~/lala-triton-killtest/`). Do not commit planning notes or this spec to any public repo.

When reporting, explain what was run, what was found, and the verdict for each hypothesis in plain language. Do not run off into new directions without saying why.

---

## 0. TL;DR

**Idea.** Combine two papers:
- **LALA**: symbolic, asymptotic miss-ratio analysis via a symbolic Denning recursion. The miss ratio is a step function of cache size, with knees at polynomial footprints.
- **LADM** (Khairy, Nikiforov, Nellans, Rogers, MICRO 2020): threadblock-centric "datablock" index analysis. On GPUs, global-memory reuse is almost entirely inter-threadblock through L2, and which threadblocks co-run decides it.

Together these give a cheap, closed-form model of **HBM traffic as a function of Triton config** (BLOCK_M/N/K, GROUP_M, num_stages, occupancy) and problem shape. Use it to prune Triton autotuning search spaces and to pick program-ID ordering.

**What the kill test must answer, in order:**
1. **H1 (validity):** Does the model predict measured DRAM bytes, including the predicted knees?
2. **H2 (relevance):** On this GPU, does L2 locality actually move runtime for realistic shapes?
3. **H3 (value):** Does the locality-aware model prune better than a naive per-tile traffic model plus roofline?

If H1 fails, the model is wrong. If H1 holds but H2 fails, the model is right but useless for tuning. If H3 fails, the naive model already suffices. Any of these is a clean kill.

---

## 1. What we take from each paper

### LALA (Loop Asymptotic Locality Analysis)

**Mechanism.** Start from a symbolic reuse-interval (RI) distribution `{(ri_i, p_i)}` sorted by RI. The miss ratio at timescale `ri_i` is `m(ri_i) = sum_{j>i} p_j`. The cache size is `c(ri_i) = c(ri_{i-1}) + m(ri_{i-1}) * (ri_i - ri_{i-1})`, and `mr(c(ri_i)) = m(ri_i)`. "Infinite Repeat" turns cold misses into "imaginary" RIs, so the distribution is closed.

**What is useful for us:**
- **The miss-ratio curve is a step function, and knee locations are polynomials in loop bounds.** Between knees, the miss ratio is flat. This is exactly the structure we want for a cheap model: decide which regime you are in, then evaluate a closed form.
- **Asymptotic simplification:** keep leading terms only and merge RIs with the same leading term. This yields formulas small enough to evaluate per config in microseconds.
- **Symbolic cache-block size `b`.** On GPUs, the analog is the sector/line size, and, more importantly, the tile.
- **The symbolic RI test** (`sum ri * P(ri) == data size` under Infinite Repeat) is a free sanity check. Use it as a unit test on any derived distribution.

**Weaknesses that shape the design:**
- **RI derivation is manual.** The "linear-time" claim covers only the recursion, not the derivation, and the derivation is the hard part.
- **The symbolic path breaks when RI ordering depends on parameters.** Convolution (Sec. 4.2) falls back to numeric evaluation. With GPU knobs (GROUP_M, occupancy, L2 size), knee ordering will depend on parameters. Design consequence: **symbolic within each regime, numeric regime selection**, evaluated per config. Do not chase a single fully symbolic formula.
- **Cache assumptions don't match GPU L2.** It uses a working-set (average cache size) approximation of fully associative LRU and assumes sequential execution. GPU L2 is sectored, hashed, partitioned, and hit by many concurrent CTAs. Effective capacity must be calibrated, not assumed.
- **The tiling section (Sec. 6) still enumerates tile sizes** rather than solving. The tiled loop steps `k` by `t_m`, which is an unexplained constraint or a bug. Most of the matmul results are not validated against simulation in the paper; only convolution is (Fig. 3).
- **The asymptotic-equivalence proof (Thm 2.2) is informal** in Case 2. That's fine for us, since we validate empirically.

### LADM (Locality-Aware Data Management)

**Mechanism.** Expand each global index into "prime" variables (thread IDs, block IDs, grid/block dims, loop induction variable). Split it into a loop-invariant part (which datablock a threadblock starts at) and a loop-variant part (threadblock motion). Then classify each access as no-locality (with stride), row/column-shared (horizontally or vertically), or intra-thread. Finally, co-schedule threadblocks and place data so that sharers land on the same node, and choose a cache insertion policy (cache-remote-once vs cache-remote-twice) per locality class.

**What is useful for us:**
- **GPU reuse is inter-CTA, through L2.** Intra-tile reuse is absorbed by shared memory/registers, so the locality question is *which CTAs co-run and in what order*. Per-CTA LRU modeling is the wrong level.
- **A "datablock" is exactly a Triton tile load per K-iteration.** In Triton, offsets are explicit (`pid_m * BLOCK_M + tl.arange(...)`, plus `k * BLOCK_K`), so LADM's index analysis is nearly free at the Triton IR level. In CUDA source it required backward substitution.
- **The sequentialization trick:** turn grid dimensions into outer loops, then apply sequential analysis. This is how we feed a LALA-style analysis, with one GPU-specific change: interleave a *window* of concurrently resident CTAs rather than running them one after another (Sec. 2).
- **LADM's tie-break** ("bind the schedule to the larger data structure") is a qualitative heuristic. A quantitative traffic model should subsume it.
- **The cache insertion policy has a direct Triton analog:** `tl.load(..., eviction_policy="evict_last" | "evict_first")`.

**Weaknesses:**
- It is categorical only, with no capacity model. It cannot say *how much* traffic a choice saves or where behavior flips.
- It is evaluated mostly in simulation, with hand-implemented hardware validation.

**Why this is timely.** LADM's NUMA scenario now exists in production: B200 is two dies, and MI300X has 8 XCDs with private L2s and round-robin workgroup dispatch. Even H100 L2 is split into two partitions. That is a later extension; the kill test targets a single GPU's L2.

---

## 2. The model (GPU-adapted LALA)

### Notation
- Problem: `C[M,N] = A[M,K] @ B[K,N]`. Element size `s` bytes (2 for bf16/fp16, 1 for fp8).
- Config: `BM, BN, BK, G` (GROUP_M), `num_warps`, `num_stages`.
- Tile counts: `Tm = ceil(M/BM)`, `Tn = ceil(N/BN)`. Grid size `P = Tm * Tn`.
- Occupancy `occ` = CTAs per SM, limited by SMEM and registers. `SMEM ≈ num_stages * (BM*BK + BK*BN) * s` (verify against compiled metadata).
- Concurrent window `W = min(P, num_SMs * occ)`.
- `L2_eff` = effective L2 capacity, **calibrated once per GPU by microbenchmark (Phase 1), never fit per shape.**

### Level 0: on-chip (not a cache model, a feasibility filter)
SMEM and register limits are deterministic. Configs exceeding SMEM, or with predicted register spills, are infeasible and pruned outright. Record pre-compile estimates vs post-compile `n_regs`, `n_spills`, `metadata.shared` to check how good the pre-compile filter is. The point of pruning is to avoid compiling, so the model must work pre-compile.

### Level 1: L2, via a sequentialized concurrent window
Triton's grouped ordering (tutorial matmul) makes `pid_m` the fastest index within a group of `G` tile-rows. Assuming hardware dispatches CTAs roughly in pid order, a window of `W` consecutive pids covers:
- `r = min(G, W, Tm)` tile-rows
- `c = ceil(W / r)` tile-columns (capped at `Tn`)

**Reuse structure (lockstep-wave approximation):**
1. **Within a wave, same k-step.** Each A block is wanted by `c` CTAs and each B block by `r` CTAs. Reuse footprint is `F1 = (r*BM + c*BN) * BK * s`, which is hundreds of KB and always fits. So each wave reads its super-tile once: `(r*BM + c*BN) * K * s`. CTA staggering of `δ` k-steps multiplies `F1` by roughly `(1+δ)`; still small.
2. **Across waves, same group (A panel reuse).** The next wave keeps the same rows and moves to new columns. Reuse footprint is one full wave's unique traffic: `F2 = (r*BM + c*BN) * K * s`.
3. **Across groups (B panel reuse).** The next group revisits the same B columns. Reuse footprint is `F3 ≈ (G*BM + N) * K * s`.

Since `F1 < F2 ≤ F3`, there are three regimes, i.e. LALA's step function with knees at polynomial footprints:

| Regime | Condition | A read traffic | B read traffic |
|---|---|---|---|
| R0: fits | `F3 ≤ L2_eff` (roughly A+B fits) | `M*K*s` | `N*K*s` |
| R1: A panels reused | `F2 ≤ L2_eff < F3` | `M*K*s` | `N*K*s * Tm/r` |
| R2: streaming | `F2 > L2_eff` | `M*K*s * Tn/c` | `N*K*s * Tm/r` |

Writes are `M*N*s_out` in all regimes. Handle the partial last wave and edge tiles exactly in code rather than asymptotically.

**Closed-form predictions to test:**
- **Streaming regime (R2)** total reads `≈ M*N*K*s * (1/(c*BN) + 1/(r*BM))` with `r*c = W`. This is the per-tile formula with an effective super-tile of `(r*BM) x (c*BN)`. It is minimized at **`r* ≈ sqrt(W * BN / BM)`**, so the traffic-optimal GROUP_M is about `sqrt(W)` for square tiles (≈ 16 at W = 264, versus the tutorial default of 8).
- **Regime R1:** A is free, so larger `r` (larger G) is strictly better for B traffic. **The optimal G flips between regimes at `K* = L2_eff / ((r*BM + c*BN) * s)`.** For H100-like numbers (bf16, BM=BN=128, G=8, W≈264), this lands in the low thousands to ~10k, i.e. realistic LLM shapes.
- **G=1** gives a huge B refetch factor (`Tm`) unless B fits. It should be visibly bandwidth-bound for large shapes. This is a sanity check, not a novel result.

### Time model (deliberately simple, so locality is the only new ingredient)
```
T = max( T_comp, T_dram, T_l2 ) + launch_overhead

T_comp = ceil(P / W) * (2*BM*BN*K) / (per_SM_peak * occ_share * eff(config))
T_dram = model_bytes / HBM_BW
T_l2   = M*N*K*s*(1/BM + 1/BN) / L2_BW
```
- `ceil(P/W)` captures wave quantization.
- `eff(config)` is a per-config compute-efficiency table **calibrated on a disjoint set of calibration shapes** and then held fixed.
- `HBM_BW` and `L2_BW` come from microbenchmarks.

### Baselines (required, otherwise results mean nothing)
- **M0:** random top-k.
- **M1:** naive per-tile traffic `M*N*K*s*(1/BM + 1/BN)` + roofline + wave quantization + the same `eff` table. This is identical to M3 except for the L2 regime model.
- **M2:** Triton's old analytical model (`triton.ops.matmul_perf_model.estimate_matmul_time`), if available in the installed Triton. If not, vendor it from an older release or note that it was skipped.
- **M3:** the proposed L2-aware model above.

M3 vs M1 is the ablation that isolates the contribution of asymptotic locality.

---

## 3. Hypotheses and pre-registered kill criteria

Write all thresholds into `PREREGISTRATION.md` with a timestamp **before** collecting the relevant data. Write model predictions to CSV **before** running the corresponding measurements.

**H1: traffic model validity** (Phase 2; uses ncu DRAM bytes, not time)
- Pass: across the K-sweep and G-sweep, the median absolute % error of predicted DRAM read bytes is ≤ 25%, Spearman ρ ≥ 0.85, and the measured knee in K lies within 2x of predicted `K*`.
- Kill: ρ < 0.7, or knees absent or off by > 4x. Stop and report.

**H2: relevance** (Phase 3)
- Restrict to shapes the model classifies as near the ridge point (predicted intensity within 0.5x to 2x of `peak_FLOPs / HBM_BW`).
- Pass: on ≥ 20% of those shapes, the best G for the best tile beats G=8 by ≥ 5% in time, *or* the best tile overall changes when locality is accounted for.
- Kill: L2 locality never moves runtime by more than noise + 3% on realistic shapes. The model may be correct but is not a tuning lever on this GPU. Report it as such.

**H3: pruning value** (Phase 3)
- Pass: M3 reaches regret ≤ 1.05 (oracle time / best-in-top-k time) with top-k ≤ 10% of feasible configs on ≥ 90% of held-out shapes. **And** M3 beats M1, either by needing ≤ half the k for the same regret, or by strictly lower regret on the near-ridge subset.
- Kill: M1 is within noise of M3. The trivial formula already captures it.

**H4: closed-form optimal G** (nice to have, not a kill criterion)
- In regime R2, the traffic-optimal G measured by ncu is within one power-of-two step of `sqrt(W * BN / BM)`.

---

## 4. Phases (with early exits)

### Phase 0: Inventory (~15 min)
- Record GPU model, SM count, L2 size, max SMEM per block (opt-in), HBM size, and driver/CUDA/Triton/PyTorch versions. Use `triton.runtime.driver.active.utils.get_device_properties`, `torch.cuda.get_device_properties`, and `nvidia-smi`.
- Check `ncu` availability and whether performance counters are permitted. `ERR_NVGPUCTRPERM` means counters are blocked. **If ncu is unavailable, stop and tell Jack.** H1 depends on it, and time alone cannot validate a traffic model.
- Check for other processes on the GPU and record them. Try locking clocks (`nvidia-smi -lgc`); if that is not permitted, record clocks during runs instead.
- Compute the ridge point from published peak and the measured bandwidth (Phase 1).

### Phase 1: Calibration microbenchmarks (~1 h)
- **`L2_eff`:** a Triton kernel that reads a buffer of size S twice, launched across all SMs. Measure DRAM bytes on the second pass with ncu as S sweeps from 1 MB to 2x nominal L2. `L2_eff` is the knee. Also run a strided/hashed-order variant. Report both and pick one before Phase 2.
- **`HBM_BW`** (streaming copy) and **`L2_BW`** (repeated reads of an L2-resident buffer).
- **`eff(config)`:** time every feasible config on 3 to 4 calibration shapes that are large, square, and compute-bound (e.g. 8192³ and 16384x8192x8192 in bf16). Then `eff = ideal_compute_time / measured`. Calibration shapes are excluded from all test sets.

### Phase 2: Knee and GROUP_M sweeps, prediction-first (~2-3 h) → H1
Use the vendored Triton tutorial matmul (grouped ordering). Fix one or two good configs (e.g. 128x128x64, 4 warps, 3-4 stages; plus 128x256 if feasible).
- **K-sweep:** M = N = 8192, K from 512 to 32768 in ~20 log-spaced steps, G ∈ {1, 4, 8, 16, 32}.
- **G-sweep:** three shapes deep in R2 and three in R1 (per model), G ∈ {1, 2, 4, 8, 12, 16, 24, 32, 64}.
- For each run, collect with ncu: DRAM read/write bytes, L2 read sectors, L2 hit rate. Confirm exact metric names with `ncu --query-metrics`; expected names are along the lines of `dram__bytes_read.sum`, `lts__t_sectors_op_read.sum`, and the L2 sector hit-rate metric. Use `--cache-control all` so each kernel starts with a cold L2.
- Plot predicted vs measured bytes as a function of K and of G (analog of LALA Fig. 3). Mark predicted knees.
- **Verdict on H1. If it is killed, stop here and write the report.**

### Phase 3: Exhaustive tile sweep and pruning evaluation (~3-5 h, mostly unattended compute) → H2, H3
- **Config space:**
  - BM, BN ∈ {32, 64, 128, 256}
  - BK ∈ {32, 64, 128}
  - G ∈ {1, 4, 8, 16, 32}
  - num_warps ∈ {4, 8}
  - num_stages ∈ {2, 3, 4, 5}
  - Drop configs infeasible by SMEM. Compile once and reuse across shapes wherever Triton's specialization allows. Log compile time per config, since compile-time savings are part of the value proposition.
- **Shapes** (test set; disjoint from calibration):
  - LLM projection shapes: M (tokens) ∈ {128, 512, 2048, 8192}; (N, K) ∈ {(4096, 4096), (14336, 4096), (4096, 14336), (28672, 8192), (8192, 28672)}.
  - A few tall-K (M = N = 2048, K = 65536) and mid-size square (4096³).
  - fp8 for the same LLM shapes if the GPU supports it, since the higher ridge point makes locality matter more.
  - Skip decode-sized M ≤ 64. There Tm ≈ 1 and ordering is irrelevant; mention that in the report.
- **Timing:** `triton.testing.do_bench` (it flushes L2 between reps; confirm this in the installed version). Report the median of ≥ 3 independent repetitions with randomized config order. Measure a noise floor by re-timing the same config 10x per shape; differences below the noise floor count as ties.
- **Correctness:** check every config against `torch.matmul` (fp32 accumulate, TF32 off) with a sane tolerance. Discard and log failures.
- For a ~10% random subset of (shape, config) pairs, collect ncu DRAM bytes too. This extends the H1 check beyond the controlled sweeps.
- **Evaluate M0 to M3:** regret vs k for k ∈ {1, 3, 5, 10, 20, 5%, 10%}, Kendall τ against the true ranking, and fraction of shapes within 5% of oracle. Break results down by near-ridge vs compute-bound subsets.
- **Verdicts on H2 and H3.**

### Phase 4: Report (~30 min)
`REPORT.md` should contain:
1. A one-paragraph plain-language summary.
2. A verdict table for H1 to H4 with the numbers.
3. Plots: knee sweep, G sweep, regret-vs-k per model, predicted-vs-measured bytes scatter.
4. What surprised you.
5. Threats to validity.
6. If anything passed, the single most promising next step.

Keep it short.

---

## 5. Implementation notes

**Directory layout:**
```
lala-triton-killtest/
  PREREGISTRATION.md
  env/            inventory.json
  kernels/        matmul_grouped.py (vendored tutorial kernel, pinned), microbench.py
  model/          locality_model.py (M3), naive_model.py (M1), triton_perf_model.py (M2)
  tests/          test_model.py
  runs/           *.csv (append-only, resumable; one row per (shape, config, rep))
  predictions/    *.csv written before the matching runs
  plots/
  REPORT.md
```

**Model code:**
- Pure Python and microsecond-cheap per config. Include a debug mode that prints the regime and the `F1`/`F2`/`F3` values.
- Unit tests:
  - Analog of LALA's symbolic RI test: total modeled reads ≥ compulsory bytes, and equal to them in R0.
  - Monotonicity in L2 size.
  - Exact match to M1 when `W = 1`.

**Robustness:**
- Every sweep must be resumable (skip rows already in the CSV) and checkpointed.
- Log GPU clocks, temperature, and the timestamp per row.

**Rigor rules:**
- No per-shape fitting of any model parameter.
- Calibration shapes stay out of every test set.
- Predictions are written before measurements.

**Budget:** if the full config space × shapes exceeds ~5 h of benchmarking, subsample configs uniformly at random (log the seed) rather than dropping shapes, and say so in the report.

---

## 6. Pitfalls to watch for

- **CTA dispatch order is not guaranteed** to follow pid order. If H1 fails mainly on the window assumption, test a persistent-kernel variant (tutorial persistent matmul) where the tile order is controlled in software, before declaring a kill.
- **Effective L2 may be well below nominal** because of partitioning or duplication across partitions. That is why `L2_eff` is calibrated. If the knee looks smeared, report the partition behavior rather than tuning `L2_eff` to fit.
- **ncu replays kernels and serializes them.** Use it only for bytes and hit rates, never for time.
- **Newer Triton versions may change the tutorial kernel** (TMA descriptors, warp specialization on Hopper). Pin the vendored kernel source so every config runs the same code.
- **Compute-bound shapes will show no time effect** from locality even when traffic changes. That is expected and is why H2 is evaluated on near-ridge shapes.

---

## 7. Stretch (only if H1 to H3 pass and time remains)

1. **Eviction hints:** apply `evict_last` to the operand the model predicts is reused across waves (A panels in R1) and `evict_first` to the streamed one. Measure bytes and time. This is the Triton analog of LADM's RONCE/RTWICE.
2. **Fused attention (Triton tutorial) and GQA ordering:** K/V for a head is reused by all query blocks of that head (and by all query heads in a GQA group). Model the cross-wave K/V reuse knee and test whether reordering programs so that heads sharing K/V are adjacent reduces DRAM bytes.
3. **Multi-die / chiplet GPUs** (B200, or MI300X with per-XCD L2), if accessible: this is the direct LADM setting.

---

## 8. Related work to check before claiming novelty (Claude Code: just list what you find, don't go deep)

- Triton's own grouped-ordering rationale (matmul tutorial), and the old `matmul_perf_model`.
- CUTLASS tile-scheduler rasterization and swizzle heuristics (raster order, max swizzle size), and Stream-K.
- Any NVIDIA analytical GEMM-heuristics library (e.g. nvMatmulHeuristics, if it exists in the installed ecosystem) and how it picks tiles and rasterization.
- Smith, Goldfarb, Ding, ICS 2022 (data movement complexity for matmul); LALA builds on it.
- LADM (MICRO 2020) and CODA (TACO 2018).
