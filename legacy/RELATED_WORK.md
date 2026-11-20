# Related work (listed only, not deep-read; 2026-10-15)
- **Triton matmul tutorial**: grouped ordering ("L2 Cache Optimizations" section) is a qualitative rationale with fixed GROUP_M=8. The old `triton.ops.matmul_perf_model` (removed by 3.2) assumes a flat "80% of re-loads hit L2", is GROUP_M-blind, and hard-codes A100 constants (108 SMs). Vendored here as M2.
- **CUTLASS**: `GemmIdentityThreadblockSwizzle<N>` / CUTLASS 3 tile-scheduler raster order + max swizzle size are the direct analog of GROUP_M. Stream-K targets wave quantization, not L2. https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
- **nvMatmulHeuristics** (NVIDIA, `pip install nvidia-matmul-heuristics`): analytic GEMM config selection; the docs list **L2 hit-rate prediction** and memory-bandwidth analysis, and it supports Ampere/Ada/Hopper/Blackwell. Integrated in CUTLASS 4.2+ heuristics. **This is the closest prior art; its internals are not public in detail.** It must be compared against before any novelty claim. https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/ , https://docs.nvidia.com/cutlass/media/docs/cpp/heuristics.html
- **Smith, Goldfarb, Ding, ICS 2022**, "Beyond Time Complexity: Data Movement Complexity Analysis for Matrix Multiplication" (arXiv 2203.02536). Gives a precise analytical MRC for (recursive) matmul under LRU.
- **LADM**: Khairy, Nikiforov, Nellans, Rogers, MICRO 2020. https://engineering.purdue.edu/tgrogers/papers/khairy.micro2020.pdf
- **CODA**: Kim et al., "Co-location of Computation and Data for Near-Data Processing" (arXiv 1710.09517). The spec says TACO 2018; venue not confirmed by search.

## nvMatmulHeuristics (NVMMH): hands-on findings, round 2 (2026-10-16)
- **Install**: `pip install nvidia-matmul-heuristics` (0.1.0.27) into an isolated venv (`.venv_nvmmh`). It ships a closed-source `libnvMatmulHeuristics.so`, a C header, and a ctypes Python wrapper.
- **sm_89 support: yes.** It has predefined hardware descriptors for Ada, including `RTX_6000_ADA`, `L40S`, `L40` and `L20`.
- **Triton target: yes** (`NVMMH_TARGET_TRITON`). Its configs give CTA tile, warp tile, stages, split-k, `gridSwizzle` ("matches CUTLASS, nvFuser and Triton definition") and `ctaOrder`.
  - They map onto our kernel except where it recommends split-k (common for small M) or N-grouped rasterization. In those cases we take its next-ranked mappable config.
  - Observed swizzle values are only {1, 4, 8}.
- **Wrapper bug**: `estimateSiliconMetric` segfaults, because the wrapper passes the target enum where the C API expects a backend handle. Calling the documented C function with a backend handle works (`nvmmh_lib.py`).
- **What it models** (estimable metrics):
  - runtime, plus a separate "relative" runtime
  - L2 hit rate
  - global-memory load/store bytes
  - L2 load bytes
  - smem bytes
  - compute/load/store time
  - static latencies
  - **energy (J) and EDP**
  
  It also has an optional "discovery" process (run benchmarks, commit results) to tune its internal model, including an **energy discovery** pass. With only the internal discovery set loaded, `ENERGY_JOULES` returned 0.
- **Does it model rasterization/GROUP_M against L2?** Partly. Its predicted DRAM bytes depend on swizzle, giving 3–6x variation across G. But **its traffic model shows no L2 capacity knee**: for M=N=8192, 128x128 tiles, G=8, predicted DRAM bytes are a constant 6.72x compulsory for every K from 512 to 32768, while ncu measures 1.0x up to K≈3k and 4.5–6.4x beyond.
  - Its byte-optimal G is 8 in all 12 G-sweeps; measured optima are 12–32. Its swizzle range appears capped at 8.
- **Accuracy vs ncu** (our configs, mapping per Amendment 2b):

  | | M3 | NVMMH |
  |---|---|---|
  | Median abs error, Phase-2 sweeps | 5.4% | 121% |
  | Median abs error, Phase-3 random subset | 0.1% | 175% |
  | Spearman ρ, Phase-2 / Phase-3 | 0.99 / 0.996 | 0.95 / 0.83 |
  | Typical bias | none | ~2.2–2.8x over-prediction |

  Its predicted L2 hit rate vs ncu L2 read-hit rate: Spearman 0.30.
- **Does it model power?** It has an energy metric and an energy-discovery API, so energy is modelled. Nothing in the header suggests runtime is coupled to a power cap, i.e. clock reduction from data movement. Not verified beyond the header.
- **Caveat**: these accuracy numbers depend on our config mapping (warp tiles, ctaOrder ↔ GROUP_M). Its internal discovery data may target CUTLASS-like kernels more than the Triton tutorial kernel.
