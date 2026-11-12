---
type: Benchmark Result
title: CUDA kernels — exact FP32 through an FP16 screen, pipelined BF16, segmented scatter (RTX 3060)
description: Second CUDA pass on the RTX 3060 — nsys profile of the training loop, each step with its measured gain, the exact FP32 top-1 through an FP16 tensor-core screen with an FP32 check (bit-equal to the FP32 kernel), the final interleaved sweep vs faiss Triton on synthetic data, exactness and fallback rates.
resource: crates/emeans-cuda/src/lib.rs
tags: [benchmark, emeans, cuda, triton, tensor-cores, exactness, profile]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: sweep
    resource: whole-run sweep on the RTX 3060 box, synthetic s768_200k and s768_1m, k 1024, seeds 1-5, 2026-11-12
    title: rs-cuda, rs-cuda-bf16, faiss Triton BF16 and FP32, interleaved, load average 0.8-6.5
  - id: profile
    resource: nsys traces of whole training runs and ops_bench, tests and fallback statistics on the box, 2026-11-12
    title: Training-window GPU traces (-t cuda) and per-op timings
  - id: old
    resource: /results/core-engine.md
    title: The first CUDA core (FP32 register-tile top-1, BF16 v1), the "old" columns
  - id: cpu
    resource: /results/cpu-kernels.md
    title: rs-cpu-fused box runs, for the cross-backend check
  - id: ptx
    resource: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    title: PTX ISA (mma.sync m16n8k16, cp.async, ldmatrix, cvt.rn.f16.f32); it does not state whether mma keeps FP16 subnormal inputs
---

# Overview

Second pass on the CUDA backend of the [core](/design/core.md) (`crates/emeans-cuda/src/lib.rs`,
`cuda_src.rs`), RTX 3060 (sm_86), driver 580, CUDA 12.4. Profile first, then the biggest items in
order. Each step and its gain is also in the [backend recipe](/recipes/adding-an-emeans-backend.md)
(CUDA steps U1-U7).

| 200k / 1M x 1024, train s (median of 5)[^old] | Triton BF16 | Triton FP32 | old rs-cuda | **rs-cuda (exact FP32)** | old rs-cuda-bf16 | **rs-cuda-bf16** |
|---|---|---|---|---|---|---|
| 200,000 | 0.15 | 0.31 | 0.29 | **0.16** | 0.15 | **0.10** |
| 1,000,000 | 0.60 | 1.38 | 1.34 | **0.62** | 0.62 | **0.39** |

The default `cuda` backend stays exact (labels and scores bit-equal to the FP32 kernel) and is now
1.9x (200k) and 2.2x (1M) faster than before, as fast as faiss's BF16 default. The opt-in BF16 backend
is 1.5x faster than Triton BF16 at both sizes.[^sweep]

# Profile (where the time went)

nsys (`-t cuda`), training window (from the end of the data upload to the last read), seed 1. Kernel
times are GPU time; idle is time with no GPU work.[^profile]

| Op, ms per run | old FP32 200k | old FP32 1M | old BF16 200k | old BF16 1M |
|---|---|---|---|---|
| top-1 (incl. combine) | 223 | 1,111 | 64 | 290 |
| BF16 conversion (rows each batch + centroids) | — | — | 16 | 75 |
| scatter (count, scan, place, sum) | 23 | 110 | 27 | 129 |
| normalize, EMA, split | 6 | 7 | 6 | 7 |
| copies (epoch-end reads, permutations) | 4 | 20 | 4 | 28 |
| GPU idle | 29 | 83 | 28 | 90 |
| window | 284 | 1,331 | 145 | 619 |

- FP32: top-1 is 78-83% of the GPU time. The only way to a large gain is a cheaper exact top-1.
- BF16: top-1 is under half. Per-batch row conversion, a latency-bound scatter (one warp per cluster
  scanning every label, one dependent load chain per thread in the sum) and host gaps are the rest.
- Idle at 1M: 22 ms at the start (the epoch-0 permutation was drawn after the init permutation),
  3-4 ms at each epoch end (host statistics while the GPU waits), 9-17 ms at the end (label capture
  before the final reads).
- Nsight Compute could not run without GPU counter rights (`ERR_NVGPUCTRPERM`), so per-kernel
  reasons come from the code and from A/B runs.

# Steps and measured gains

`ops_bench` per 16,384-row batch (200k x 768, k 1024; a sync after each op) and the nsys training
window.[^profile]

| # | Step | Before | After |
|---|---|---|---|
| U2 | Segmented stable counting sort for `scatter_add`: a warp owns 8 clusters and one 2,048-row segment; count, scan (`segs x k`), place with 4 label loads in flight; place writes the resolved data row; the sum issues 8 row loads before its 8 adds (order kept) | 0.47 ms (16k), 1.74 ms (65k) | 0.23 ms, 0.74 ms; count 83 -> 15 us, place 92 -> 23 us |
| U3 | BF16: convert the data set once per bound buffer (on the second `top1` call with the same buffer; the cache holds a weak handle, a write drops it), read rows by id in the kernel | conversion 16 ms (200k run), 75 ms (1M) | 4 ms, 16 ms (centroids per call + one data pass) |
| U4 | BF16 kernel v2: `cp.async` 3-stage pipeline, `ldmatrix.x4` from XOR-swizzled 64-byte rows, the (centroid tile, K step) loop flattened so the pipeline runs across tiles, centroid split + combine; 128 x 256 tile for k >= 512 | 1.33 ms per batch (977 us kernel) | 0.87 ms (784 us kernel); 1M batch 3.63 -> 2.87 ms |
| U5 | Exact FP32 top-1 through an FP16 screen (below) | 3.69 ms per batch | 1.39 ms; training 200k 0.29 -> 0.16 s, 1M 1.34 -> 0.62 s |
| U6 | Host overlap in `core/train.rs` / `model.rs`: epoch statistics on a helper thread, epoch 0's permutation drawn during the init, final reads before the last statistics join | GPU idle 1M BF16 90 ms, 200k 28 ms | 58 ms, 16 ms (1M FP32: 83 -> 43 ms) |

BF16 v2 variants at one 16k batch: 128 x 128 x 3 stages 1.09 ms, 128 x 128 x 4 1.16 ms, 128 x 256 x 3
0.87 ms (196 registers, 72 KiB shared, 1 block per SM), v1 with the data cache 1.25 ms.

# The exact FP32 path: FP16 screen + FP32 check

For a row x and centroids c_j, let f_j be the FP32 kernel's score (one `fmaf` chain over d in column
order) and b_j a tensor-core estimate. If |b_j - f_j| <= e for every j, then every centroid whose FP32
score can be the best has b_j >= max b - 2e. So:

1. **Screen** (`screen_128_3`, `screen_256_3`, the U4 pipeline): FP16 operands, FP32 accumulate. Each
   row and centroid is divided by the power of two just above its largest |element| (exact), and
   elements below FP16's normal range (2^-14 after scaling) are zeroed, their |mass| kept per row
   (`gather_f16s` aux rows: squared norm, scale, zeroed mass). Per row it keeps the four best keys per
   centroid part and a bound on every key it dropped (each thread keeps its two best and the largest
   key it evicted; 16 threads per row merge in shared memory).
2. **Check** (`screen_verify`): e = eps * |x| max|c| + (zeroed mass of x * max scale of c + max zeroed
   mass of c * scale of x), eps = (2u + u^2 + 1.01 d 2^-21 + gamma_d) * 1.05, u = 2^-11 (FP16
   rounding; the tensor-core accumulation term is four times the truncation bound; gamma_d is the FP32
   chain's own error). L2 doubles it and adds the rounding of |c|^2 - 2 dot. If every dropped key is
   below max b - 2e, the candidates are complete: the check computes each candidate's score with the
   FP32 kernel's exact arithmetic (the same `fmaf` chain, the BK = 16 padding step, the L2 norm in the
   kernel's 16-lane shuffle order) and keeps the best, lower id on ties. A warp owns 32 rows: x tiles
   are staged once through shared memory with all 32 loads in flight, and each lane runs its own chain.
3. **Fallback**: rows whose candidates may be incomplete (or with non-finite keys) go on a device-side
   list to the FP32 kernel in list mode, with the centroids split in up to 8 parts.

The FP32 kernel's L2 expressions were pinned to explicit `fmaf` so both sides round the same way.
Subnormal FP16 inputs never reach the tensor cores (the PTX ISA does not say whether `mma` keeps them).[^ptx]

| Screen step (16k batch, 200k x 1024) | screen | check | fallback | top-1 total |
|---|---|---|---|---|
| BF16 screen (u = 2^-8, delta 0.017), check one thread per row, fallback one part | 1.34 ms | 1.51 ms | 1.41 ms | 4.6 ms (slower than FP32) |
| FP16 screen with scales (delta 0.0029), fallback in 8 parts | 0.96 ms | 1.73 ms (warp-staged, loads not batched) | 0.30 ms | 3.2 ms |
| + 32 loads in flight per staged tile | 0.98 ms | 0.41 ms | 0.32 ms | 1.4 ms (nsys); `ops_bench` 1.39 ms |

**Rows on the FP32 fallback** in training (synthetic bench files, seed 1, `EMEANS_CUDA_SCREEN_STATS=1`):

| Data | k 256 | k 1024 |
|---|---|---|
| s768_16k | 1.1% | 3.3% |
| s768_100k | 0.003% | 1.6% |
| s768_200k | 0.002% | 6.1% |
| s768_1m | — | 0.41% |
| `ops_bench` mixture 200k | — | 0.10% |

The 200k k 1024 rate comes from groups of near-identical centroids after revival splits (twins 1e-4
apart are always both candidates).

Cost: the FP16 copy of the data set adds 2 bytes per element on the GPU (1M x 768: +1.5 GB; peak GPU
4.54 GiB vs Triton's 3.88). Off with `EMEANS_CUDA_SCREEN=0`; it runs when d % 8 == 0 and k >= 128.

# Exactness

| Check | Result |
|---|---|
| `contract::check` and `check_production` (16,384 x 768, k 1024/1000, exact duplicates), screen on and off (`crates/emeans-cuda/tests/contract.rs`) | pass; scatter, EMA, split and norm mismatches 0; ties keep the lower id |
| `cuda_screen_is_bit_equal_to_fp32_on_near_ties` (32-centroid groups within 1e-6, exact duplicate pairs, rows x 1, 1e3, 1e-3, 1e-15, IP and L2, d 768 and 72) | labels and score bits equal to the FP32 kernel; 6-39% of rows on the fallback, the rest decided by the check |
| unit tests in `crates/emeans-cuda/src/lib.rs`: screen vs FP32 (twins 1e-4, duplicates, a crowd that forces the fallback, d 64/72/768, k 200-1024, indexed and dense sub-ranges, centroid split); scatter across many segments; BF16 cache | pass, bit-equal |
| `backends_train_to_the_same_objective` (`crates/emeans/tests/backends.rs`) | pass |
| New rs-cuda vs old rs-cuda, same seed, final objective | identical, 6 of 6 seeds (200k and 1M, seeds 1-3) |
| New rs-cuda vs `rs-cpu-fused` on the box[^cpu] | identical, 6 of 6 seeds |
| New rs-cuda vs Mac `rs-cpu` (200k x 1024) | within 6e-11 relative, 5 of 5 (the evaluator's float sum on two machines) |
| `cuda_bf16_is_close_to_the_reference` (BF16 v2) | pass, 24 of 3000 labels differ, all within 1e-2 |

ARI vs faiss NumPy and objectives: rs-cuda 0.426 / 0.701 (noise floor 0.416 / 0.674), objectives
within the seed spread of faiss.[^sweep]

# What is left

- 200k x 1024 sends 6% of rows to the fallback (0.3 ms per batch): eight entries per part and a third
  key per thread would cut it, at more registers in the screen kernel (228 now).
- The screen kernel costs 0.98 ms vs 0.78 ms for the plain BF16 kernel (merge and top-2 tracking).
- Start-of-run idle: 10 ms (200k) and 22 ms (1M) of host init (the MT19937 init permutation).
- `topk` is still one warp per query (dedup is not a hot path); not tuned in this pass.
- Done since: the engine no longer reaches `cudarc` through candle; `emeans-cuda` depends on
  `cudarc` 0.19.10 directly (features `driver`, `nvrtc`, dynamic loading,
  `fallback-latest`), see [crate layout](/decisions/crate-layout.md).

[^sweep]: Interleaved whole-run sweep, 2026-11-12; faiss NumPy runs of the earlier sweep as the ARI reference.
[^profile]: nsys traces and `ops_bench` runs on the box, 2026-11-12.
[^old]: [Core engine](/results/core-engine.md).
[^cpu]: [CPU kernels](/results/cpu-kernels.md).
[^ptx]: PTX ISA, `mma`, `cp.async`, `ldmatrix`, `cvt` sections.
