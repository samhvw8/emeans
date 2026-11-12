---
type: Benchmark Result
title: Metal kernels — profile and speed steps of the Metal core on the M1 Pro
description: GPU-timestamp profile of the Metal device loop before and after the second Metal pass (pipelined 64x128x32 top-1 with centroid parts, simdgroup-per-row norms, block counting sort for scatter, one-dispatch EMA, no-copy data binding), the top-1 tile sweep, the 9-cell sweep against rs-cpu and faiss NumPy on synthetic data, and the exactness checks.
resource: crates/emeans-metal/src/engine.rs
tags: [benchmark, emeans, core, metal, apple-silicon, simdgroup, exactness, profile]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: profile
    resource: bench/src/bin/train_profile.rs
    title: Whole training runs in one process; EMEANS_METAL_PROFILE=op|batch prints GPU start/end times per op or per command buffer (MetalCore profile), 2026-11-12
  - id: top1-bench
    resource: bench/src/bin/top1_bench.rs
    title: GPU time of one top-1 dispatch per tile and centroid parts (MetalCore::gpu_ms), median of 7-9 after one warm-up
  - id: sweep
    resource: whole-run sweep on the M1 Pro, synthetic s768 data, 9 cells x 5 seeds, rs-cpu, rs-metal-fused, py-numpy, 2026-11-12
    title: Interleaved, fresh process per run; plus a rerun of the 200k x 1024 cell at load 9.3-10.1
  - id: old
    resource: /results/core-engine.md
    title: The first Metal core sweep, the "old" columns
  - id: parity
    resource: bench/py/compare.py
    title: bench/src/bin/parity.rs on this build, 7 faiss parity cases
---

# Overview

A second pass over the Metal backend of the [e-means core](/design/core.md) (`crates/emeans-metal`),
on a busy 16 GB M1 Pro (16 GPU cores). Each step was profiled with Metal GPU timestamps, kept only
when it won, and checked against the op contract, the CPU float order and the faiss parity table.
The steps are recorded with their gains in the [backend recipe](/recipes/adding-an-emeans-backend.md);
this concept holds the measurements. All data is synthetic.

# Profile before

GPU time per op, one training run per cell (`EMEANS_METAL_PROFILE=op` commits each dispatch alone;
`batch` commits one command buffer per batch), load average 13-15:[^profile]

| Op (ms per batch) | 200k x 1024 (16,384 rows) | 16k x 256 (1,024 rows) | 16k x 64 (1,024 rows) |
|---|---|---|---|
| `top1` (64 x 32 x 32 tile) | 15.09 | 0.600 | 0.150 |
| `gather` | 0.63 | 0.038 | 0.033 |
| `normalize` (thread per row; simdgroup per row for k < 256) | 0.37 | 0.357 | 0.079 |
| `scatter_add`: count, scan, place, sum | 0.27, 0.006, 0.42, 0.46 | 0.017, 0.006, 0.016, 0.032 | 0.017, 0.005, 0.015, 0.027 |
| `ema_update`: positions, masses | 0.045, 0.005 | 0.016, 0.004 | 0.008, 0.004 |
| GPU per batch (batch mode) | 17.5 | 1.03 | — |
| Upload (`newBufferWithBytes`, 614 MB) | 258-445 ms | — | — |

At 200k x 1024 the GPU was busy 92% of the run and `top1` was 86% of the GPU time. At 16k x 256 the
GPU time per batch was 1 ms for a 0.4 GFLOP product: `top1` filled 16 threadgroups on 16 cores, and the
thread-per-row norms waited on memory.

# Top-1 tile sweep

One dispatch, 16,384 dense rows, d = 768, GPU time, median of 7 (load 5-23, so +-10%):[^top1-bench]

| Kernel, tile (BM, BN, BK, WM, WN) | k 1024 | k 256 |
|---|---|---|
| old `assign_fused` 64, 32, 32, 2, 2 (the default) | 14.55 ms (1,771 GFLOP/s) | 3.41 ms |
| old 64, 64, 16, 2, 2 | 10.61 ms | 2.63 ms |
| old 64, 128, 16, 2, 4 | 10.30 ms | 2.49 ms |
| old 128, 64, 32, 4, 2 | 10.10 ms | 2.57 ms |
| pipelined, 64, 64, 16, 2, 2 | 10.62 ms | 2.59 ms |
| pipelined, 64, 128, 32, 2, 4 | **8.6-9.0 ms (2,850-3,000 GFLOP/s)** | **2.17-2.22 ms** |
| pipelined, 128, 128, 16, 4, 4 | 9.2-9.6 ms | — |
| pipelined, 64, 256, 16, 2, 8 | 9.5-10.4 ms | — |
| pipelined, 64, 128, 32, 2, 2 (32 x 64 per simdgroup) | 18.1-18.5 ms | — |
| pipelined, double-buffered threadgroup memory, 64, 64, 16 | 13.6-13.8 ms | — |
| pipelined, transposed centroid staging + float2 fragment loads, 64, 128, 16 | 9.3-9.7 ms | — |
| pipelined, `simdgroup_load` fragments, 64, 128, 32 | 10.5-11.0 ms (vs 9.7-10.4 in the same run) | 2.37 ms (vs 2.22) |

Labels were equal to the first tile's in every row. 32 x 32 per simdgroup with BK 32 is the sweet spot:
a wider block per simdgroup spills, BK 16 doubles the barriers, two threadgroup buffers halve the
occupancy. For k <= 64 the 128-wide tile wastes half its columns (16k x 64: 1.12 vs 0.68 ms for 64 x 64
x 16), and small batches (1,024 rows, k 256) fill the GPU only with centroid parts: 0.36 to 0.22 ms per
batch in the op profile. The tiles kept are `FUSED_TILES`; `EMEANS_FUSED_TILE` and
`EMEANS_TOP1_PARTS` force one for experiments.

# Profile after

| Op (ms per batch) | 200k x 1024 | 16k x 64 |
|---|---|---|
| `top1` (64 x 128 x 32; 64 x 64 x 16 for k = 64) | 8.8-9.7 | 0.103 |
| `gather` | 0.67 | 0.033 |
| `normalize` (simdgroup per row, in-order shuffle sum) | 0.085 | 0.027 |
| `scatter_add`: block, scan, place, sum | 0.045, 0.046, 0.009, 0.47 | 0.048, 0.007, 0.005, 0.028 |
| `ema_update` (one dispatch) | 0.045 | 0.009 |
| Upload (`bind_shared`: no-copy view of an `Arc` the engine keeps + one residency dispatch) | 0.06-0.20 s in the sweep (load 15-51); 0.018 s after the ownership fix (3 quiet runs) | 2 ms |

# Results

Whole training runs, 5 seeds, interleaved, fresh process per run. Load average median 11.2
(9.0-73.2); the 200k x 1024 cell ran at 40-51, so its `rs-cpu` and `py-numpy` times (IQR 5.1 s and
10.8 s) are load, not code. Train s, median:[^sweep]

| n | k | py-numpy | rs-cpu | rs-metal-fused | Metal vs CPU | old rs-cpu | old rs-metal-fused |
|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.09 | 0.06 | **0.03** | 2.0x | 0.07 | 0.07 |
| 16,000 | 256 | 0.15 | 0.09 | **0.04** | 2.3x | 0.08 | 0.10 |
| 16,000 | 1024 | 0.30 | 0.12 | **0.08** | 1.5x | 0.13 | 0.15 |
| 100,000 | 64 | 0.49 | 0.23 | **0.09** | 2.6x | 0.19 | 0.13 |
| 100,000 | 256 | 0.70 | 0.32 | **0.18** | 1.8x | 0.26 | 0.22 |
| 100,000 | 1024 | 1.61 | 0.60 | **0.38** | 1.6x | 0.62 | 0.58 |
| 200,000 | 64 | 1.02 | 0.57 | **0.16** | 3.6x | 0.35 | 0.23 |
| 200,000 | 256 | 1.53 | 0.77 | **0.31** | 2.5x | 0.48 | 0.42 |
| 200,000 | 1024 | 5.36 | 6.64 (load 40-51) | **0.77** | — | 1.28 | 1.21 |
| 200,000 | 1024, rerun at load 9.3-10.1 | — | 1.13 | **0.74** | 1.5x | 1.28 | 1.21 |

(`rs-cpu` here is the CPU code of that moment, which was being changed in parallel; the old columns
are the [first core sweep](/results/core-engine.md).[^old]) Metal wins all 9 cells, 1.5-3.6x where
the CPU runs were clean (the 200k x 1024 cell from its rerun); Metal's own time is 1.2-2.5x below its
old time in every cell. Outside the timer, the upload of the data set fell from 0.11-0.44 s to
0.01-0.20 s (0.023 s in the quiet rerun), and the peak footprint at 200k x 1024 from 1.81 to 1.19 GiB
(no second copy of the data).

# Exactness

| Check | Result |
|---|---|
| `contract::check` (`crates/emeans-metal/tests/contract.rs`) on both tiles, parts included | pass; norms, normalize, scatter, EMA, split bit-equal to the CPU |
| `crates/emeans-metal/tests/metal.rs`, `emeans-cluster` `dedup.rs`, `hierarchy.rs`, `emeans-core` `model.rs` | pass |
| Same seed, Metal vs `rs-cpu` in the sweep | final objective identical in 45 of 45 runs (and 5 of 5 in the rerun); centroid files byte-identical |
| faiss parity, 7 cases[^parity] | final objectives identical to the earlier Metal (= CPU) rows in every seed; worst difference of means 0.43%, unchanged |

The top-1 kernel changed its summation order (tile and pipelining) but not its labels on these data:
the near-tie rule never fired differently. The norm kernels keep the CPU's sequential f32 sum.

# Left

- `top1` runs at about 55% of the M1 Pro's FP32 peak and is 85% of the GPU time on large runs; the
  variants in the sweep above did not move it. A different design (more reuse of the centroid tile
  across rows) is the remaining lever.
- About 8 ms per small run is GPU idle time at the revival syncs (every 5 batches) and epoch ends;
  spin-waiting did not reduce it. Removing it changes the training loop's sync points.
- `scatter_sum` still reads rows by id (0.47 ms per 16k x 1024 batch); it could read the batch
  gathered for `top1`.

[^profile]: `EMEANS_METAL_PROFILE=op|batch cargo run --release -p emeans-bench --bin train_profile -- --data bench/data/s768_200k.f32 --k 1024 --reps 3`.
[^top1-bench]: `cargo run --release -p emeans-bench --bin top1_bench -- --rows 16384 --k 1024,256 --tiles ... --parts ...`; the old and experimental kernels were measured before they were removed from the source.
[^sweep]: Interleaved whole-run sweep on 2026-11-12, plus the 200k x 1024 rerun with the same binaries.
[^old]: [Core engine](/results/core-engine.md), Mac table.
[^parity]: `bench/data/results/` (gitignored), `uv run bench/py/compare.py <dir>`.
