---
type: Benchmark Result
title: Host-loop baseline — faiss e-means vs the first Rust port on an M1 Pro and an RTX 3060 box
description: The first fair comparison on synthetic d=768 data, before the core layer existed — faiss contrib/e_means (NumPy, torch CPU/MPS/CUDA with native and Triton kernels) vs the Rust port with the training loop on the host — measured on a busy M1 Pro and an idle Ubuntu box with an RTX 3060. Superseded for the Rust rows; kept as the baseline the later passes are measured against.
tags: [benchmark, emeans, faiss, metal, cuda, triton, baseline]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: runs
    resource: whole-run sweeps on the M1 Pro and the RTX 3060 box, synthetic s768 data, 5 seeds, 2026-11-12
    title: Per-run train time, objective, ARI and peak memory
  - id: faiss-emeans
    resource: https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means
    title: faiss contrib/e_means at the pinned commit
---

# Overview

A fair comparison of faiss's Python `contrib/e_means`[^faiss-emeans] and the Rust port
([e-means algorithm](/design/emeans-algorithm.md)) at a time when the Rust GPU backends still ran the
training loop on the host (one sync per batch) and the CUDA backend was candle matmul + argmax.

Setup: spherical Gaussian mixtures (synthetic), d = 768 (n = 16k, 100k, 200k; 1M on the box),
k = 64, 256, 1024, `nepochs` 5, batch law auto, seeds 1-5 for every implementation. One shared
evaluator computes the final objective (mean 2 - 2 max cos, float64) and ARI vs `py-numpy` from each
run's centroids. Train time excludes loading and one-time device setup (untimed in-process
warm-up). One warm-up run per implementation and case, then implementations interleaved
round-robin per seed, fresh process per run.[^runs]

**Superseded for the Rust rows:** the Rust backends now train with the core layer's device loop.
See [core engine results](/results/core-engine.md), [Metal kernels](/results/metal-kernels.md),
[CUDA kernels](/results/cuda-kernels.md) and [CPU kernels](/results/cpu-kernels.md).

# Train time (s, median of 5 seeds)

Mac M1 Pro, 16 GB, busy (load average 12-97, median 21; the 200k x 1024 row ran at load 60-97):

| n | k | py-numpy | py-torch-cpu | rs-cpu | rs-metal-fused |
|---|---|---|---|---|---|
| 16,000 | 64 | 0.11 | 0.22 | 0.09 | 0.12 |
| 16,000 | 256 | 0.18 | 0.24 | 0.13 | 0.25 |
| 16,000 | 1024 | 0.35 | 0.31 | 0.19 | 0.33 |
| 100,000 | 64 | 0.62 | 0.65 | 0.31 | 0.36 |
| 100,000 | 256 | 0.86 | 0.74 | 0.35 | 0.58 |
| 100,000 | 1024 | 1.87 | 1.15 | 0.74 | 1.46 |
| 200,000 | 64 | 1.32 | 1.06 | 0.64 | 0.54 |
| 200,000 | 256 | 1.66 | 1.23 | 0.73 | 0.99 |
| 200,000 | 1024 | 5.02 | 3.48 | 2.34 | 2.89 |

`py-torch-mps` fails on every case: faiss keeps statistics in float64, which MPS does not support.

Ubuntu box, 2x Xeon E5-2683 v4 (64 threads), RTX 3060 12 GB, otherwise idle (`-f32` = native FP32
operands; without it faiss converts FP32 CUDA input to BF16). `rs-cpu-candle-gemm` and `rs-cuda`
(unfused, host loop) are now the candle baselines of `emeans-bench`:

| n | k | py-numpy | py-torch-cpu | torch-cuda-native | torch-cuda-native-f32 | torch-cuda-triton | torch-cuda-triton-f32 | rs-cpu-blocked | rs-cpu-candle-gemm | rs-cuda |
|---|---|---|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.47 | 0.24 | 0.03 | 0.03 | 0.03 | 0.03 | 0.50 | 0.75 | 0.19 |
| 16,000 | 256 | 0.81 | 0.34 | 0.05 | 0.05 | 0.03 | 0.04 | 1.58 | 1.03 | 0.32 |
| 16,000 | 1024 | 1.33 | 0.44 | 0.05 | 0.05 | 0.03 | 0.05 | 1.90 | 1.39 | 0.35 |
| 100,000 | 64 | 2.10 | 0.58 | 0.07 | 0.07 | 0.06 | 0.07 | 1.00 | 1.87 | 0.43 |
| 100,000 | 256 | 7.99 | 0.82 | 0.08 | 0.09 | 0.06 | 0.08 | 2.47 | 2.98 | 0.41 |
| 100,000 | 1024 | 22.76 | 1.82 | 0.13 | 0.19 | 0.10 | 0.17 | 9.06 | 11.75 | 0.78 |
| 200,000 | 64 | 4.72 | 1.06 | 0.11 | 0.11 | 0.10 | 0.11 | 1.63 | 3.39 | 0.64 |
| 200,000 | 256 | 7.11 | 1.75 | 0.12 | 0.14 | 0.10 | 0.13 | 4.45 | 5.90 | 0.69 |
| 200,000 | 1024 | 12.03 | 3.74 | 0.20 | 0.34 | 0.15 | 0.31 | 16.53 | 22.84 | 1.20 |
| 1,000,000 | 1024 | 57.55 | 16.84 | 0.79 | 1.51 | 0.60 | 1.38 | 76.21 | 111.54 | 5.04 |

# Findings

- **Quality is equal.** All implementations reach the same objective within seed noise (means within
  0.4% at k = 64 and 1024, up to 3% at k = 256 where the seed sd is 0.4-2.5%). torch behaves like
  another seed of the same algorithm: its ARI vs `py-numpy` equals the seed-to-seed ARI of
  `py-numpy`. The Rust port repeats faiss's random draws (ARI 0.99-1.00 at k = 64). At 1M, the BF16
  CUDA runs ended 0.8% lower than FP32 (one case, ~2.5 sd).
- **Mac:** `rs-cpu` (Accelerate/AMX) is fastest in 8 of 9 cases, 1.2-2.5x faiss NumPy. The fused Metal
  kernel lost to it in training (per-batch overhead of the host loop), although it was already the
  fastest assignment kernel. This is why the device loop exists.
- **Ubuntu GPU:** faiss Triton BF16 is fastest (0.15 s at 200k x 1024; FP32 0.31 s). The unfused
  `rs-cuda` backend of this sweep took 1.20 s (3.6-3.9x Triton FP32). The follow-up (fused CUDA
  kernel, device loop) reached 0.29 s FP32 and 0.15 s BF16, later 0.16 s and 0.10 s.
- **Ubuntu CPU:** torch CPU is fastest (3.7 s at 200k x 1024). faiss NumPy is slower with 64 OpenMP
  threads than with 16 (7.96 s vs 4.01 s at 100k x 256). Rust on x86 lacked a BLAS-class kernel: the
  blocked kernel is scalar code off aarch64 (16.5 s), candle GEMM 22.8 s. The x86 default is now
  `cpu-fused`: 1.83 s, 6.92 s at 1M ([CPU kernels](/results/cpu-kernels.md)).

[^runs]: Whole-run sweeps on 2026-11-12; per-cell objectives, ARI and peak memory were recorded with each run.
[^faiss-emeans]: The original package at the pinned commit; its NumPy files are byte-identical to `reference/faiss_e_means`.
