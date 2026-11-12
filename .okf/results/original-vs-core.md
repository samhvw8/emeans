---
type: Benchmark Result
title: Original faiss e-means vs the Rust core — fair interleaved run
description: faiss contrib/e_means (NumPy, torch CPU, CUDA native and Triton) against the Rust core after the device-loop rebuild (CUDA FP32/BF16, x86 CPU) and the old unfused Rust CUDA path, synthetic data, 3 seeds interleaved on the RTX 3060 box, plus a light Mac check.
tags: [benchmark, faiss, cuda, triton, metal, cpu]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: fair-run
    resource: whole-run sweep on the RTX 3060 box, synthetic s768 data, 9 implementations x 7 cells x 3 seeds, 2026-11-12
    title: Long table (IQR, memory, ARI) of the fair run
---

# Setup

Ubuntu box (RTX 3060, 2x Xeon E5-2683 v4), 2026-11-12. 9 implementations, 3 seeds, one warm-up per
implementation, order rotated per seed, fresh process per run, data upload outside the train timer
for every implementation (Python `.to(device)` and Rust `upload_s` alike).[^fair-run] Data:
synthetic spherical Gaussian mixtures, d = 768 (`bench/py/gen_data.py`). No other job on the box;
the load average (1.0-54) comes from this sweep's own multi-threaded CPU runs. 189 runs, 0 failures.
Objective: one shared float64 evaluator. "Rust CPU (x86 blocked)" is `cpu-blocked`, before
`cpu-fused` existed ([CPU kernels](/results/cpu-kernels.md)); "Rust CUDA old (unfused)" is the candle
baseline now in `emeans-bench`.

# Train s, median of 3 seeds

| n | k | faiss NumPy | faiss torch CPU | faiss CUDA native f32 | faiss Triton f32 | faiss Triton bf16 | Rust CPU (x86 blocked) | Rust CUDA old (unfused) | **Rust CUDA f32** | **Rust CUDA bf16** |
|---|---|---|---|---|---|---|---|---|---|---|
| 16k | 256 | 0.983 | 0.363 | 0.049 | 0.035 | 0.033 | 1.592 | 0.238 | 0.023 | 0.017 |
| 16k | 1024 | 1.446 | 0.502 | 0.054 | 0.050 | 0.031 | 2.040 | 0.305 | 0.050 | 0.034 |
| 100k | 256 | 8.217 | 0.860 | 0.092 | 0.080 | 0.064 | 2.595 | 0.374 | 0.069 | 0.046 |
| 100k | 1024 | 22.688 | 1.984 | 0.190 | 0.168 | 0.100 | 9.234 | 0.746 | 0.172 | 0.093 |
| 200k | 256 | 7.871 | 1.944 | 0.148 | 0.133 | 0.101 | 4.572 | 0.552 | 0.112 | 0.082 |
| 200k | 1024 | 13.133 | 3.853 | 0.336 | 0.307 | 0.154 | 16.831 | 1.118 | 0.287 | 0.143 |
| 1M | 1024 | 64.259 | 18.236 | 1.513 | 1.381 | 0.603 | 76.322 | 4.293 | 1.347 | 0.608 |

# Objective, mean of 3 seeds (lower is better)

| n | k | faiss NumPy | faiss Triton f32 | faiss Triton bf16 | Rust CUDA f32 | Rust CUDA bf16 |
|---|---|---|---|---|---|---|
| 16k | 256 | 0.29494 | 0.29331 | 0.29224 | 0.29407 | 0.29226 |
| 16k | 1024 | 0.26240 | 0.26215 | 0.26242 | 0.26231 | 0.26271 |
| 100k | 256 | 0.37954 | 0.37230 | 0.37363 | 0.37983 | 0.37862 |
| 100k | 1024 | 0.28256 | 0.28244 | 0.28268 | 0.28251 | 0.28226 |
| 200k | 256 | 0.46835 | 0.45819 | 0.45741 | 0.47248 | 0.46757 |
| 200k | 1024 | 0.28522 | 0.28535 | 0.28510 | 0.28489 | 0.28509 |
| 1M | 1024 | 0.30322 | 0.30325 | 0.30039 | 0.30315 | 0.30106 |

Rust draws faiss NumPy's random stream, so it tracks NumPy (within 0.9%); the torch-based faiss impls
draw their own and land up to 3% lower at k = 256, where 3 seeds cannot separate seed noise from a real
difference (the 5-seed [host-loop baseline](/results/compare-mac-ubuntu.md) shows the same pattern
with per-seed sd 0.4-2.5%). At k = 1024 all impls agree within 0.1% except bf16 at 1M (0.9% lower in
both faiss and Rust).

# Light Mac check (M1 Pro, load about 3, seed 1, interleaved)

| n | k | faiss NumPy | faiss torch CPU | Rust CPU (Accelerate) | Rust Metal |
|---|---|---|---|---|---|
| 16k | 256 | 0.146 | 0.192 | 0.075 | 0.095 |
| 200k | 1024 | 3.18 | 1.80 | 1.28 | 1.16 |

faiss on MPS does not run (float64 statistics).

# Reading

- CUDA FP32: the Rust core matches or beats faiss Triton FP32 in every cell (200k x 1024: 0.287 vs
  0.307 s; 1M: 1.347 vs 1.381 s), 3.9x faster than the old unfused Rust path.
- CUDA BF16 (opt-in in Rust, faiss's default): equal to Triton BF16 (0.143 vs 0.154 s at 200k; 0.608
  vs 0.603 s at 1M).
- CPU on x86: faiss torch CPU (oneDNN) was 4.4x faster than the Rust blocked kernel at 200k x 1024;
  `cpu-fused` closed that gap (1.83 s vs 3.73 s, [CPU kernels](/results/cpu-kernels.md)).
- Mac: Rust CPU and Metal are 1.4-2.7x faster than faiss NumPy and torch CPU.
- Later passes moved the Rust CUDA numbers further: [CUDA kernels](/results/cuda-kernels.md).

[^fair-run]: Interleaved whole-run sweep on 2026-11-12; per-run IQR, memory and ARI were recorded with each run.
