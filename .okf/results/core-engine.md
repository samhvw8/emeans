---
type: Benchmark Result
title: Core engine — device-loop training on Metal and CUDA vs CPU and faiss
description: The core layer (op contract + device-resident loop) measured on synthetic d=768 data on the M1 Pro (CPU vs Metal, 9 cells) and the RTX 3060 (CUDA FP32 and BF16 vs faiss Triton, 10 cells incl. 1M), with what each timer covers, faiss parity reruns and cross-backend exactness. First core pass; later passes are in the CPU, Metal and CUDA kernel results.
resource: crates/emeans-core/src/core/train.rs
tags: [benchmark, emeans, core, metal, cuda, triton, parity, exactness]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: runs
    resource: training sweeps on synthetic spherical Gaussian mixtures (bench/py/gen_data.py), Mac and Ubuntu box, 2026-11-12
    title: Interleaved whole-run sweeps, 5 seeds, fresh process per run, one shared float64 evaluator
  - id: parity
    resource: bench/py/compare.py
    title: bench/src/bin/parity.rs vs bench/py/ref_emeans.py on the seven parity cases
  - id: ops
    resource: bench/src/bin/ops_bench.rs
    title: Per-op timings, one 16,384-row batch, n 200k, d 768
  - id: old
    resource: /results/compare-mac-ubuntu.md
    title: The earlier sweep (host loop), same method
---

# Overview

`Emeans::train` runs the [core layer](/design/core.md)'s device loop. This concept measures it with
the same method as the [host-loop baseline](/results/compare-mac-ubuntu.md):[^old] spherical Gaussian
mixtures (synthetic, d = 768), `nepochs` 5, batch law auto, seeds 1-5, a warm-up run per
implementation and case, implementations interleaved round-robin, a fresh process per run, and one
shared float64 evaluator for the objective and the ARI against faiss NumPy (same seed).[^runs]

# What each timer covers

| Impl | `train_s` covers | Outside `train_s` |
|---|---|---|
| Rust | `Emeans::train` only: epoch permutations, every batch, revival syncs, epoch-end reads, closing means | `load_s` (read the file), `setup_s` (build the engine, compile kernels), `warm_s` (one epoch on the first max(4k, 4096) rows, same k and backend), `upload_s` (copy the data set to the device; 0 on CPU) |
| Python (faiss) | the faiss fit, then `torch.cuda.synchronize()` | `load_s` (read the file and `.to(device)`), `warm_s` (one epoch on the first max(4k, 4096) rows) |

So the host-to-device copy is outside `train_s` for every GPU implementation, Rust and Python alike.
In the earlier sweep the Rust GPU runs uploaded inside `train` (the copy was in `train_s`) and warmed
up with one assignment only; the old `rs-metal-fused` and `rs-cuda` numbers below include it. Upload
medians: Mac 200k 0.11-0.44 s (`newBufferWithBytes`; later 0.018 s with `bind_shared`), box 200k
0.27 s, 1M 0.56 s (pageable `cudaMemcpy`).

# Mac M1 Pro

Superseded for Metal by the second pass in [Metal kernels](/results/metal-kernels.md) (same method;
200k x 1024 Metal 0.77 s, all 9 cells faster than `rs-cpu`). The table below is the first core sweep.

Load average 7.7-14.5 (median 10.4) during the sweep. Train s, median of 5 seeds:

| n | k | py-numpy | rs-cpu (Accelerate) | rs-metal-fused | Metal vs CPU | old rs-cpu | old rs-metal-fused |
|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.11 | 0.07 | 0.07 | 1.0x | 0.09 | 0.12 |
| 16,000 | 256 | 0.18 | 0.08 | 0.10 | 0.8x | 0.13 | 0.25 |
| 16,000 | 1024 | 0.35 | 0.13 | 0.15 | 0.9x | 0.19 | 0.33 |
| 100,000 | 64 | 0.62 | 0.19 | 0.13 | 1.5x | 0.31 | 0.36 |
| 100,000 | 256 | 0.86 | 0.26 | 0.22 | 1.2x | 0.35 | 0.58 |
| 100,000 | 1024 | 1.87 | 0.62 | 0.58 | 1.1x | 0.74 | 1.46 |
| 200,000 | 64 | 1.32 | 0.35 | 0.23 | 1.5x | 0.64 | 0.54 |
| 200,000 | 256 | 1.66 | 0.48 | 0.42 | 1.1x | 0.73 | 0.99 |
| 200,000 | 1024 | 5.02 | 1.28 | 1.21 | 1.06x | 2.34 | 2.89 |

The old columns ran at load average 12-97 (the 200k x 1024 row at 60-97), so the old-to-new change of
`rs-cpu` is mostly load: its code path computes the same numbers. Same-sweep comparison: Metal wins 6
of 9 cells and loses the two 16k cells with k >= 256, where a run is 8-16 batches and the per-batch
dispatch latency (normalize, four scatter kernels, two EMA kernels) is not hidden. At 200k x 1024
Metal is 5% faster on `train_s`, and slower once its 0.11-0.44 s upload is added.

Objectives (mean of 5 seeds) and ARI of `rs-metal-fused` equal `rs-cpu` in all 9 cells; per seed the
final objectives are bit-identical (45 of 45 pairs). `rs-cpu-blocked` (another CPU kernel) differs
from `rs-cpu` by up to 4.5e-4 at k >= 256 (near-tie flips).

# Ubuntu box, RTX 3060

The second CUDA pass ([CUDA kernels](/results/cuda-kernels.md)) supersedes the `rs-cuda` and
`rs-cuda-bf16` numbers below: 200k / 1M x 1024 FP32 0.16 / 0.62 s, BF16 0.10 / 0.39 s.

Box (2x Xeon E5-2683 v4, RTX 3060 12 GB) otherwise idle (load average 2.0-17.5, median 5.2);
driver 580, CUDA 12.4. Train s, median of 5 seeds. `rs-cuda-unfused` is the candle baseline of
`emeans-bench`:

| n | k | py-numpy | Triton BF16 (faiss default) | Triton FP32 | rs-cuda (FP32) | rs-cuda-bf16 | rs-cuda-unfused | old rs-cuda (unfused, host loop) |
|---|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.47 | 0.03 | 0.03 | 0.01 | 0.01 | 0.14 | 0.19 |
| 16,000 | 256 | 0.81 | 0.03 | 0.03 | 0.02 | 0.02 | 0.25 | 0.32 |
| 16,000 | 1024 | 1.33 | 0.03 | 0.05 | 0.05 | 0.03 | 0.24 | 0.35 |
| 100,000 | 64 | 2.10 | 0.06 | 0.07 | 0.04 | 0.04 | 0.43 | 0.43 |
| 100,000 | 256 | 7.99 | 0.06 | 0.08 | 0.07 | 0.05 | 0.36 | 0.41 |
| 100,000 | 1024 | 22.76 | 0.10 | 0.17 | 0.17 | 0.09 | 0.78 | 0.78 |
| 200,000 | 64 | 4.72 | 0.10 | 0.11 | 0.07 | 0.07 | 0.62 | 0.64 |
| 200,000 | 256 | 7.11 | 0.10 | 0.13 | 0.11 | 0.08 | 0.55 | 0.69 |
| 200,000 | 1024 | 12.03 | 0.15 | 0.31 | **0.29** | **0.15** | 1.12 | 1.20 |
| 1,000,000 | 1024 | 57.55 | 0.60 | 1.38 | **1.34** | 0.62 | 4.59 | 5.04 |

(`py-numpy` here is the earlier sweep's runs, kept as the ARI reference; the Triton columns were
rerun in this sweep, interleaved with the Rust runs.)

- **FP32 (exact):** `rs-cuda` matches or beats faiss Triton FP32 in every cell: 0.29 vs 0.31 s at
  200k x 1024, 1.34 vs 1.38 s at 1M. 4.1x faster than the unfused host-loop backend at 200k x 1024.
- **BF16 (opt-in):** equal to Triton BF16 at 200k x 1024 (0.15 s) and 3% behind at 1M (0.62 vs 0.60).
- **Exactness across machines:** `rs-cuda`'s final objectives equal the Mac's `rs-cpu` per seed in all
  45 pairs to 6e-11 relative (35 identical; the rest is the evaluator's float sum on two machines). The
  BF16 runs end at other optima within the seed spread (ARI 0.95-0.97 vs `py-numpy` at k = 64, where
  FP32 is 1.000).
- `py-numpy` on the box differs from the Mac's (another BLAS), so ARI vs `py-numpy` is a weaker check
  there than on the Mac; the Mac table is the parity reference.

# faiss parity

Rerun of the seven parity cases of [e-means algorithm](/design/emeans-algorithm.md) on the core
build:[^parity]

| Backend | Worst difference of means | Table |
|---|---|---|
| `cpu-accelerate` | 0.43% (s768_20k, k 256, quality) | identical to the earlier table, value for value |
| `metal-fused` (device loop) | 0.43% | identical to the CPU rows |
| `metal-fused`, tree-summed norms (rejected) | 0.52% | 3 cases moved: the run left the CPU's trajectory |

At k = 64 on s768_100k and s768_200k the Rust runs repeat faiss's draws: ARI 1.000 vs `py-numpy` on the
Mac for `rs-cpu` and `rs-metal-fused`, and 1.000 for `rs-cuda` on the box.

# Per op, one batch (16,384 rows, d 768)

| Op | M1 Pro Metal, k 1024 | RTX 3060, k 1024 | RTX 3060, k 64 |
|---|---|---|---|
| `top1` | 14.5 ms (rows gathered first; 39 ms read by id) | 3.69 ms FP32 (3 centroid parts), 1.33 ms BF16 | 0.36 ms |
| `normalize_rows` | 0.85 ms | 0.22 ms | 0.02 ms |
| `scatter_add` | 1.3 ms | 0.47 ms | 0.48 ms |
| `ema_update` | 0.27 ms | 0.05 ms | 0.01 ms |
| training loop, data bound (`loop`) | 1.07-1.10 s | 0.30 s FP32, 0.17 s BF16 | 0.08 s |

Each op is timed with one sync after it, so small ops include a device round trip.[^ops]

# Goals

| Goal | Result |
|---|---|
| faiss parity unchanged (worst gap 0.43%), Rust repeats faiss draws | met: same table on CPU and Metal; ARI 1.000 at k = 64 on 100k/200k |
| Same seed, same objective on CPU, Metal, CUDA within 1e-4 | met on the bench data: Metal = CPU bit-identical (45/45), CUDA = Mac CPU within 6e-11 (45/45). On harder data a revival split can send two backends (or two CPU kernels) to different optima; `crates/emeans/tests/backends.rs` checks 1e-4 with revival on (spherical) and off (Euclidean) |
| Mac 200k x 1024: Metal faster than rs-cpu (2.34 s) | met on `train_s`: 1.21 s vs 2.34 s (old) and vs 1.28 s (rs-cpu in the same sweep, 5%); not met if the 0.11-0.44 s upload is counted |
| RTX 3060 200k x 1024 FP32 <= 0.31 s | met: 0.29 s |
| BF16 opt-in <= 0.15 s | met at the bound: 0.15 s (Triton BF16 0.15 s) |
| 1M x 1024 | FP32 1.34 s (Triton FP32 1.38), BF16 0.62 s (Triton BF16 0.60) |

[^old]: [Host-loop baseline](/results/compare-mac-ubuntu.md), the "old" columns.
[^runs]: Whole-run sweeps on 2026-11-12; the data files come from `bench/py/gen_data.py`.
[^parity]: `bench/data/results/` (gitignored), `uv run bench/py/compare.py <dir>`.
[^ops]: `cargo run --release -p emeans-bench --bin ops_bench -- --backend metal|cuda|cuda-bf16 --k 1024|64`.
