---
type: Benchmark Result
title: CPU kernels — the fused AVX2 top-1 on x86, Accelerate tuning on the Mac, small-op threading
description: Profile of the CPU training loop per op on a 2x Xeon box and an M1 Pro, the cpu-fused kernel (packed GEMM with the argmax fused per tile, AVX2+FMA, no dependency), the physical-core pool, the Accelerate changes, final sweeps against faiss NumPy and torch CPU on synthetic data, and exactness (bit-identical to the old CPU and the CUDA runs).
resource: crates/emeans-core/src/assign/fused.rs
tags: [benchmark, emeans, cpu, avx2, accelerate, rayon, exactness]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: ops
    resource: bench/src/bin/ops_bench.rs
    title: ops_bench on the x86 box (one 16,384-row batch, n 200k, d 768), thread scaling 1/8/16/32/64, tile sweep, NUMA interleave, 2026-11-12
  - id: box-runs
    resource: whole-run sweeps on the x86 box, synthetic s768_200k and s768_1m, 3 seeds, 2026-11-12
    title: py-numpy, py-torch-cpu, rs-cpu-fused, rs-cpu-blocked, interleaved, fresh process per run
  - id: mac-runs
    resource: whole-run sweep on the M1 Pro, synthetic data, 9 cells, 3 seeds, 2026-11-12
    title: rs-cpu, py-torch-cpu, py-numpy (load 11-18, swap nearly full)
  - id: mac-ab
    resource: bench/src/bin/cpu_probe.rs
    title: Mac probe (gather, sgemm, argmax, tiled and pipelined variants) and an interleaved old-vs-new A/B at 200k, 2026-11-12
  - id: cuda-libs
    resource: bench/Cargo.toml
    title: The "split regression" hunt on the x86 box, 2026-11-13 - emeans-compare builds with and without candle's CUDA, LD_PRELOAD of the CUDA libraries, 1 vs 32 threads, interleaved, fresh process per run
  - id: parity
    resource: bench/py/compare.py
    title: bench/src/bin/parity.rs, 7 faiss parity cases, seeds 1-5, on Accelerate and cpu-fused
---

# Overview

The CPU backend of the [core](/design/core.md) had no fast x86 kernel: on the box (2x Xeon E5-2683 v4,
32 cores / 64 threads, AVX2, no AVX-512) `rs-cpu-blocked` ran scalar code (16.5 s at 200k x 1024)
while faiss torch CPU (oneDNN/MKL) took 3.7 s. This work added `cpu-fused`, the x86 default
(`CpuKernel::Fused`), and tuned the Mac's Accelerate path and the small per-batch ops. Every result
is bit-identical to the CPU runs before it, and `cpu-fused` reproduces the CUDA FP32 backend exactly.
All data is synthetic (`bench/py/gen_data.py`, or the unit rows `ops_bench` makes).

# Profile: where a training step goes

`ops_bench` times each op for one 16,384-row batch (n 200k, d 768); the ops add up to the `loop`
time (one training run), so they are the profile. `perf` was not available to users on the box
(`perf_event_paranoid` 4).[^ops]

| Op, k 1024 | Box, `cpu-blocked`, 64 thr (before) | Box, `cpu-fused`, 32 thr (after) | M1 Pro, Accelerate (before) | M1 Pro (after) |
|---|---|---|---|---|
| `top1` (rows by id) | 265-267 ms | 22-26 ms | 17-18 ms (gather 1.4, sgemm 11.1, scalar argmax 3.2) | 13.8-14.6 ms |
| `scatter_add` | 5.4 ms | 2.7-3.2 ms | 1.4-1.8 ms | 1.4-2.4 ms (noise) |
| `normalize_rows` | 1.6 ms | 0.4-0.5 ms | 0.4 ms | 0.4 ms |
| `ema_update` | 0.66 ms | 0.5-0.8 ms | 0.12 ms | 0.12 ms |
| training loop, 65 steps | 16.5 s | 1.62-1.79 s | 1.24-1.51 s | 1.03-1.33 s |

The "before" box column's small ops come from the first `cpu-fused` run (old small-op code, 64
threads); they are the same code for every kernel. At k = 64 on the box, `cpu-fused` top-1 is 3.8-4.2
ms and scatter 2.4-2.9 ms: the scatter is then 40% of a step.

# Thread scaling on the box

`cpu-fused`, k 1024, one batch (`RAYON_NUM_THREADS`):[^ops]

| Threads | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| `top1` | 407-418 ms | 60-61 ms | 33-34 ms | 23-26 ms | 24-25 ms |
| GFLOP/s per core | 62 | 53 | 49 | 33-36 | — |
| Busy-core clock | 3.0 GHz | 2.55 GHz | 2.22 GHz | 1.71 GHz | — |
| FLOP per cycle per core | 21 | 21 | 22 | 20 | — |
| loop (65 steps) | — | 4.02 s | 2.39 s | 1.68-1.79 s | 1.88-2.13 s |

The kernel runs at ~21 of the 32 FLOP per cycle a Broadwell core can issue (65%) at every thread
count; the per-core drop is the AVX clock falling from 3.0 to 1.7 GHz as cores join. Hyperthreads
add nothing to the FMA-bound kernel and slow every small op (more threads to wake): 32 threads beat
64 in the loop (1.68-1.72 vs 1.88-1.91 s at k 1024, 0.56 vs 0.74 s at k 64, interleaved). The scalar
`cpu-blocked` kernel is the opposite (top-1 265-267 ms on 64 threads, 472-473 ms on 32), so only
`cpu-fused` runs its top-1 on the physical-core pool.

The tile sweep (mc 48-192, nc 64-256, kc 128-768, `EMEANS_CPU_TILES`) moved top-1 between 21.8 and
30.9 ms with run-to-run noise of about 3 ms: no setting beat the default 96 x 128 x 256 reliably.
`numactl --interleave=all` did not help (24.0-26.2 ms). Runs that overlapped another job on the box
were redone on an idle box; the tables use the redone values.[^ops]

# Final sweeps

**Box** (otherwise idle; the 1-minute load average of 7-60 is the benchmark's own 32-64 threads),
train s, median of 3 seeds:[^box-runs]

| n | k | py-numpy | py-torch-cpu | rs-cpu-blocked (old) | rs-cpu-blocked | **rs-cpu-fused** | fused vs torch |
|---|---|---|---|---|---|---|---|
| 200,000 | 256 | 7.56 | 1.98 | 4.57 | 4.51 | **0.77** | 2.6x |
| 200,000 | 1024 | 12.83 | 3.73 | 16.83 | 16.52 | **1.83** | 2.0x |
| 1,000,000 | 1024 | 63.29 | 18.31 | 76.32 | 76.06 | **6.92** | 2.6x |

"Old" is the [original vs core](/results/original-vs-core.md) run before this work. Goals met: at or
below torch CPU (3.74 s and 16.8 s).

**Mac** (M1 Pro), train s, median of 3 seeds:[^mac-runs]

Not a quiet machine: load average 11.2-18.3 (median 12.7) before the runs and swap nearly full from
other work, so the absolute times are 1.5-5x the calm Mac's and the spread is wide. Use the
same-sweep columns and the A/B below, not the absolute values. "Old" is the
[core engine](/results/core-engine.md) sweep (load 7.7-14.5, 5 seeds).

| n | k | py-numpy | py-torch-cpu | rs-cpu (this sweep) | rs-cpu old | rs-cpu vs best faiss, same sweep |
|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.14 | 0.20 | 0.10 | 0.07 | 1.4x |
| 16,000 | 256 | 0.16 | 0.22 | 0.08 | 0.08 | 2.0x |
| 16,000 | 1024 | 0.32 | 0.31 | 0.15 | 0.13 | 2.1x |
| 100,000 | 64 | 0.51 | 0.73 | 0.23 | 0.19 | 2.2x |
| 100,000 | 256 | 0.75 | 0.85 | 0.37 | 0.26 | 2.0x |
| 100,000 | 1024 | 1.83 | 1.49 | 0.93 | 0.62 | 1.6x |
| 200,000 | 64 | 1.86 | 1.70 | 1.95 (IQR 0.59) | 0.35 | 0.9x |
| 200,000 | 256 | 2.20 | 2.25 | 1.36 | 0.48 | 1.6x |
| 200,000 | 1024 | 3.42 | 2.38 | 2.10 | 1.28 | 1.1x |

The Mac goal (faster than 1.28 s at 200k x 1024) is shown by an interleaved A/B of the old and new
binaries under the same load (16-19), alternating per seed:[^mac-ab]

| 200k, k | old median | new median | new vs old | final objectives |
|---|---|---|---|---|
| 64 (9 seeds) | 0.626 s | 0.630 s | 0.99x | identical in 9 of 9 |
| 256 (9 seeds) | 0.827 s | 0.794 s | 1.04x | identical in 9 of 9 |
| 1024 (5 seeds) | 1.508 s | 1.199 s | 1.26x | identical in 5 of 5 |

On the calm Mac that is 1.28 s / 1.26 = ~1.0 s, in line with `ops_bench` (loop 1.03-1.12 s new vs
1.24-1.51 s old, interleaved). A quiet-Mac rerun of the 9-cell sweep is still to do.

# CUDA libraries in the process slow the CPU path

A `cpu-fused` run looked 8-12% slower after the library split. It was not the split: the new
`emeans-compare` was built with `--features cuda` and the candle baselines, the old one without CUDA.
candle-core 0.11.1's `cuda` feature forces cudarc's `dynamic-linking`, so that binary maps libcudart,
libcuda, libnvrtc, libcurand, libcublas and libcublasLt at start, and multi-threaded CPU training in
the process is slower even though it never touches the GPU.[^cuda-libs] `emeans-cuda` alone uses
cudarc's dynamic loading and maps nothing until a CUDA backend is built.

Box, 200k x 1024, `cpu-fused`, `train_s`, medians of interleaved fresh-process runs:

| Binary | Median | Range | Runs |
|---|---|---|---|
| Old code, no CUDA (`aabfef0`) | 1.78 s | 1.72-1.88 | 6 |
| Old code, `cuda` (candle CUDA linked) | 1.87 s | 1.82-1.97 | 4 |
| New code, `cuda` + baselines (candle CUDA linked) | 2.09 s | 1.79-2.18 | 6 |
| New code, no CUDA | 1.81 s | 1.74-1.84 | 6 |
| New code, no CUDA, the six libraries `LD_PRELOAD`ed | 2.03 s | 1.81-2.26 | 5 |
| New code, `cuda` without candle CUDA (the fix) | 1.81 s | 1.70-2.04 | 6 |

- One thread (16k rows): no difference (2.07 vs 2.10 s). At 32 threads on the global pool
  (`RAYON_NUM_THREADS=32`) the gap stays (1.80 vs 1.94 s), so it is not the physical-core pool.
- With the libraries loaded the run uses more user CPU (55-60 s vs 49-54 s); system time and page
  faults barely change. NUMA interleave does not remove the gap. The exact cause in the hardware is
  not known: `perf` is closed to users on the box.
- Centroids are byte-identical in every run, CUDA included; the CUDA backend is equally fast in both
  builds (0.162 s median).
- Mac (no CUDA, so not affected): `cpu-accelerate` at 100k x 1024, `train_profile`, old (`aabfef0`)
  vs new, 12 interleaved runs each: medians 596 ms vs 552 ms (busy Mac, ranges 514-905 and
  503-785 ms); final objectives identical per seed in 12 of 12 pairs.
- Fix: `emeans-bench`'s `cuda` no longer turns on candle's CUDA; the `cuda-unfused` baseline needs
  `cuda-baselines`. The photo tool's `box_build.sh` builds `emeans-compare` with `cuda` only.

# Exactness

| Check | Result |
|---|---|
| `cpu-fused` AVX2 vs portable `mul_add` kernel (`EMEANS_CPU_PORTABLE`), several tile sizes, IP and L2 | same bits (unit tests in `assign/fused.rs`, run on the box) |
| Contract (`crates/emeans-core/tests/contract.rs`), incl. `check_production` (16,384 x 768, k 1024 and 1000, duplicate centroids and rows) on `cpu-fused` and Accelerate | pass on the Mac and the box |
| `streaming_api_reproduces_train_bit_for_bit` (one-pass scatter vs the streaming API's `add_rows`) | pass |
| faiss parity, 7 cases, seeds 1-5, on Accelerate and on `cpu-fused`[^parity] | 33 of 33 runs bit-identical to the old Accelerate table (final and every epoch objective); worst difference of means 0.43%, unchanged |
| Box: `rs-cpu-fused` vs `rs-cuda` (FP32), same seeds | every epoch objective identical in 9 of 9 runs (200k k 256/1024, 1M k 1024)[^box-runs] |
| Box `rs-cpu-fused` vs the Mac's `rs-cpu` (Accelerate) | identical in 6 of 6 runs (200k, k 256 and 1024) |
| `rs-cpu-blocked` new vs old | same objectives per seed (its own float order; 4e-5 to 4e-3 from the others, near-tie flips) |
| Mac Accelerate old vs new (200k, k 64/256/1024) | final objectives identical in all 33 interleaved pairs[^mac-ab] |
| Mac `rs-cpu` sweep vs the old core sweep, 9 cells x seeds 1-3 | every epoch objective identical in 27 of 27 runs[^mac-runs] |

Why `cpu-fused` matches the GPU bit for bit: each dot product is one sequential FMA chain over the
dimensions, which is what the CUDA FP32 kernel computes; Accelerate happens to give the same top-1
on these sets.

# New dependency and running cost

None. `cpu-fused` is ~200 lines of Rust (`std::arch` AVX2+FMA intrinsics, runtime detection, a
portable twin with the same bits); no BLAS, no system package. The physical-core pool is one rayon
pool built once per process. Options not taken: OpenBLAS/MKL (a system or vendored library to keep
in step across hosts) and the `gemm` crate as the default x86 kernel (`cpu-gemm`, the call candle's
CPU matmul makes, ran 275-326 ms per 16k x 1024 batch on 32-64 threads against 22-26 ms for
`cpu-fused`, and it writes the `rows x k` matrix). `gemm` stays as the off-macOS engine of `gemm_nt`
(kNN, metrics) and as `CpuKernel::Gemm`.

[^ops]: `cargo run --release -p emeans-bench --bin ops_bench -- --backend cpu-fused|cpu-blocked --k 1024|64 --reps 5-9` with `RAYON_NUM_THREADS`, `EMEANS_CPU_TILES`, `numactl`; clocks read from `/proc/cpuinfo` during the run.
[^box-runs]: Interleaved whole-run sweep, 3 seeds, 2026-11-12; Rust rows rerun after the pool fix.
[^mac-runs]: Interleaved whole-run sweep on the Mac, 3 seeds, load average recorded per run.
[^mac-ab]: Interleaved runs of the old and new binaries (same CLI), 200k, seeds 1-5 (k 1024) and 1-9 (k 64, 256), Mac load 16-19.
[^cuda-libs]: Box logs of the hunt: builds of `emeans-compare` with `--features cuda` (candle CUDA via `baselines`), with `--no-default-features --features cuda` and without CUDA; `ldd` shows which binaries map the CUDA libraries.
[^parity]: `bench/data/results/` (gitignored), `uv run bench/py/compare.py <dir>`.
