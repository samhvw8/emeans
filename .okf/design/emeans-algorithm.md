---
type: Design
title: e-means algorithm — the Rust port of faiss contrib/e_means
description: How emeans-core ports faiss NumpyEmeans (recipe, EMA update, revival, closing means), its API, its CPU and Metal assignment kernels, where it deviates from faiss, and the measured parity and timing on synthetic data.
resource: crates/emeans-core
tags: [emeans, algorithm, metal, parity, benchmark]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: faiss-emeans
    resource: reference/faiss_e_means
    title: Vendored faiss contrib/e_means (e_means.py, base.py, e_means_spherical.py), MIT, pinned commit in UPSTREAM_COMMIT (2b648f06)
  - id: paper
    resource: https://arxiv.org/abs/2611.09501
    title: "E-Means: Scaling Laws for Mini-Batch K-Means"
  - id: mlx-gemm
    resource: https://docs.rs/crate/candle-metal-kernels/0.11.1/source/src/metal_src/mlx_gemm.metal
    title: candle-metal-kernels 0.11.1 MLX steel GEMM (BlockMMA fragment layout, tile choice for M1 Pro)
  - id: core
    resource: /results/core-engine.md
    title: Core-layer parity reruns and training sweeps (2026-11-12)
  - id: parity-runs
    resource: bench/py/compare.py
    title: Parity and timing outputs of bench/py/ref_emeans.py, bench/src/bin/parity.rs and bench/src/bin/bench_assign.rs on bench/data (synthetic, gitignored), 2026-11-12
---

# Overview

`emeans-core` is Algorithm 1 of the e-means paper[^paper] as shipped in faiss `NumpyEmeans`, in
Rust.[^faiss-emeans] It follows the Python code, not the README summary. The port draws the same
random numbers as faiss: `rng::RandomState` is numpy's legacy MT19937 (seeding, `permutation`,
`random_sample`). So with the same seed it picks the same initial centroids, visits rows in the same
order and samples the same split donors. Only float summation order differs, so runs match faiss
draw for draw until a near-tie flips one label.

| Module (`crates/emeans-core/src`) | Content |
|---|---|
| `recipe` | Batch law (Eq. 7), step law (Eq. 6), `batch_law` auto/throughput/quality, power-of-two rounding and caps, `epoch_rows` (fractional epochs), `epoch_seed`, cosine schedule to alpha0/10 (Eq. 5), `Recipe` |
| `model` | `Emeans`: `train` (runs the core's device loop on the assigner's engine), the streaming API `begin` / `accumulate` / `update` / `end_epoch` / `finalize` (host loop on the same CPU blocks), `assign`, `resume` |
| `assign` | `Assigner` trait, `Rows` (dense or row ids), CPU kernels: Accelerate `cblas_sgemm` (macOS default), `cpu-fused` (`assign/fused.rs`: packed AVX2+FMA GEMM with the argmax fused per tile; x86-64 default), `cpu-gemm` (the `gemm` crate, off macOS), hand-blocked NEON (`cpu-blocked`); `gemm_nt`; the physical-core rayon pool (`cpu_pool`); `SharedAssigner` |
| `core` | The [core layer](/design/core.md): the `Device` op contract, the `Engine` trait, the f64 reference and contract check, and the device-resident training loop that `train` runs |
| `knn` | CPU exact top-k (`gemm_nt` + rayon), `Neighbors`, `GPU_KMAX` |
| `io` | `save` / `load` model files, raw little-endian f32 matrices |
| `rng` | numpy legacy `RandomState` |

The Metal engine (`MetalCore`, `MetalKnn`) is in `crates/emeans-metal`, the CUDA engine (`CudaCore`)
in `crates/emeans-cuda`; both implement `Device` and `Assigner`.

`gemm_nt` (`out = x y^T` on the CPU) calls Accelerate on macOS and the `gemm` crate elsewhere, with the
same version, features and call that candle-core 0.11.1's CPU matmul uses, so a binary that also
links candle compiles one copy of `gemm` and gets the same numerics.

# API

```rust
use emeans::{Backend, Config, Emeans, Trained, Assignment, make_assigner};
let mut cfg = Config::new(d, k);          // nepochs 5, batch_law Auto, seed 1234
cfg.spherical = true;                     // unit vectors, inner product, centroids on the sphere
let mut m = Emeans::with_assigner(cfg, make_assigner(Backend::MetalFused)?)?;
let t: Trained = m.train(&x)?;            // centroids (k x d), last-pass labels (-1 = not sampled)
let a: Assignment = m.assign(&new)?;      // labels + score (cosine, or squared L2)
m.save("model.emeans")?;
let mut m = Emeans::load("model.emeans")?;
m.train_resume(&grown)?;                  // continue from the saved EMA positions and masses
```

Inputs are plain row-major `f32` slices (`n x d`); outputs are plain structs and vectors.

| Call | Use |
|---|---|
| `begin(n, init)` / `begin_with_state(n, positions, usage)` / `resume(n)` | Start a run from fresh centroids or from saved EMA state |
| `accumulate(&[f32])` / `accumulate_rows(Rows)` | Add rows to the current logical batch; returns labels and scores of those rows |
| `update()` / `end_epoch()` / `finalize()` | Apply the batch; close an epoch; closing means |
| `local_batch_size()`, `in_last_epoch()`, `recipe()`, `iteration_stats()`, `ema_state()`, `current_centroids()` | Inspect the run |
| `assign_rows(Rows)` | Top-1 centroid and similarity for new vectors (incremental assignment, dedup) |
| `assign::assign_all(&mut dyn Assigner, ...)` | Nearest-centroid search without a model |

# Algorithm details kept from faiss

- f32 centroids, masses, batch sums and counts; f64 epoch counts and objectives. The EMA step is
  computed in f64 and cast to f32 where numpy casts a Python float (NEP 50).
- `update`: `N = (1 - alpha) N + alpha n_j`, `c += (alpha n_j / N)(s_j / n_j - c)`; masses of
  untouched clusters decay too; when alpha is exactly 1, masses are replaced, not decayed.
- Revival every 5 updates: clusters below `tau * mean(N)` (tau 0.1, or 0 when d <= 2) take a donor
  drawn by Gumbel top-k on `log N`; donor and child get `c (1 -/+ 1e-4 s)` with an alternating sign
  `s`, and half the mass each. The threshold uses numpy's float32 pairwise sum.
- Batch sums add rows to each centroid in row order, the same additions as the scipy product in faiss.
- Closing: last-epoch sums / counts; clusters with no last-epoch rows keep their EMA position;
  spherical closing means are normalized (`normalize_into`: prescale by max, then by norm).
- Initial mass 2^-16, remainder batches kept, fractional epochs (the partial pass comes before the
  closing pass).

# Deviations from faiss

| Deviation | Why |
|---|---|
| `max_points_per_centroid` draws its subset from MT19937 (`seed ^ 0x9E3779B9`), not `default_rng(seed).choice` | numpy's PCG64 `choice` is not ported; the subset differs from faiss, the rule (k * cap random rows, others labelled -1) is the same |
| EMA positions and masses survive `finalize`; `begin_with_state`, `resume`, `train_resume`, `save`/`load` are new | Incremental updates without retraining |
| `update()` takes no statistics; it applies the batch that `accumulate` built | No distributed or external statistics |
| Misuse fails: too many rows for a batch, `update` on an incomplete batch, `end_epoch` before all epoch rows, `finalize` before the last epoch, a truncated model file. A failed object refuses more work | Crashing beats corrupting; faiss leaves these unchecked |
| `accumulate` returns labels for every epoch, not only the last | The caller may want them; it costs nothing |
| No f64 widening of batch sums when a batch exceeds 2^24 rows | Batches here are at most a few thousand rows |
| Equal Gumbel keys break ties by higher index | numpy's unstable `argsort` order cannot be reproduced |
| Seeds are u32 | numpy `RandomState` accepts only 0..2^32-1 |
| Not ported: torch `Emeans`, quantized FP8/INT8, distributed, sharded, IVF, Triton/CuTe | Out of scope |

# Assignment kernels

The fused Metal kernel (`crates/emeans-metal/src/engine.rs`, `TOP1_SOURCE`, re-exported as
`FUSED_SOURCE`) is one compute shader compiled at runtime from source with `newLibraryWithSource`
(no offline `metal` compiler needed). In its first version a threadgroup of 128 threads (2 x 2
simdgroups) took 64 rows and walked the centroids in tiles of 32: it staged 64x32 row and 32x32
centroid slices in threadgroup memory (float4 loads, padded stride), multiplied `simdgroup_float8x8`
fragments loaded MLX-style by lane coordinates, and kept the running best per row in registers.
Lanes and simdgroups combine once at the end. The `n x k` matrix never exists. That tile was MLX's
choice for this shape on an M1 Pro;[^mlx-gemm] the current software-pipelined 64 x 128 x 32 tile and
the per-call tile and centroid-part choice (`FUSED_TILES`) are in [Metal kernels](/results/metal-kernels.md).
The kernel takes separate label and score outputs, a position offset and a centroid-norm buffer.

Without `#pragma clang loop unroll(full)` the kernel ran at 50-290 GFLOP/s: a plain `#pragma unroll`
left fragment-array indices dynamic and the accumulators spilled. Full unroll gave 1300-1600 GFLOP/s.

# Parity with faiss NumpyEmeans

Final objective = mean squared distance (Euclidean) or mean of 2 - 2 cos (spherical) of every row to
its nearest final centroid. Same synthetic data (`bench/py/gen_data.py`), seeds and settings; Rust on
`cpu-accelerate`.[^parity-runs]

| Case (data, k, nepochs, law) | Seeds | faiss (mean ± sd) | Rust (mean ± sd) | Rel. diff of means | Max per-seed rel. diff | Recipe equal | Splits faiss vs Rust |
|---|---|---|---|---|---|---|---|
| s768_20k, 64, 5, auto | 5 | 0.689032 ± 0.006515 | 0.689032 ± 0.006515 | 5.6e-08 | 1.8e-07 | yes | 1/2/3/1/2 vs 1/2/3/1/2 |
| s768_20k, 64, 2.5, auto | 5 | 0.692136 ± 0.004576 | 0.690882 ± 0.004487 | 1.8e-03 | 9.0e-03 | yes | 1/1/2/1/2 vs 1/1/3/1/2 |
| s768_20k, 256, 3, quality | 5 | 0.299956 ± 0.001613 | 0.298671 ± 0.003632 | 4.3e-03 | 1.7e-02 | yes | 528/494/516/484/493 vs 541/485/532/489/501 |
| s768_100k, 1024, 3, auto | 3 | 0.283127 ± 0.000294 | 0.283149 ± 0.000448 | 7.7e-05 | 1.6e-03 | yes | 2109/2165/2159 vs 2086/2137/2194 |
| e16_50k (Euclidean), 64, 5, auto | 5 | 2.302078 ± 0.127133 | 2.302077 ± 0.127135 | 5.8e-07 | 3.0e-06 | yes | 14/16/16/10/16 vs 14/16/16/10/16 |
| e16_50k (Euclidean), 256, 1, auto | 5 | 1.683241 ± 0.004756 | 1.683043 ± 0.004939 | 1.2e-04 | 5.9e-04 | yes | 148/148/137/135/139 vs 148/144/137/135/139 |
| s16_20k, 32, 3, throughput | 5 | 0.223688 ± 0.005362 | 0.223688 ± 0.005362 | 1.3e-08 | 9.6e-08 | yes | 2/2/1/1/3 vs 2/2/1/1/3 |

Worst difference of means: 0.43%. The core-layer rerun gives this same table, value for value, on
`cpu-accelerate`, on `cpu-fused` and on `metal-fused` (the device loop).[^core] Where seeds differ,
the per-epoch objectives agree to about 1e-8 until one epoch, then drift within the seed-to-seed
spread: a near-tie in the dot products (BLAS vs Accelerate summation order) changed one label and the
process is chaotic after that.

Unit tests (`crates/emeans-core/tests/reference.rs`) check, bit for bit against values computed by
the Python reference: batch size, alpha0, epoch rows and steps, steps per pass, total steps and
revival threshold for 1443 (n, k, nepochs, batch_law, d) combinations; the cosine schedule; MT19937
permutations and samples. They also train on two small fixture data sets (5 runs) and require the
faiss last-pass labels, split counts, assign labels exactly, and objectives within 1e-5.

# Timing

One assignment pass over all rows (`bench_assign`), synthetic data bound on the device, d = 768,
spherical, median of 7 (17k) or 5 (200k) runs, first Metal tile, on a busy M1 Pro (load average 19-33
from other jobs, so treat single cells as ±2x). Mismatches are counted against `cpu-accelerate`; all
were ties (exact f64 scores within 1e-5). `cpu-candle-gemm` and `metal-unfused` are the candle
baselines of `emeans-bench` (feature `baselines`).

| n | k | cpu-accelerate | cpu-candle-gemm | cpu-blocked | metal-unfused | metal-fused |
|---|---|---|---|---|---|---|
| 17,000 | 64 | 21.2 ms (noisy; 3.6-4.3 ms in calm runs) | 13.4 ms | 5.6 ms | 10.9 ms | **1.6 ms** |
| 17,000 | 256 | 8.7 ms | 23.8 ms | 16.7 ms | 13.0 ms | **4.5 ms** |
| 17,000 | 1024 | 19.0 ms | 74.5 ms | 65.6 ms | 19.5 ms | **17.5 ms** (1530 GFLOP/s) |
| 200,000 | 64 | 94.3 ms | 147.0 ms | 64.2 ms | 135.4 ms | **15.0 ms** |
| 200,000 | 256 | 151.3 ms | 859.0 ms | 325.3 ms (2 ties) | 154.2 ms | **48.7 ms** |
| 200,000 | 1024 | 386.8 ms | 859.4 ms | 1081.2 ms (1 tie) | 278.6 ms | **193.1 ms** (1629 GFLOP/s) |

The fused kernel is the fastest backend in every cell: 1.1-9x faster than candle matmul + argmax, which
writes and re-reads the `n x k` matrix. On the CPU, Accelerate (AMX) wins at k >= 256 and is the default
CPU kernel on macOS; the hand-blocked NEON kernel wins at small k. On x86-64 the default is
`cpu-fused` (200k x 1024 training 1.83 s on a 2x Xeon E5-2683 v4 box vs 16.5 s for `cpu-blocked`
and 3.73 s for faiss torch CPU; [CPU kernels](/results/cpu-kernels.md)).

Whole training runs were dominated by per-batch overhead while the loop ran on the host (centroids
up, labels down, EMA on the CPU every batch): Metal lost to the CPU in full training (2.89 vs 2.34 s
at 200k x 1024, [host-loop baseline](/results/compare-mac-ubuntu.md)). `train` now runs the
[core layer](/design/core.md)'s device loop: on Metal it gathers each batch's rows into one
contiguous block, runs the fused kernel on them, and keeps the EMA on the GPU. The current training
numbers are in [core engine results](/results/core-engine.md) and [Metal kernels](/results/metal-kernels.md).

# Reproduce

```bash
# Synthetic data sets (bench/data, gitignored)
uv run --python 3.12 bench/py/gen_data.py --name s768_20k --n 20000 --d 768
uv run --python 3.12 bench/py/gen_data.py --name s768_17k --n 17000 --d 768 --seed 11
uv run --python 3.12 bench/py/gen_data.py --name s768_100k --n 100000 --d 768 --centers 400 --seed 3
uv run --python 3.12 bench/py/gen_data.py --name s768_200k --n 200000 --d 768 --centers 500 --seed 5
uv run --python 3.12 bench/py/gen_data.py --name e16_50k --n 50000 --d 16 --centers 60 --sigma 0.35 --euclidean --seed 9
uv run --python 3.12 bench/py/gen_data.py --name s16_20k --n 20000 --d 16 --centers 40 --sigma 0.5 --seed 13
uv run --python 3.12 bench/py/ref_emeans.py --data s768_20k --k 64 --seeds 1,2,3,4,5 > bench/data/results/py_X.json
cargo run --release -p emeans-bench --bin parity -- --data s768_20k --k 64 --seeds 1,2,3,4,5 > bench/data/results/rs_X.json
uv run bench/py/compare.py bench/data/results
cargo run --release -p emeans-bench --features baselines --bin bench_assign -- --data s768_200k --ks 64,256,1024
uv run --python 3.12 bench/py/fixtures.py    # regenerates crates/emeans-core/tests/fixtures
```

Related: [automatic k, incremental, dedup](/design/autok-incremental-dedup.md), [improvements over faiss](/results/improvements-over-faiss.md).

[^paper]: The e-means paper, Eq. 5-7 and Algorithm 1.
[^faiss-emeans]: `reference/faiss_e_means/e_means.py` (`NumpyEmeans`) and `base.py` (recipe, lifecycle).
[^mlx-gemm]: candle's `mlx_gemm.rs` picks `TILE_64_32_32_2_2` for an "nt" product on base and Pro devices; `BlockMMA` gives the fragment layout.
[^parity-runs]: `bench/data/results/py_*.json`, `rs_*.json` (gitignored), compared with `bench/py/compare.py`, 2026-11-12.
[^core]: See [core engine results](/results/core-engine.md) and [CPU kernels](/results/cpu-kernels.md).
