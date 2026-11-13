---
type: Playbook
title: Adding an e-means backend, and taking it to full speed
description: The ordered steps to put e-means on a new device through the core layer — implement Device, pass the op contract first, then the speed steps with their measured gains and exactness checks — and the "Don't bother" list of steps that lost.
resource: crates/emeans-core/src/core
tags: [emeans, core, recipe, metal, cuda, exactness, benchmark]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: tensorfold
    resource: https://github.com/ashhart/TensorFold/blob/main/docs/recipes/adding-a-zig-family.md
    title: TensorFold, "Adding a Zig family, and taking it to full speed" (the format this recipe follows)
  - id: ops
    resource: bench/src/bin/ops_bench.rs
    title: Per-op timings, one 16,384-row batch, n 200k, d 768, synthetic unit rows (M1 Pro; RTX 3060 idle), 2026-11-12
  - id: core-results
    resource: /results/core-engine.md
    title: Training sweeps and parity reruns of the core layer
  - id: compare
    resource: /results/compare-mac-ubuntu.md
    title: The host-loop baseline (unfused CUDA, host loop), the source of the old numbers
  - id: metal-kernels
    resource: /results/metal-kernels.md
    title: Second Metal pass — GPU-timestamp profile, top-1 tile sweep, 9-cell sweep, exactness (M1 Pro, 2026-11-12)
  - id: cuda-kernels
    resource: /results/cuda-kernels.md
    title: CUDA kernels, second pass — nsys profile, FP16 screen + FP32 check, pipelined BF16, segmented scatter, final sweep vs faiss Triton (RTX 3060)
  - id: cpu-kernels
    resource: /results/cpu-kernels.md
    title: CPU kernels — ops_bench profiles, thread scaling and final sweeps on the 2x Xeon box and the M1 Pro, 2026-11-12
---

# How to use it

This is how a device goes onto the [e-means core](/design/core.md) and gets every speed-up we know.
The steps are in order. Each one is a measured win (or loss), so a new backend goes through them
without measuring each one again. It comes from one bring-up: the Metal engine on an M1 Pro and the
CUDA engine on an RTX 3060, 12 Nov 2026.[^tensorfold]

1. Do step 0 first. No speed step goes in before the backend passes the contract.
2. Do the steps in order. Re-run `contract::check` after each one (the backend crate's `tests/contract.rs`).
3. Time single ops with `cargo run --release -p emeans-bench --bin ops_bench`; claim a training speed
   only from whole training runs (5 seeds, interleaved with the CPU backend, a fresh process per run,
   the data upload outside the timer) on the same machine. `train_profile` times whole runs in one process.
   Time CPU backends with a binary that does not link candle's CUDA (no `cuda-baselines`, no candle
   `cuda` from another crate): it maps six CUDA libraries at start and makes multi-threaded CPU
   training ~10% slower on the x86 box. Compare old and new only between binaries with the same
   features and the same `ldd` list.[^cpu-kernels]
4. When a step proves out, add it here with its result. When one loses, add it to "Don't bother".

# What a backend writes

A new crate (say `emeans-mydev`) that depends on `emeans-core` and writes:

- `impl Device for MyCore` (buffers, `top1`, `topk`, `scatter_add`, `row_norms`, `normalize_rows`,
  `ema_update`, `split`);
- `impl Assigner for MyCore`, with `with_engine` returning `Some(f(self))`. Every `Device` is a
  `core::Engine` (blanket impl), so this one line gives training the device loop and kNN the top-k;
- a `tests/contract.rs` that calls `emeans_core::core::contract::check` and `check_production`.

Then make it selectable: from any crate, `Backend::Custom(CustomBackend { name: "mydev", make })`
with `make: fn() -> Result<Box<dyn Assigner>>`; or, for a backend that ships with the library, a
`Backend` variant and a `FromStr` name in `crates/emeans-cluster/src/backend.rs` plus the dependency
in `emeans-cluster` and the facade (by target or feature). Nothing else: the training loop, auto-k,
incremental updates, dedup and the benches run on it unchanged.

# Step 0: exact on day one

Write every op the simplest correct way and pass `contract::check`:

| Op | Gate |
|---|---|
| `top1`, `topk` | labels equal the f64 reference except near-ties (1e-5 of the score magnitude); scores within 1e-5; ties between bit-identical centroids or rows keep the lower id; also pass `check_production` (16,384 x 768, k 1024 and 1000) |
| `scatter_add`, `ema_update`, `split` | bit-equal to the CPU float order (rows added in row order; no FMA contraction: Metal `MTLMathModeSafe` + `#pragma METAL fp contract(off)`, NVRTC `--fmad=false`; IEEE divide) |
| `row_norms`, `normalize_rows` | equal to the CPU (a sequential f32 sum per row, IEEE divide and sqrt) |

Then `crates/emeans/tests/backends.rs` `backends_train_to_the_same_objective` must hold: same seed,
final objective within 1e-4 relative of the CPU backend (Metal and CUDA matched the CPU bit for bit
on that set). The CUDA engine passed all of this on its first run.

# Speed steps, in order

| # | Step | Measured gain | Exactness check |
|---|---|---|---|
| 1 | **Fused top-1**: dots and argmax in one kernel; the `rows x k` matrix never exists | Metal assignment 1.1-9x over candle matmul + argmax ([e-means algorithm](/design/emeans-algorithm.md)); CUDA 3.7 ms per 16k x 1024 batch vs candle unfused 28.8 ms per 65k batch (~7 ms per 16k) | contract `top1` |
| 2 | **Device loop** (layer 2): data, centroids, sums and counts stay on the device; sync only at revival checks and epoch ends | Comes with `Device`. Alone it is not enough on Metal (step 3). CUDA training at 200k x 1024: 1.12 s (unfused, host loop, same sweep) to 0.29 s with steps 1-6[^core-results] | `backends_train_to_the_same_objective`; CPU backend bit-equal to the streaming API |
| 3 | **Gather rows chosen by id into one contiguous block before `top1`** (Metal) | 39 ms to 14.5 ms per 16k x 1024 batch; training loop 2.58 s to 1.13 s at 200k x 1024[^ops] | none needed (a copy) |
| 4 | **Readout norms in the CPU's float order** | Parity back to the CPU table: worst faiss gap 0.52% to 0.43%, Metal runs identical to the CPU rows. Cost on Metal: normalize 0.44 to 0.85 ms (k = 1024, thread per row); for k < 256 a simdgroup per row with lane 0 summing staged squares (1.40 to 0.49 ms at k = 64). CUDA: one warp per row with lane 0: 0.22 ms at k = 1024, 0.02 ms at k = 64[^ops] | contract `row_norms`, `normalize_rows` (equal to the CPU) |
| 5 | **`split` on the device**: send the (donor, dead) pairs, not the positions both ways | CUDA: no 2 x 3 MB pageable copy per revival with splits (1.6 ms each way); with step 4, loop 364 to 332 ms at 200k x 1024[^ops] | contract `split` (zero-row nudge included) |
| 6 | **CUDA tile and centroid split**: 128 x 128 tile (127 registers, 2 blocks per SM); split the centroids over `parts` blocks per row tile when it fills the last wave (`top1_combine` keeps the lower id on ties) | 16k-row batch: 128 blocks = 2.3 waves; 3 parts: 4.10 to 3.69 ms, loop 327 to 295 ms. 1M (65k rows): 2 parts 15.3 to 14.9 ms[^ops] | contract `top1` (same labels: each dot is computed the same way) |
| 7 | **BF16 opt-in** (`cuda-bf16`, tensor cores `mma.sync.m16n8k16`, FP32 accumulate): only as an explicit choice | top-1 4.10 to 1.33 ms per batch, loop 175 ms at 200k x 1024[^ops] | not exact: `cuda_bf16_is_close_to_the_reference` (24 of 3000 labels differ, all within 1e-2) |
| 8 | **Profile with GPU timestamps before choosing** (Metal: `EMEANS_METAL_PROFILE=op` or `batch` with `train_profile`; `MetalCore::gpu_ms` with `top1_bench`) | Showed `top1` at 86% of the GPU time at 200k x 1024 and the norms above `top1` at 16k x 256; the host-side `ops_bench` timings include a round trip per op[^metal-kernels] | — |
| 9 | **Metal top-1: 32 x 32 per simdgroup, BK 32, software-pipelined** (64 x 128 x 32 tile, 2 x 4 simdgroups; the next depth slice loads into registers while the current one multiplies) | 14.5 to 8.6-9.0 ms per 16k x 1024 batch (1,770 to 2,850-3,000 GFLOP/s); training 200k x 1024 1.21 to 0.77 s[^metal-kernels] | contract `top1`; labels equal to the old tile in every row |
| 10 | **Metal tile and centroid parts per call**: 64 x 64 x 16 for k <= 64 or small batches with k <= 512; centroid parts (`top1_combine`) until a dispatch has 48 threadgroups | 16k x 64 batch: 1.12 to 0.68 ms with the small tile; 1,024 rows x 256: 0.36 to 0.22 ms with parts[^metal-kernels] | contract `top1` runs both tiles with parts |
| 11 | **Metal norms: a simdgroup per row, the sum in column order through `simd_shuffle`** (loads, max and divides on 32 lanes; every lane adds the same rounded squares in the CPU's order) | `normalize` 0.37 to 0.04 ms (k 256), 0.37 to 0.085 ms (k 1024); bit-equal to the CPU[^metal-kernels] | contract norms: 0 mismatches against the CPU float order |
| 12 | **Scatter: stable counting sort by blocks of 256 rows** (block ranks and counts, one scan of blocks x k, place) instead of one scan of all labels per cluster | count, scan and place 0.70 to 0.10 ms per 16k x 1024 batch (the sum, 0.47 ms, is unchanged)[^metal-kernels] | contract `scatter_add` (bit-equal sums, exact counts) |
| 13 | **Metal: bind the data set without a copy, owned by the engine**: `Assigner::bind_shared(Arc<...>)` keeps the `Arc` and wraps its pages with `newBufferWithBytesNoCopy` (offset into the first page); one tiny dispatch maps them | upload 0.26-0.44 s to 0.018 s; footprint 1.81 to 1.19 GiB at 200k; the first training run no longer pays the GPU mapping (25-230 ms)[^metal-kernels] | `crates/emeans-metal/tests/metal.rs` `bind_shared_owns_the_data_it_reads_in_place`: labels equal the CPU's, the engine keeps the data alive after the caller drops it, `unbind_data` releases it. `bind_data` (a borrow) and `Device::bind` copy. The owner lives in the buffer (`MBuf::keep`), not in the engine: `a_view_of_shared_data_keeps_it_alive` holds a view past `unbind_data` and the caller's drop, and drops it with work still queued. A copy is never served by (pointer, length): `rebinding_changed_data_reads_the_new_data` |
| 14 | **Metal EMA in one dispatch** (a threadgroup per cluster: positions read the old mass, then thread 0 writes it) | 2 to 1 dispatches per batch; 0.050 to 0.045 ms[^metal-kernels] | contract `ema_update` |

# CPU backend steps, in order

The CPU backend (`crates/emeans-core/src/core/cpu.rs`, `assign.rs`, `assign/fused.rs`) is the
reference float order for the exact ops, so a CPU step may change speed but never the bits of
`scatter_add`, `ema_update`, `split` or the norms; check `streaming_api_reproduces_train_bit_for_bit`
and the parity table after each one.[^cpu-kernels]

| # | Step | Measured gain | Exactness check |
|---|---|---|---|
| C1 | **Profile per op with `ops_bench`** (the ops add up to the `loop` time; `perf` was closed to users on the Linux box) | Box, `cpu-blocked` at 200k x 1024: top-1 265-267 ms per batch, nearly all of a step; Mac: top-1 17-18 ms = gather 1.4 + one sgemm 11.1 + scalar argmax 3.2 | — |
| C2 | **x86: `cpu-fused`**, a packed GEMM with the argmax fused per tile: centroids packed per call into 16-wide panels, rows gathered straight into 6-row panels, 6 x 16 AVX2+FMA microkernel (12 accumulators), `mc x nc` score tile folded into the running best; each dot one sequential FMA chain | top-1 265-267 to 22-26 ms per 16k x 1024 batch; training 200k x 1024 16.5 to 1.83 s, 1M 76.1 to 6.92 s (torch CPU 3.73 / 18.31 s) | AVX2 = portable `mul_add` twin bit for bit (unit tests); contract incl. `check_production`; equal to CUDA FP32 per epoch objective |
| C3 | **Physical-core pool** for the small ops and the `cpu-fused` top-1 (`assign::cpu_pool`, Linux with SMT, unless `RAYON_NUM_THREADS` is set) | loop 1.88-1.91 to 1.68-1.72 s (k 1024), 0.74 to 0.56 s (k 64) on the box, interleaved | none needed (same work) |
| C4 | **Small ops in tasks of ~16k floats** (`with_min_len`), and scatter as one pass per centroid (zero, add rows in row order, add into `acc`) | box: normalize 1.25 to 0.5 ms, scatter 4.1 to 3.2 ms per batch (32 threads) | `streaming_api_reproduces_train_bit_for_bit`; contract `scatter_add` |
| C5 | **Scatter: split a centroid's columns when there are few centroids, prefetch the next row** (x86 `_mm_prefetch`) | box k 64: 2.7-3.6 to 2.4-2.5 ms per batch; k 1024 unchanged | same (each sum keeps its row order) |
| C6 | **Mac Accelerate: one sgemm per 16k-row batch and a branch-free 8-lane argmax** | argmax 3.2 to 0.8 ms; top-1 17-18 to 14 ms; training 200k x 1024 1.26x faster in an interleaved A/B (k 64/256 equal) | 33 of 33 parity runs bit-identical |

# CUDA backend steps, in order

From the second CUDA pass on the RTX 3060 (the first pass is steps 1, 2, 4-7 above). Timings are per
16,384-row batch at 200k x 768, k 1024, and medians of whole training runs.[^cuda-kernels]

| # | Step | Measured gain | Exactness check |
|---|---|---|---|
| U1 | **Profile the training window with nsys** (`-t cuda`, GPU trace from the end of the data upload; Nsight Compute needs GPU counter rights) | FP32: top-1 78-83% of GPU time; BF16: top-1 under half, then per-batch conversion, scatter and 28-90 ms of GPU idle | — |
| U2 | **Scatter as a segmented stable counting sort**: a warp owns 8 clusters and one 2,048-row segment (count, scan of `segs x k`, place, each with 4 label loads in flight); place writes the resolved data row; the sum issues 8 row loads before its 8 adds | 0.47 to 0.23 ms per 16k batch, 1.74 to 0.74 ms per 65k batch | `scatter_many_segments_is_bit_equal`; contract `scatter_add` |
| U3 | **BF16: convert the data set once per bound buffer** (on the second `top1` with the same buffer; weak handle, a write drops it), rows read by id in the kernel | conversion per run 16 to 4 ms (200k), 75 to 16 ms (1M) | `bf16_cached_rows_match_converted_rows` |
| U4 | **Tensor-core kernel v2**: `cp.async` 3 stages, `ldmatrix.x4` from XOR-swizzled 64-byte rows, the (centroid tile, K step) loop flattened, centroid split; 128 x 256 tile for k >= 512 | BF16 top-1 1.33 to 0.87 ms; training 200k 0.15 to 0.10 s, 1M 0.62 to 0.39 s (Triton BF16 0.15 / 0.60) | `cuda_bf16_is_close_to_the_reference` |
| U5 | **Exact FP32 top-1 through an FP16 screen**: power-of-two-scaled FP16 operands on the U4 pipeline keep each row's candidates under a rigorous error bound; a check scores them with the FP32 kernel's exact arithmetic; rows whose candidates may be incomplete go to the FP32 kernel | FP32 top-1 3.69 to 1.39 ms; training 200k 0.29 to 0.16 s, 1M 1.34 to 0.62 s (Triton FP32 0.31 / 1.38); 0-6% of rows on the fallback | labels and scores bit-equal to the FP32 kernel: `cuda_screen_is_bit_equal_to_fp32_on_near_ties`, `screen_is_bit_equal_to_fp32`, `check_production` with the screen on and off; objectives equal to the old CUDA and `cpu-fused` per seed |
| U6 | **Check kernel: a warp per 32 rows, x tiles staged once with all 32 loads in flight**, each lane its own `fmaf` chain over the staged tile | 1.51 ms (thread per row) and 1.73 ms (staged, one load at a time) to 0.41 ms | same as U5 |
| U7 | **Host overlap in the loop**: epoch statistics on a helper thread, epoch 0's permutation drawn during the init, final reads before the last statistics join | GPU idle 1M BF16 90 to 58 ms, 200k 28 to 16 ms, 1M FP32 83 to 43 ms | `streaming_api_reproduces_train_bit_for_bit` (same statistics) |

# Don't bother

| Idea | Result | Why |
|---|---|---|
| A fused kernel alone, host EMA, sync every batch | Mac 200k x 1024: Metal 2.89 s vs CPU 2.34 s[^compare] | Per-batch upload of centroids and ids, a device sync and the EMA on the host eat the kernel's lead. Move the loop (step 2) |
| Device loop with rows read by id inside the Metal kernel | loop 2.58 s, slower than the old host loop | The kernel re-reads its rows once per centroid tile; scattered rows cost 2.8x on the M1 Pro. Gather first (step 3) |
| Gathering rows on CUDA | 4.10 vs 4.09 ms per batch | Rows read by id are already fast on the 3060; the copy only adds traffic |
| Tree-summed norms on the GPU | 0.44 ms (fast), but worst faiss gap 0.43% to 0.52% | 1-ulp centroid differences flip near-ties; the run leaves the CPU's trajectory |
| Metal norms staged 32 x 32 through threadgroup memory, or a simdgroup per row with lane 0 for large k | 1.98 ms and 1.83 ms vs 0.85 ms (k = 1024) | Too few threads or a long serial chain per threadgroup |
| Smaller CUDA tiles for 16k-row batches | 64 x 128: 4.75, 128 x 64: 4.76, 64 x 64: 5.78 ms vs 128 x 128: 3.96 ms | Less register reuse; the wave tail is better fixed by the centroid split (step 6) |
| A BF16 screen for the exact FP32 path (u = 2^-8, delta 0.017) | 6-20% of rows on the FP32 fallback at k 1024 on the bench files; top-1 4.6 ms, slower than FP32 alone[^cuda-kernels] | The bound is 6x wider than FP16's; screen in FP16 with power-of-two scales (U5) |
| The FP32 fallback over all centroids in one part | 1.41 ms per batch for 6% of the rows[^cuda-kernels] | A few row blocks leave the GPU empty; split the centroids in up to 8 parts (0.30 ms) |
| BF16 v2 with 4 stages, or a 128 x 128 tile at k 1024 | 1.16 and 1.09 ms vs 0.87 ms (128 x 256 x 3)[^cuda-kernels] | 4 stages drop to 1 block per SM; the 256-wide tile reuses each row tile twice as long |
| Metal top-1: double-buffered threadgroup memory, transposed centroid staging with float2 fragment loads, `simdgroup_load`, BK 16 with the 128-wide tiles, 32 x 64 per simdgroup | 13.6, 9.3-9.7, 10.5-11.0, 9.2-10.6, 18.1-18.5 ms vs 8.6-9.0 ms (16k x 1024)[^metal-kernels] | Half the occupancy, no gain, no gain, twice the barriers, register spill |
| Metal top-1 reading rows by id with the 64 x 128 tile (no gather) | 9.94 ms vs 8.81 + 0.64 ms gather at 200k x 1024; 1.12 vs 1.11 + 0.09 ms at 16k x 1024[^metal-kernels] | Fewer centroid tiles shrink the penalty, but gathering still wins on large batches |
| Spin-waiting on the last command buffer instead of `waitUntilCompleted` | 16k x 64 train 27-29 ms either way[^metal-kernels] | The idle gaps at revival syncs are not the wake-up latency |
| A no-copy view of borrowed data (`Device::bind`, `bind_data(&[f32])`) | Row 0 of a gathered batch read as the allocator's free-list words (`tests/metal.rs`) | The borrow ends before the GPU reads: `core::assign_on` frees its rows before reading the labels, and safe callers can drop or change a bound `Vec`. Only an owned `Arc` (`bind_shared`) makes the view sound, and only when the buffer carries the `Arc`: with the owner kept in the engine, a clone from `Device::bind` outlived `unbind_data` |
| A cache of copied data keyed by (pointer, length) | After the caller freed the data, a new buffer of the same size at the same address was served the old copy (1,467 of 1,500 labels wrong in the test) | An address identifies data only while something pins it. Serve by address only a buffer that holds the `Arc`; copy otherwise. The CUDA `bind_data` had the same early return; it now always uploads a fresh copy (`crates/emeans-cuda/tests/cuda.rs` `rebinding_changed_data_reads_the_new_data` failed on the old code), and `Device::bind` serves that copy by address only while it is bound |
| CubeCL for the fused kernel (spike with CubeCL 0.11.4: `cmma` f32 on Metal, FMA f32 and bf16 on CUDA) | Metal 0-6% behind ours; CUDA FMA f32 equal to faiss Triton FP32; CUDA bf16 3.2x behind Triton BF16 after a tile sweep | Keep hand-written kernels per backend |
| faiss torch on MPS (as a GPU baseline on the Mac) | Fails: faiss keeps statistics in float64, MPS has none[^compare] | Not patched: it would change faiss's numerics |
| Mac: Accelerate per thread on row tiles (gather, sgemm, argmax per tile; 1-8 threads, tiles 256-2048) | 17.8 ms best vs 13.3 ms (gather 1.4 + one sgemm 11.1 + lane argmax 0.8, timed separately)[^cpu-kernels] | Accelerate already spreads one sgemm over the AMX units; concurrent calls compete |
| Mac: overlap gather and argmax with the sgemm (helper thread, 2-8 blocks) | 15.3-17.9 ms vs 13.3 ms for the same steps in sequence[^cpu-kernels] | The NEON work slows the AMX sgemm more than it hides |
| Mac: centroids pre-transposed (`d x k`, NoTrans) for sgemm | 12.5 vs 11.9 ms[^cpu-kernels] | Accelerate handles the transposed operand itself |
| Mac: `cpu-fused` instead of Accelerate | top-1 6-32 ms (k 64), 48-86 ms (k 1024) vs 3.3-4.5 and 16-17 ms[^cpu-kernels] | AMX beats the NEON cores; Accelerate stays the Mac default |
| x86 box: 64 threads (hyperthreads) for `cpu-fused` | top-1 equal (24-25 ms), loop slower (1.88-1.91 vs 1.68-1.72 s)[^cpu-kernels] | FMA-bound kernel; extra threads only cost wake-ups. The scalar `cpu-blocked` is the opposite: it keeps the global pool |
| x86 box: `cpu-fused` tile sweep (mc 48-192, nc 64-256, kc 128-768), NUMA interleave | 21.8-30.9 ms, noise ~3 ms; interleave 24.0-26.2 ms[^cpu-kernels] | The kernel holds 65% of the FMA peak at every thread count; the per-core loss is the AVX clock (3.0 to 1.7 GHz) |
| Comparing CPU timings of a CPU-only build with a build that links candle's CUDA | Box 200k x 1024 `cpu-fused`: 1.78 s (old, no CUDA) vs 2.09 s (new, `cuda` + candle baselines) looked like a regression from the crate split; the new code without candle CUDA ran 1.81 s[^cpu-kernels] | candle forces cudarc `dynamic-linking`; loading the CUDA libraries alone (`LD_PRELOAD`) costs the same 8-12%, at 32 threads but not at 1. Keep candle's CUDA out of CPU benchmark binaries (`cuda-baselines` is separate) |
| x86 box: scatter columns split by each cluster's row count | 2.7-3.1 ms either way[^cpu-kernels] | Not imbalance: the scattered 3 KB row reads (~20 GB/s) bound it |

[^tensorfold]: Same format as TensorFold's family recipe and the candle-embeddings library's family recipe.
[^ops]: `cargo run --release -p emeans-bench --bin ops_bench -- --backend metal|cuda|cuda-bf16 --k 1024` (and `EMEANS_CUDA_TILE`, `EMEANS_CUDA_PARTS`, `EMEANS_CUDA_GATHER` for the sweeps); median of 5-9 reps, synced after each op.
[^core-results]: [Core engine results](/results/core-engine.md): whole training runs, median of 5 seeds.
[^compare]: [Host-loop baseline](/results/compare-mac-ubuntu.md).
[^metal-kernels]: [Metal kernels](/results/metal-kernels.md): GPU-timestamp profiles, the top-1 tile sweep and the 9-cell sweep.
[^cuda-kernels]: [CUDA kernels](/results/cuda-kernels.md): nsys training-window profiles, `ops_bench`, fallback statistics (`EMEANS_CUDA_SCREEN_STATS=1`) and the final sweep.
[^cpu-kernels]: [CPU kernels](/results/cpu-kernels.md): `ops_bench` profiles and thread sweeps, `cpu_probe` (Mac), whole-run sweeps, parity reruns.
