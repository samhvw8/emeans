---
type: Benchmark Result
title: WebAssembly vs native — train time, memory and centroid bits at d = 768
description: The wasm build (bindings/wasm) in Node.js and headless Chrome against native Rust on the same machine, same data and seed, n = 16k / 100k / 200k, d = 768, k = 256 / 1024, kernels gemm (the wasm default) and fused (bit-identical to native); on the 2x Xeon box and on an M1 Pro.
resource: bindings/wasm/scripts/bench-run.mjs
tags: [benchmark, emeans, wasm, webassembly, nodejs, chrome, exactness, memory]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T10:35:00Z'
sources:
  - id: box
    resource: bindings/wasm/scripts/bench-run.mjs on the Ubuntu box, 2026-11-13 09:06-10:09, out/bench-box.json
    title: 2x Xeon E5-2683 v4 (32 cores, 64 threads), Node 26.13.0, Chrome 155, Rust 1.99.1; re-runs of 200k x 1024 gemm (10:23-10:29) and of the 16k shapes under the box lock (out/bench-box-16k.json, out/bench-box-16k-256.json, out/rerun16k-256-node.log)
  - id: mac
    resource: bindings/wasm/scripts/bench-run.mjs on the M1 Pro through quiet-run, 2026-11-13 09:09-09:14, out/bench-mac.json
    title: M1 Pro (8 performance + 2 efficiency cores), Node 26.13.1, Chrome 157, Rust 1.99.1
  - id: golden
    resource: bindings/wasm/tests/golden.test.mjs
    title: Byte-for-byte golden tests against native on both hosts
---

# Setup

- **Data:** `blobs(n, 768, 512 centers, seed 1234)` from `bindings/wasm/src/api.rs`. It uses
  integer xorshift draws, and every value is exact in `f32`. So native Rust, Node and Chrome
  generate the same bits in-process, and no data files are needed.
- **Training:** `train` with the defaults (5 epochs, batch law auto) and seed 1234. Each run
  is its own process.
- **Native:** `cargo run --release --example bench`, at 1 thread (`RAYON_NUM_THREADS=1`) and
  at all cores. On the box, "all cores" means rayon's 64 threads; `fused` runs on its pool
  of 32 physical cores.
- **wasm:** one thread in Node.js (`scripts/bench-node.mjs`) and in headless Chrome
  (`demo/bench.html` over the DevTools protocol, `scripts/chrome.mjs`).
- **Threads:** wasm has no threaded build. Wasm threads need nightly Rust, `-Z build-std`
  and COOP/COEP headers, see [web version](/design/web.md).
- **Bit check:** the SHA-256 of the final centroids.
  - e-means centroids are closing means of the rows each cluster holds, and the EMA
    positions move only by assigned rows. So equal centroids mean that every assignment
    decision of the run matched.
  - The [golden tests](/design/web.md) compare the scores bit for bit as well.
- **Size:** `emeans_bg.wasm` is 784 KiB raw (295 KiB gzip, 230 KiB brotli) after
  `wasm-opt -O3`. It holds no absolute path: the mise build remaps `$HOME` to `~`, and a
  test checks for `/Users/` and `/home/`.
- **Commands:**
  - full run: `node scripts/bench-run.mjs --slow-max-n 16000 --slow-jobs 6`; on the box,
    wrap it in `flock /tmp/emeans-box-gpu.lock`;
  - Mac: `node scripts/bench-run.mjs --shapes 16000x256,16000x1024` through `quiet-run`.

# Results: 2x Xeon E5-2683 v4 (box)

Train time in seconds:[^box]

| n | k | native `fused` 1 thread | native `gemm` 1 thread | native `fused` all cores | native `gemm` all cores | wasm `gemm` Node | wasm `gemm` Chrome | wasm `fused` Node | wasm `fused` Chrome |
|---|---|---|---|---|---|---|---|---|---|
| 16,000 | 256 | 0.74 | 1.25 | 0.36 | 0.51 | 2.05 ³ | 2.07 | 53.3 | 54.7 |
| 16,000 | 1024 | 2.36 | 2.93 | 0.53 | 0.58 | 7.26 | 7.44 | 214 | 217 |
| 100,000 | 256 | 4.17 | 6.64 | 0.47 | 1.35 | 13.1 | 12.9 | 357 ¹ | |
| 100,000 | 1024 | 13.6 | 18.3 | 1.10 | 3.25 | 46.3 | 45.3 | 1,355 ¹ | |
| 200,000 | 256 | 7.86 | 12.5 | 0.70 | 3.12 | 25.8 | 25.3 | 697 ¹ | |
| 200,000 | 1024 | 26.5 | 36.9 | 1.71 | 6.21 | 92.5 ² | 91.0, 120.5 ² | 2,724 ¹ | |

The 16,000 rows come from re-runs under `flock /tmp/emeans-box-gpu.lock` (the box's timing
lock): 16,000 x 1024 at 10:36-10:44, 16,000 x 256 at 10:47-10:50. Two earlier runs of the
16k rows overlapped another agent's builds (09:10, 10:34-10:35); all runs gave the same
hashes and agree within 8%. The first full run (09:06-10:09) did not hold the lock; apart
from the overlaps named here the box was quiet (load under 2).

³ Median of 3 runs (2.02, 2.06, 2.05 s) at 10:50 under the lock; in the locked 10:47 batch
this cell took 5.51 s once, with nothing else running (cause not found).

¹ These wasm `fused` Node runs ran as 6 single-thread processes at the same time, after the
other runs. At 16,000 x 256 the parallel batch took 57.3 s against 53.3 s alone, so the
sharing cost about 7%.

² Re-run at 10:23-10:29 on a quiet box: Node 92.7 s and 92.3 s; Chrome 120.5 s. The first
Node run (238 s) overlapped another agent's test build on the box and is discarded. The two
Chrome runs (91.0 s, 120.5 s) both ran on a quiet box (load under 2). That spread is real
variance at 0.7 GiB on this two-socket machine; the cause, perhaps memory placed on the
other NUMA node, was not checked.

Peak memory in GiB. Native and Node show the peak RSS of the process. wasm shows its linear
memory, which only grows, so its final size is its peak:

| n | k | data (n x 768 x 4 B) | native `fused` | Node, wasm `gemm` (RSS) | wasm linear memory, `gemm` |
|---|---|---|---|---|---|
| 16,000 | 256 | 0.05 | 0.06 | 0.16 | 0.06 |
| 16,000 | 1024 | 0.05 | 0.07 | 0.19 | 0.09 |
| 100,000 | 256 | 0.29 | 0.30 | 0.67 | 0.33 |
| 100,000 | 1024 | 0.29 | 0.32 | 0.71 | 0.37 |
| 200,000 | 256 | 0.57 | 0.59 | 1.28 | 0.64 |
| 200,000 | 1024 | 0.57 | 0.61 | 1.34 | 0.71 |

Centroid SHA-256, first 12 hex digits:

| n | k | native `fused` (1 and 64 threads) | wasm `fused` (Node, Chrome) | native `gemm` | wasm `gemm` (Node, Chrome; also the Mac) |
|---|---|---|---|---|---|
| 16,000 | 256 | `3d58a7adc62e` | `3d58a7adc62e` | `0bb96ef63b6e` | `10b30cc6ebc1` |
| 16,000 | 1024 | `3ea6757656b9` | `3ea6757656b9` | `44f2732599ed` | `b9a94b6bdcf4` |
| 100,000 | 256 | `53f91a6823bb` | `53f91a6823bb` | `1a72ef5fef24` | `1a72ef5fef24` |
| 100,000 | 1024 | `0c9e82eb3432` | `0c9e82eb3432` | `0da47109c700` | `9c9d06b0652a` |
| 200,000 | 256 | `3587a7350730` | `3587a7350730` | `f6982bc5eab6` | `4677cb0bda84` |
| 200,000 | 1024 | `2dd4842ade67` | `2dd4842ade67` | `40586ab8694b` | `d97a29db2229` |

# Results: M1 Pro (Mac)

Run through `quiet-run`, 16k only (the Mac is the user's workstation):[^mac]

| n | k | native `fused` 1 thread | native Accelerate 1 thread | native `fused` 10 threads | native Accelerate 10 threads | wasm `gemm` Node | wasm `gemm` Chrome | wasm `fused` Node | wasm `fused` Chrome |
|---|---|---|---|---|---|---|---|---|---|
| 16,000 | 256 | 1.39 | 0.19 | 0.30 | 0.12 | 1.00 | 1.00 | 21.0 | 20.8 |
| 16,000 | 1024 | 4.85 | 0.28 | 0.86 | 0.14 | 3.46 | 3.45 | 84.4 | 82.5 |

The centroid hashes equal the box's in every column: `fused` native and wasm, and wasm
`gemm`. Native Accelerate also gave the `fused` hash on this data.

# Reading

- **Default kernel (`gemm`) speed:**
  - On the Xeon, wasm `gemm` takes 2.8-3.4x the time of native `fused` on one thread, and
    1.6-2.5x that of native `gemm` on one thread.
  - Against native on all cores it takes 5.7x as long at 16k x 256, 14x at 16k x 1024,
    and 28-53x at 100k and up.
  - On the M1 Pro, wasm `gemm` is faster than native `fused` on one thread (0.7x). The
    portable `fused` loop has no hand-written NEON kernel. Accelerate on one thread is
    5-12x faster than wasm.
- **`fused` speed:** wasm `fused` is 72-103x slower than native `fused` on the Xeon and
  15-17x on the M1. wasm has no FMA instruction, so every `mul_add` is a software `fmaf`
  call. Use `fused` only when the bits must equal native. The plan for a fast exact kernel
  (relaxed SIMD) is in [web version](/design/web.md).
- **Node and Chrome:** within 5% of each other for the same kernel. Both run V8. Chrome on
  macOS and Linux gives the same timings shape and the same bits.
- **Memory:**
  - wasm linear memory is the data plus 12-30% at 100k rows and up; at 16k the fixed
    buffers weigh more. The binding copies the input into wasm memory.
  - A Node process holds two copies, the JS `Float32Array` and the wasm one, so its RSS is
    about twice the data.
  - wasm32 memory stops at 4 GiB. At d = 768 that limits one call to about 1.2 million rows
    (an estimate from the measured overhead, not tested).
- **Bits:**
  - wasm `fused` gave the same centroids as native `fused` at every shape, at every native
    thread count, in both engines and on both machines.
  - wasm `gemm` gave the same centroids in Node and Chrome, on x86-64 Linux and on aarch64
    macOS. So its bits do not depend on the engine or the machine.
  - wasm `gemm` differs from native `gemm` in 5 of 6 shapes. The `gemm` crate picks other
    microkernels per instruction set. At 100k x 256 no assignment decision flipped, so the
    hashes matched.

[^box]: bench-run.mjs on the Ubuntu box, out/bench-box.json.
[^mac]: bench-run.mjs on the M1 Pro, out/bench-mac.json.
