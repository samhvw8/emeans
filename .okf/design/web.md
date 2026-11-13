---
type: Design
title: Web version — emeans on WebAssembly with a JavaScript/TypeScript API
description: bindings/wasm, the CPU backend compiled to wasm32-unknown-unknown and exported with wasm-bindgen to the browser and Node.js; what emeans-core changed for it, the tooling and why, bit equality with native Rust, size and timings, and the WebGPU and threads research.
resource: bindings/wasm
tags: [emeans, wasm, webassembly, browser, nodejs, bindings, webgpu, exactness]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T10:00:00Z'
sources:
  - id: code
    resource: bindings/wasm
    title: src/api.rs (binding API in plain Rust), src/web.rs (wasm-bindgen exports, TypeScript types), examples/golden.rs, tests/*.test.mjs, demo/, scripts/
  - id: exec
    resource: crates/emeans-core/src/exec.rs
    title: Helper threads inline and web-time Instant on wasm
  - id: wasm-bindgen
    resource: https://github.com/wasm-bindgen/wasm-bindgen/releases/tag/0.2.130
    title: wasm-bindgen 0.2.130 (2026-10-30); the CLI and the crate must be the same version
  - id: wasm-pack
    resource: https://github.com/wasm-bindgen/wasm-pack/releases
    title: wasm-pack moved from rustwasm to the wasm-bindgen organization; v0.15.1 (2026-10-30), not archived
  - id: wasm-bindgen-rayon
    resource: https://github.com/RReverser/wasm-bindgen-rayon
    title: wasm-bindgen-rayon 1.3.0 (2024-12); needs nightly Rust, -Z build-std, +atomics, and cross-origin isolation (COOP/COEP) for SharedArrayBuffer
  - id: rayon-core
    resource: https://github.com/rayon-rs/rayon/blob/main/rayon-core/src/registry.rs
    title: rayon-core 1.13 default_global_registry — falls back to the current thread where threads are unsupported (wasm)
  - id: wgpu
    resource: https://docs.rs/wgpu/30.0.2/wgpu/struct.Device.html
    title: wgpu 30.0.2 — Device::poll is a no-op on WebGPU; map_async resolves only on the JS event loop
  - id: jspi
    resource: https://github.com/wasm-bindgen/wasm-bindgen/pull/5193
    title: wasm-bindgen 0.2.128 experimental JSPI exports and jspi_block_on_promise
  - id: wgsl
    resource: https://www.w3.org/TR/WGSL/
    title: WGSL floating-point accuracy (fma may be split; reassociation and fusion allowed; x / y 2.5 ULP)
  - id: webgpu-status
    resource: https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
    title: WebGPU implementation status per browser and platform
---

# Overview

`bindings/wasm` builds the CPU backend of emeans for `wasm32-unknown-unknown` and exports it
to JavaScript with wasm-bindgen.[^code] One ES module (`--target web`) runs in the browser
and in Node.js (Node passes the `.wasm` bytes to `initSync`). It is its own Cargo workspace,
so the root `cargo test --workspace` never builds it. Nothing is published.

| Layer | File | Role |
|---|---|---|
| API in plain Rust | `src/api.rs` | Options as JSON over the Rust config defaults, the calls, input checks. Native code (the golden program) calls the same functions |
| JS exports | `src/web.rs` | Typed arrays in and out, `Model` and `Hierarchy` classes, plain result objects, TypeScript types (custom section), thrown `Error`s, a panic hook that prints to the console |
| Tests | `examples/golden.rs`, `tests/*.test.mjs` | Native reference outputs, then the wasm run on the same inputs, compared byte for byte |
| Demo | `demo/index.html`, `scripts/check-demo.mjs` | A static page that clusters synthetic points; a headless Chrome check over CDP |

# API

| JavaScript | Rust |
|---|---|
| `train(x, d, k, options?)` → `Model` | `Emeans::with_assigner(Config, CpuAssigner)` + `train` |
| `Model`: `assign`, `centroids`, `trainLabels`, `emaState`, `stats`, `toBytes`, `Model.fromBytes` | `assign`, `Emeans::to_bytes` / `from_bytes_with` |
| `autoCluster(x, d, options?)` → `{ hierarchy, labels }` | `auto_cluster` |
| `Hierarchy`: `assign`, `add`, `toBytes`, `Hierarchy.fromBytes`, `chosenK`, `diagnostics`, centroids | `Hierarchy::add`, `to_bytes` / `from_bytes` |
| `dedup(x, d, options?, inputs?)` | `dedup` |

Options are plain objects whose keys are the Rust config fields in camelCase, the same
JSON-override idea as the [native bindings](/design/bindings.md). `api.rs`
serializes the Rust default, writes the given keys over it and deserializes the result, so a
new config field needs no binding change. An unknown key fails. `kernel` picks the CPU
kernel: `gemm` (the default since 2026-11-13, team-lead's decision: the fastest on wasm),
`fused` (bit-identical to native `cpu-fused`) or `blocked`. `blobs(n, d, centers, seed)`
makes synthetic rows from integer draws with exact `f32` values, so native, Node and
browser runs of the benchmark train on the same bits without data files.

# Changes to emeans-core and emeans-cluster

| Change | Why | Native effect |
|---|---|---|
| `exec` module: `exec::spawn`, `exec::spawn_scoped`, `exec::Instant`[^exec] | `std::thread::spawn` and `std::time::Instant::now` panic on `wasm32-unknown-unknown`. The device loop's helper threads (next epoch's permutation, epoch statistics) run inline at their join point there; the clock is `performance.now()` through `web-time` | None: the same threads, the same arithmetic |
| `CpuKernel::fastest()` is `Gemm` on wasm | Measured fastest there (see below); the binding's default too | None |
| `Emeans::to_bytes` / `from_bytes_with` | A browser has no file system. `save` / `load_with` now call them; the file format is unchanged | Loading checks sizes with checked arithmetic, so a corrupt header cannot wrap on 32-bit targets |
| `Hierarchy::to_bytes` / `from_bytes` | One blob: magic `EMHIER\0\x01`, manifest JSON, then each model's file bytes with a length prefix | `save` / `load` share the manifest and assembly code with it |
| `auto.rs` uses `exec::Instant` | Timing of the k sweep | None |

rayon needs no change: rayon-core falls back to a one-thread pool on the current thread
where threads are unsupported.[^rayon-core] The `gemm` crate (used off macOS for `gemm_nt`)
builds for wasm with its `wasm-simd128-enable` feature; the build enables `+simd128`.

# Tooling

| Need | Choice | Why |
|---|---|---|
| Bindings | wasm-bindgen 0.2.130, `--target web`[^wasm-bindgen] | Current; one ES module for the browser and Node |
| Build driver | mise tasks calling `cargo`, `wasm-bindgen` and `wasm-opt` | wasm-pack is alive again under the wasm-bindgen organization[^wasm-pack], but three commands need no extra tool; `bindings/wasm/mise.toml` pins the CLI to the crate version |
| Size | `wasm-opt -O3` (binaryen version_133) | |
| Tests | Node.js `node:test` on the built package; a native golden program | Tests the shipped artifact; no wasm-bindgen-test runner needed |
| Browser check | headless Chrome over the DevTools protocol from Node (`scripts/check-demo.mjs`) | No Puppeteer dependency |

# Exactness

`mise run test` in `bindings/wasm` runs `examples/golden.rs` natively (the same `api`
functions) for every kernel the host builds, then the wasm build in Node on the same inputs
and options, and compares every output byte for byte (`tests/golden.test.mjs`, report in
`out/golden-report.json`). It ran against two native hosts: an M1 Pro (aarch64 macOS) and
the Ubuntu box (2x Xeon E5-2683 v4, x86-64 with AVX2 and FMA).

| Case | Data | Outputs compared |
|---|---|---|
| `train` + `assign`, L2 | 20,000 x 32 Gaussian mixture, k = 64, 3 epochs | centroids, last-pass labels, EMA positions and masses, assign labels and scores, model bytes (header without the wall-clock `time`, and payload) |
| `train` + `assign`, spherical | 12,000 x 64 nested unit rows, k = 48, 2.5 epochs, `maxPointsPerCentroid` 200 | same |
| native model file -> `Model.fromBytes` -> `assign` | both cases | scores |
| `autoCluster` -> `toBytes` -> `Hierarchy.fromBytes` -> `add` | 2,000 + 400 nested unit rows, d = 48 | every `Labels` field after the fit and after `add`, chosen k, action |
| `dedup` with times | 3,000 x 64 rows plus bursts | `groupOf`, `representatives`, linked pairs, `nnSim` |

| wasm kernel | vs native aarch64 macOS | vs native x86-64 Linux |
|---|---|---|
| `fused` | Same bits in every output except `nnSim` (1 ulp) | Same bits in every output except `nnSim` (3 ulps) |
| `blocked` | Differs (native uses NEON FMA tiles) | Same bits except `nnSim` (3 ulps) |
| `gemm` (default) | No native `gemm` on macOS | Differs from the first centroid on; labels differ too |

Why `fused` matches: every dot product is one `mul_add` chain in a fixed order on any
instruction set (the AVX2 microkernel, the portable loop on aarch64, the software `fmaf`
on wasm), and the rest of training (scatter sums in row order, EMA, splits, MT19937 draws)
is plain IEEE arithmetic in a fixed order; Rust never contracts `a * b + c` on its own.
`blocked` matches off aarch64 because both sides run the same portable multiply-add loop.
`gemm` does not match native `cpu-gemm`: the `gemm` crate picks its microkernels per
instruction set (AVX2+FMA on the box, simd128 multiply-add on wasm), the dot products
differ in the last bits, and over the epochs a few rows change cluster. wasm `gemm` gives
the same bits in every wasm engine and machine (centroid hashes in the benchmark).

`dedup` and the kNN under it multiply with `gemm_nt`: Accelerate on macOS, the `gemm`
crate elsewhere, so `nnSim` can differ in the last bits; the groups were the same. In the
`autoCluster` case `gemm_nt` only scores candidate k (silhouette), and the scores chose the
same k; on other data a near tie could choose differently.

# Size and timings

`emeans_bg.wasm` after `wasm-opt -O3`, with simd128: 784 KiB raw, 295 KiB gzip, 230 KiB
brotli (JS glue 27 KiB, `.d.ts` 11 KiB). The mise build remaps `$HOME` to `~` in panic
locations (`CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS`); before that the module held 39
absolute home paths, and a test now fails on `/Users/` or `/home/`. The size budget test fails above 1 MiB raw or
400 KiB gzip. Not measured: which crates take the space (the `gemm` crate links kernels
for f16, f64 and complex types too).

Timings in ms on an M1 Pro (8 performance cores), synthetic data from the golden run; wasm
is one thread in Node 26.13.1 (median of 3), native is `cargo run --release`:

| Case | wasm `fused` | wasm `blocked` | wasm `gemm` | native `fused`, 1 thread | native `fused`, all cores | native Accelerate, all cores |
|---|---|---|---|---|---|---|
| L2 train (20,000 x 32, k = 64, 3 epochs) | 170 | 56 | 14 | 16 | 13 | 18 |
| L2 assign (20,000 rows) | 55 | 17 | 3.7 | 4.0 | 0.84 | |
| Spherical train (9,600 sampled x 64, k = 48, 2.5 epochs) | 102 | 37 | 6.9 | 8.8 | 9.0 | 13 |
| Spherical assign (12,000 rows) | 49 | 17 | 2.4 | 3.1 | 0.80 | |
| `autoCluster` (2,000 x 48, k1 2..8, 2 restarts) | 429 | | | 123 | 328 | |

In headless Chrome 157 the demo (50,000 x 16, k = 12, 5 epochs) trains in 117 ms and assigns
in 19 ms with `fused`; with `gemm` it trains in 35 ms and assigns in 3 ms.

Reading: `fused` on wasm is 12x slower than `gemm` here (21-26x at d = 768) because wasm has no scalar FMA
instruction: `f32::mul_add` becomes a call to the software `fmaf`. The `gemm` crate's simd128
kernels run at the speed of native one-thread `fused`. `CpuKernel::fastest()` is therefore
`Gemm` on wasm, and so is the binding's default; `fused` is the opt-in for native bits. The
benchmark at d = 768 is in [wasm benchmark](/results/wasm-bench.md).

Next step for speed with native bits: a `fused` microkernel on relaxed SIMD
(`f32x4.relaxed_madd`), which current V8 and SpiderMonkey run as a hardware FMA. It needs a
second `.wasm` (an engine without relaxed SIMD rejects the whole module), feature detection
in the loader, and a start-up check that the engine fuses (else fall back to the exact build).
The `gemm` timings above show the speed such a kernel can reach.

# Threads

Not built. Wasm threads in Rust still need nightly, `-Z build-std` with `+atomics,+bulk-memory`,
and a page served with COOP/COEP headers so `SharedArrayBuffer` exists.[^wasm-bindgen-rayon]
An opt-in threaded build (wasm-bindgen-rayon, a second `.wasm`) is possible later. The fused
kernel and the scatter sums keep the same bits at any thread count, so threads would not
change results.

# WebGPU (phase 2, research only)

| Question | Finding |
|---|---|
| Rust or JS | wgpu 30 with only the `webgpu` feature (no naga or wgpu-core in the wasm) can implement `core::Device`, so `train.rs` keeps the loop; the same crate tests natively on Metal or Vulkan. A TS device either repeats the readback problem in two languages or duplicates the training loop |
| Readback | `Device::poll` is a no-op on WebGPU; `map_async` resolves only on the JS event loop, so a synchronous `read_f32` cannot work.[^wgpu] Options: JSPI (`jspi_block_on_promise`, experimental since wasm-bindgen 0.2.128; Chrome 137+, Firefox 153+, Safari 27+, Node 25+),[^jspi] or async `read_*` in `train.rs` and every backend |
| Browsers | Chrome/Edge on macOS, Windows, ChromeOS, Android 12+; Linux only on some GPUs. Firefox on Windows and Apple Silicon. Safari 26 on macOS and iOS.[^webgpu-status] Node: the Dawn-based `webgpu` package |
| Limits | 128 MiB storage binding and 256 MiB buffer by default: about 43k rows of d = 768 per binding; 16 KiB workgroup memory (the Metal 64x128x32 tile needs about 27 KiB, so use 64x64x16) |
| Numerics | WGSL may split `fma`, fuse, reassociate and flush subnormals; division is 2.5 ULP.[^wgsl] top-1 is fine under the contract's 1e-5 tolerance; the exact ops (`scatter_add`, `ema_update`, `split`) cannot be guaranteed bit-equal and need a ULP tier in `core::contract` |
| Effort | A new crate `emeans-wgpu`: top-1, scatter-add, EMA and the small ops (about 550 WGSL lines and 1,000 lines of Rust), JSPI readback, contract tests natively and in headless Chrome: 9-11 days |
| Risks | JSPI is experimental; no matrix-multiply units in WebGPU, so top-1 runs well below Metal; data above the binding limit must be split; long dispatches can lose the device |

Not built: the effort is not small, and the readback question (JSPI or an async `Device`)
is a design decision for the whole library.

[^code]: bindings/wasm sources.
[^exec]: crates/emeans-core/src/exec.rs.
[^wasm-bindgen]: wasm-bindgen 0.2.130 release.
[^wasm-pack]: wasm-pack releases under the wasm-bindgen organization.
[^wasm-bindgen-rayon]: wasm-bindgen-rayon README.
[^rayon-core]: rayon-core registry fallback.
[^wgpu]: wgpu 30.0.2 Device docs.
[^jspi]: wasm-bindgen JSPI pull request.
[^wgsl]: WGSL specification, floating-point accuracy.
[^webgpu-status]: gpuweb implementation status.
