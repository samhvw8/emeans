---
type: Reference
title: Getting started — emeans
description: What the emeans library is, its crate map, how to build and test it, and where to read next.
tags: [getting-started, e-means, clustering, rust, metal, cuda]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T08:00:00Z'
---

# Overview

emeans is e-means (mini-batch k-means with scaling laws, faiss `contrib/e_means`, MIT) in Rust, on
CPU, Metal and CUDA, plus algorithms on top: automatic k in two levels, incremental assignment with
stable cluster ids, near-duplicate groups and exact kNN. It repeats faiss NumPy's random draws, so a
run with the same seed matches faiss draw for draw, and every backend gives the same objective as
the CPU for the same seed ([e-means algorithm](design/emeans-algorithm.md), [core](design/core.md)).

Inputs are plain row-major `f32` slices (`n x d`), outputs plain structs; nothing in the public API
is generic, so the Python, Node.js and Go bindings ([bindings](/design/bindings.md)) wrap it as is.

```rust
use emeans::{Backend, Config, Emeans, make_assigner};
let mut cfg = Config::new(d, k);
cfg.spherical = true;
let mut model = Emeans::with_assigner(cfg, make_assigner(Backend::cpu())?)?;
let trained = model.train(&x)?;          // centroids, last-pass labels
let labels = model.assign(&x)?.labels;
```

# Crate map

| Crate | What | Read |
|---|---|---|
| `crates/emeans-core` | `Emeans` model, recipe, RNG, IO, CPU kernels and `gemm_nt`, CPU kNN, the core layer (`Device` op contract, f64 reference, contract checks, device-resident training loop, `Engine` trait). No candle | [core](design/core.md), [e-means algorithm](design/emeans-algorithm.md) |
| `crates/emeans-metal` | `MetalCore`, `MetalKnn` (macOS; empty elsewhere) | [Metal kernels](results/metal-kernels.md) |
| `crates/emeans-cuda` | `CudaCore` on `cudarc` (builds without a CUDA toolkit; errors at run time without a driver) | [CUDA kernels](results/cuda-kernels.md) |
| `crates/emeans-cluster` | `auto_cluster`, `Hierarchy`, incremental updates, `dedup`, `knn`, metrics, synthetic data, `Backend` / `make_assigner` | [automatic k, incremental, dedup](design/autok-incremental-dedup.md) |
| `crates/emeans` | Facade over all of them; feature `cuda`, Metal by target | [crate layout](decisions/crate-layout.md) |
| `bench/` (`emeans-bench`, not published) | Benchmark and parity binaries; candle baselines behind feature `baselines`; `bench/py/` Python reference scripts | [backend recipe](recipes/adding-an-emeans-backend.md) |
| `reference/faiss_e_means` | Upstream faiss source, MIT, pinned commit; read-only | [e-means algorithm](design/emeans-algorithm.md) |
| `bindings/wasm` (own Cargo workspace, not published) | The CPU backend on WebAssembly with a JavaScript/TypeScript API for the browser and Node.js; `mise run build`, `mise run test`, `mise run example:wasm` in that folder | [web version](design/web.md), [wasm benchmark](results/wasm-bench.md) |

# Build and test

| Task | Command |
|---|---|
| Format, lint, test | `mise run check` (`cargo fmt --check`, `cargo clippy -D warnings`, tests) |
| Tests only | `mise run test` |
| With CUDA | `cargo test --features emeans/cuda` (CUDA tests skip without a device) |
| Bench binaries | `cargo run --release -p emeans-bench --bin <name> -- ...` (`--features baselines` for the candle baselines) |
| faiss reference runs | `uv run --python 3.12 bench/py/ref_emeans.py ...` on data from `bench/py/gen_data.py` (gitignored `bench/data/`) |

Benchmark results here come from synthetic data only, on two machines: an M1 Pro Mac (16 GB) and an
Ubuntu box with an RTX 3060 and 2x Xeon E5-2683 v4.

# Where to read next

- New to the code: [core](design/core.md), then [e-means algorithm](design/emeans-algorithm.md).
- Adding a backend: [adding an e-means backend](recipes/adding-an-emeans-backend.md).
- Why the crates are split this way: [crate layout](decisions/crate-layout.md).
- Running it in a browser or Node.js: [web version](design/web.md).
- What it buys over faiss: [improvements over faiss](results/improvements-over-faiss.md).
