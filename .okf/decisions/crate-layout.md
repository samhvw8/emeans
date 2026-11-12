---
type: Decision
title: Crate layout — a candle-free core, one crate per backend, an algorithms crate and a facade
description: Why emeans is one Cargo workspace of five library crates plus an unpublished bench crate, how backends plug in (the Engine trait, Backend enum, Backend::Custom), the dependency trade-off of emeans-cluster, and the binding-friendly facade API.
resource: Cargo.toml
tags: [architecture, crates, build-time, reuse, decision, bindings]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: workspace
    resource: Cargo.toml
    title: Workspace manifest (resolver 3, members, workspace dependencies) and each crate's Cargo.toml
  - id: backend
    resource: crates/emeans-cluster/src/backend.rs
    title: Backend, CustomBackend, make_assigner
---

# Decision

One repository, one Cargo workspace (resolver 3, edition 2024, license `MIT OR Apache-2.0`; the
faiss-derived parts keep faiss's MIT notice). Crates that change together share one version and one
place for issues.[^workspace]

| Crate | Holds | Dependencies |
|---|---|---|
| `emeans-core` | `assign` (`Assigner`, `CpuAssigner`, `CpuKernel` {`Blocked`, `Gemm`, `Accelerate`, `Fused`}, `gemm_nt`), `core` (`Device` op contract, CPU backend, f64 `reference`, `contract` checks, `train` device loop, `Engine` trait), `model` (`Emeans`), `recipe`, `rng`, `io`, `knn` (CPU top-k, `Neighbors`) | `anyhow`, `rayon`, `serde`, `serde_json`; `gemm` only off macOS (Accelerate needs no crate). No candle |
| `emeans-metal` | `MetalCore` (`Device` + `Assigner`), `MetalKnn`, `KNN_SOURCE`, `FUSED_TILES` | `objc2`, `objc2-foundation`, `objc2-metal` with only the features used. Empty crate off macOS |
| `emeans-cuda` | `CudaCore` (`Device` + `Assigner`; `new`, `new_bf16`), NVRTC kernels in `cuda_src.rs` | `cudarc` 0.19.10 (the version candle 0.11.1 pins) |
| `emeans-cluster` | `auto` (`auto_cluster`, `Hierarchy`), `incremental`, `dedup`, `knn` (backend dispatch), `metrics`, `synth`, `backend` | `emeans-core`; `emeans-metal` on macOS (by target); `emeans-cuda` with feature `cuda` |
| `emeans` | Facade: re-exports the public API of all of the above | feature `cuda`; Metal by target |
| `emeans-bench` (`bench/`, `publish = false`) | Benchmark binaries (`bench_assign`, `ops_bench`, `knn_bench`, `top1_bench`, `cpu_probe`, `parity`, `train_profile`, `autok_synth`); candle baselines behind feature `baselines` | `candle-core` 0.11.1 only with `baselines` |

`bench/py/` holds the Python reference scripts (faiss runs, fixtures, synthetic data generator);
`reference/faiss_e_means/` the upstream faiss `contrib/e_means` source (MIT, pinned commit), read-only.

# Why

- **Build cost and reuse.** The engine used to pull about 163-180 dependencies, nearly all through
  candle, which only three side paths (the unfused baselines) used. A CPU-only consumer of
  `emeans-core` needs about 20. The candle baselines (`cpu-candle-gemm`, `metal-unfused`,
  `cuda-unfused`) moved to `emeans-bench` behind `baselines`, exposed there as `Backend::Custom`, so
  `cargo test --workspace` never builds candle.
- **Isolated rebuilds.** Editing an algorithm in `emeans-cluster` never recompiles a backend; editing
  a backend never recompiles the core.
- **CUDA without a toolkit.** `emeans-cuda` uses `cudarc` directly with dynamic loading and
  `fallback-latest`, so it builds on machines without `nvcc` or a driver; `CudaCore::new()` returns
  an error when no driver or device is present. In a binary that also enables candle's `cuda`
  feature, Cargo unifies `cudarc` to candle's `dynamic-linking`, which wins, and one `cudarc` is built.
- **One GEMM.** Off macOS `gemm_nt` uses the `gemm` crate with the version and features candle-core
  0.11.1 uses, called the same way, so a binary with candle compiles one copy and gets the same numerics.

# How backends plug in

Two levels, so neither the core nor the facade needs to know every backend:

1. **The core is open.** `core::Engine` is an object-safe trait (`engine_name`, `run_train(Job) ->
   Outcome`, `run_topk`) with a blanket `impl<D: Device> Engine for D`; `Assigner::with_engine`
   hands `&mut dyn Engine` to a callback. Any crate that implements `Device` gets the device loop and
   top-k ([core](/design/core.md)). This replaced a closed `Engine` enum and its `with_device!` macro.
2. **Naming a backend** is `emeans-cluster`'s job: `Backend` {`Cpu(CpuKernel)`, `MetalFused`, `Cuda`,
   `CudaBf16`, `Custom(CustomBackend { name, make: fn() -> Result<Box<dyn Assigner>> })`},
   `make_assigner`, and `FromStr` names (`cpu`, `cpu-blocked`, `cpu-gemm`, `cpu-accelerate`,
   `cpu-fused`, `metal`/`metal-fused`, `cuda`/`cuda-fused`, `cuda-bf16`). A backend this build or
   machine lacks is an error at `make_assigner`, not at compile time, so code that names
   `Backend::Cuda` builds everywhere. The candle baseline names give an error that points at
   `emeans-bench`.

**Trade-off recorded.** `emeans-cluster` depends on the backend crates (Metal by target, CUDA by
feature) so that `Backend::MetalFused` and `Backend::Cuda` can be built by name. The alternative, a
runtime registry the backend crates fill in, would let `emeans-cluster` depend on the core alone,
but it adds global state and registration order, and a missing registration becomes a run-time
surprise. Cost of the choice: editing a backend rebuilds `emeans-cluster` (and the facade); editing
`emeans-cluster` never rebuilds a backend. An outside backend does not need a variant: it uses
`Backend::Custom`.

# Facade API for language bindings (planned)

Bindings are planned for Python (PyO3 + maturin; inputs through the buffer protocol, `PyBuffer<f32>`,
so NumPy is not a dependency), Node.js (napi-rs, `Float32Array`) and Go (a C ABI with a cbindgen
header, wrapped by cgo). The facade is shaped for that: plain row-major `f32` slices in, plain
structs and vectors out, no generics in the public API, over a small surface: `train`, `assign`,
`auto_cluster`, `dedup`, and the incremental model (`add`, `save`, `load`). Prebuilt packages first
for macOS arm64 (CPU + Metal) and Linux x86-64 (CPU); CUDA builds from source. Binding crate
versions are to be checked when that work starts.

# Checks

`mise run check` runs `cargo fmt --check`, `cargo clippy -D warnings` and the tests; `mise run test`
the tests alone. Backend tests skip with a message when the device is missing (Metal off macOS,
CUDA without a driver). The full contract suite runs on every backend before a backend change lands.

# Measured (2026-11-12, M1 Pro, Rust 1.99.1)

Clean debug builds with plain cargo (no shared compile cache), each into a fresh target dir, two
rounds interleaved with the pre-split crate. The Mac was shared, so the ranges are wide; compare
within a row.

| Build | Unique dependencies | Clean debug build |
|---|---|---|
| Before the split: `emeans` with Metal (candle inside) | 151 | 28.6-58.4 s |
| After: the `emeans` facade (Metal on macOS) | 29 | 6.0-8.6 s |
| Before: `emeans` without features (candle CPU) | 139 | 20.7-22.7 s |
| After: `emeans-core` | 19 | 4.8-5.1 s |
| After: `emeans-core` on x86_64 Linux (with `gemm`) | 48 | not measured |
| After: `emeans` with `cuda` on x86_64 Linux | 53 | not measured |

Dependencies counted with `cargo tree -e normal,build`, unique packages. Tests after the split: all
pass on macOS (Metal contract at the production shape included); the CUDA tests were not run on a
GPU at the time of the split, only compiled and linted for Linux with `cuda`.

[^workspace]: Root `Cargo.toml` and the five crate manifests, 2026-11-12.
