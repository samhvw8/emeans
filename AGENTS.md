# emeans

e-means (mini-batch k-means with scaling laws, a port of faiss `contrib/e_means`) as a Rust
library workspace: a candle-free core, one crate per GPU backend, the clustering algorithms, and
a facade. A public repo: keep personal data out of it (see Rules).

## Layout

| Path | What |
|---|---|
| `crates/emeans-core` | `Emeans` model, `Assigner` trait, CPU kernels, `gemm_nt`, the op contract (`core::Device`), f64 `reference`, `contract` checks, the device loop (`core::train`), the object-safe `core::Engine` every `Device` gets, recipe, RNG, io, CPU kNN |
| `crates/emeans-metal` | `MetalCore` (macOS only; empty elsewhere) |
| `crates/emeans-cuda` | `CudaCore` on `cudarc` 0.19.10 (candle 0.11.1's version); builds without a CUDA toolkit |
| `crates/emeans-cluster` | `Backend` / `make_assigner` / `CustomBackend`, `auto`, `incremental`, `dedup`, `knn`, `metrics`, `synth` |
| `crates/emeans` | Facade: re-exports; feature `cuda`; Metal by target |
| `crates/emeans-ffi` | C ABI (not published): status codes, opaque handles, JSON configs; header `include/emeans.h` by cbindgen (`mise run bind:go:header`) |
| `bindings/common` | `emeans-bindings-common` (not published): the plain API every binding wraps, and the cross-language fixture writer |
| `bindings/python`, `bindings/node`, `bindings/go` | PyO3 + maturin, napi-rs, cgo; Python and Node are their own Cargo workspaces. See `.okf/design/bindings.md` |
| `bench/` | `emeans-bench` (not published): benchmark binaries; candle baselines behind `baselines` |
| `bench/py/` | Python reference runner and fixture generators (`uv run --script`) |
| `reference/faiss_e_means/` | Upstream faiss source (MIT, pinned in `UPSTREAM_COMMIT`). Read-only |
| `.okf/` | Design, recipes, decisions and synthetic benchmark results (OKF bundle) |

## Commands

| Task | Command |
|---|---|
| Format, lint (also the baselines), test | `mise run check` |
| Tests only | `mise run test` |
| Binding tests (byte-for-byte against the facade) | `mise run bind:test` (or `bind:py:test`, `bind:node:test`, `bind:go:test`) |
| Tests with CUDA (Linux with a GPU) | `mise run test:cuda` |
| A benchmark | `cargo run --release -p emeans-bench --bin bench_assign -- ...` |
| Regenerate test fixtures | `uv run --script bench/py/fixtures.py`, `uv run --script bench/py/metrics_fixture.py` |

## Rules

- Correctness first: a backend passes `core::contract::check` before it is optimized; the CPU
  path must match the faiss reference fixtures (`crates/emeans-core/tests/reference.rs`).
- `emeans-core` stays free of candle and of GPU crates. A new backend is a new crate that
  implements `core::Device` and `Assigner`; follow `.okf/recipes/adding-an-emeans-backend.md`.
- The facade API is the binding surface (Python, Node, Go in `bindings/`): plain `f32` slices in, plain
  structs out, no new type generics, no closures in it.
- `cargo test --workspace` must never build candle: candle stays behind `emeans-bench/baselines`.
- Crashing beats corrupting: a load or a partial run fails loudly, never returns short data.
- Public repo: no personal data, no absolute home paths, benchmark numbers from synthetic data
  only (`bench/py/gen_data.py`, `synth`).
- Long GPU or benchmark runs on a shared workstation: run them at low priority, in short chunks.

Project knowledge lives in `.okf/`. Update it in the same change as the code it describes, and
validate it (`okf_validate.py .okf --strict`).
