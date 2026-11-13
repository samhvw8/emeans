# emeans

e-means in Rust: mini-batch k-means with scaling laws, ported from faiss
[`contrib/e_means`](https://github.com/facebookresearch/faiss) ([arXiv 2611.09501](https://arxiv.org/abs/2611.09501)).
It runs on the CPU, on Apple GPUs (Metal) and on NVIDIA GPUs (CUDA), and adds what you need to
cluster embeddings in practice: automatic k in two levels, incremental assignment of new rows,
near-duplicate groups and exact kNN.

e-means reaches the loss of Lloyd's k-means in about a tenth of the epochs. It replaces the full
pass per iteration with mini-batches and an exponential moving average per centroid, with batch
size and step size set by scaling laws (see the paper and `NOTICE`).

## Crates

| Crate | What it holds | Heavy dependencies |
|---|---|---|
| [`emeans`](crates/emeans) | The facade: re-exports everything below. Depend on this one. | — |
| [`emeans-core`](crates/emeans-core) | `Emeans` (train, assign, save/load, resume), the `Assigner` trait, CPU kernels, the backend op contract (`core::Device`) with its f64 reference and checks, the device-resident training loop, recipe, numpy-compatible RNG | none (`rayon`, `serde`, `anyhow`; `gemm` off macOS) |
| [`emeans-metal`](crates/emeans-metal) | `MetalCore`: fused top-1 and top-k Metal kernels and the exact device ops (macOS only; empty elsewhere) | `objc2-metal` (only the headers it uses) |
| [`emeans-cuda`](crates/emeans-cuda) | `CudaCore`: NVRTC-compiled fused top-1, top-k and device ops | `cudarc` (loads the driver at run time) |
| [`emeans-cluster`](crates/emeans-cluster) | `Backend`, `auto_cluster` / `Hierarchy`, incremental `add`, `dedup`, `knn`, metrics, synthetic data | none |
| `emeans-bench` (`bench/`, not published) | Benchmarks, probes and the candle baselines (feature `baselines`) | candle, only with `baselines` |

Dependency graph: `emeans-core` ← `emeans-metal`, `emeans-cuda` ← `emeans-cluster` ← `emeans`.
`emeans-cluster` depends on the backend crates so that `Backend::MetalFused` and `Backend::Cuda`
can be built by name; an edit to an algorithm never rebuilds a backend.

## Usage

```toml
[dependencies]
emeans = { git = "https://github.com/samhvw8/emeans" }            # CPU, plus Metal on macOS
# emeans = { git = "...", features = ["cuda"] }                    # plus CUDA
```

Train and assign (rows are row-major `f32`, `n x d`):

```rust
use emeans::{Backend, Config, Emeans, make_assigner};

let mut cfg = Config::new(d, k);
cfg.spherical = true; // unit rows, cosine similarity
let mut model = Emeans::with_assigner(cfg, make_assigner(Backend::cpu())?)?;
let trained = model.train(&x)?; // centroids: k x d
let a = model.assign(&x)?;      // labels and scores per row
model.save("model")?;
```

Two-level clustering with automatic k, then new rows later:

```rust
use emeans::{AutoConfig, Hierarchy, auto_cluster};

let cfg = AutoConfig { backend: "metal".parse()?, ..Default::default() };
let (mut h, labels) = auto_cluster(&x, d, &cfg)?; // labels.l1, labels.l2: stable group ids
h.save("hierarchy")?;
// Later: all rows so far, the first `n_old` of them already seen.
let report = h.add(&x_all, n_old)?;               // assign, update, retrain when needed
```

Near-duplicate groups and exact kNN:

```rust
let groups = emeans::dedup(&x, d, &emeans::DedupInput::default(), &emeans::DedupConfig::default(), Backend::cpu())?;
let nb = emeans::knn_self(&x, d, 10, Backend::cpu())?; // ids and similarities, best first
```

Read `crates/emeans/src/lib.rs` and the module docs for the full API.

## Bindings

The same API from other languages, in `bindings/` (built from source; nothing is on PyPI or npm yet):

| Language | Path | Built with | Data in / out |
|---|---|---|---|
| Python | [`bindings/python`](bindings/python) | PyO3, maturin (abi3, CPython 3.11+) | any float32 buffer (NumPy, `array`, `memoryview`, `bytes`), no copy / read-only buffers, `.numpy()` |
| Node.js | [`bindings/node`](bindings/node) | napi-rs 3, TypeScript types | `Float32Array`, async / typed arrays |
| Go | [`bindings/go`](bindings/go) | cgo over the C ABI in [`crates/emeans-ffi`](crates/emeans-ffi) (cbindgen header) | `[]float32` / slices |
| C / C++ (and any C FFI) | [`crates/emeans-ffi`](crates/emeans-ffi) | `mise run c:dist`: `libemeans.{a,dylib,so}`, `emeans.h`, pkg-config, CMake `find_package(emeans)` | `float*`, opaque handles; ABI version 1 |
| Browser and Node.js (WebAssembly) | [`bindings/wasm`](bindings/wasm) | wasm-bindgen, TypeScript types; one thread, CPU only | `Float32Array` (copied) / typed arrays; `kernel: "fused"` is byte-identical to native `cpu-fused`, the default `gemm` is faster |

Each binding has `train`, `assign`, `auto_cluster`, `dedup` and the incremental `Hierarchy`
(`add`, `save`, `load`), takes configs as overrides of the library defaults (an unknown key is an
error), and is tested to give byte-identical results to this crate (`mise run bind:test`,
`mise run c:test`). A new language wraps the C API: see `.okf/recipes/adding-a-language-binding.md`.

## Backends

| Name (`"…".parse::<Backend>()`) | Where | Notes |
|---|---|---|
| `cpu` | everywhere | the fastest kernel for the target: `cpu-accelerate` on macOS, `cpu-fused` on x86-64, `cpu-blocked` elsewhere |
| `cpu-accelerate` | macOS | Apple Accelerate `cblas_sgemm`, then argmax |
| `cpu-fused` | everywhere (AVX2+FMA microkernel on x86-64) | packed GEMM with the argmax fused per tile |
| `cpu-gemm` | not macOS | the `gemm` crate, called the way candle's CPU matmul calls it |
| `cpu-blocked` | everywhere | cache-blocked register tiles |
| `metal` | macOS | `MetalCore`; training runs the device loop on the GPU |
| `cuda` | feature `cuda` | `CudaCore`; training runs the device loop on the GPU |
| `cuda-bf16` | feature `cuda` | BF16 operands, FP32 accumulate. Not exact (opt-in) |

Your own backend: implement `emeans::Assigner` (and, to get the device-resident training loop and
top-k for free, `emeans::core::Device`, returning `Some(f(self))` from `with_engine`), then pass
`Backend::Custom(CustomBackend { name, make })`.

## Exactness

Every backend implements the op contract in `emeans-core` (`core::Device`): top-1 and top-k match
an f64 reference except near-ties, scatter-add, EMA, norms and splits are bit-equal to the CPU code.
`core::contract::check` tests a backend against it; the test suites run it on every backend the
machine has. Training with the same seed reaches the same objective on every backend (within 1e-4
relative, `crates/emeans/tests/backends.rs`), and the CPU path matches the vendored faiss Python
reference draw for draw on fixtures (`crates/emeans-core/tests/reference.rs`).

## Features

| Crate | Feature | Effect |
|---|---|---|
| `emeans` | `cuda` | builds the CUDA backend (`emeans-cuda`). Building needs no CUDA toolkit; running needs a driver and device |
| `emeans` | — | Metal is part of every macOS build; there is no `metal` feature |
| `emeans-bench` | `baselines` | the candle baselines `cpu-candle-gemm`, `metal-unfused`, `cuda-unfused` |

## Limits

- f32 only; metrics are inner product (cosine for unit rows) and squared L2.
- `knn` and `dedup` are exact and cost `O(n^2 d)`: fine at tens of thousands of rows, not at millions.
- Metal is tested on an M1 Pro, CUDA on one RTX 3060 (sm_86) on Linux. Other GPUs, Windows and
  CUDA versions other than the one on that machine are untested.
- The automatic-k defaults (`AutoConfig`) were tuned for 768-d image embeddings; check them on
  your data.
- Version 0.1: the API may change. No crates.io release yet.

## Development

```sh
mise run check   # fmt, clippy -D warnings (also with the candle baselines), all tests
mise run test    # tests only; GPU tests skip when the machine has no device
```

The Python reference (`bench/py/`, `reference/faiss_e_means/`) regenerates the test fixtures:
`uv run --script bench/py/fixtures.py`. Design notes, recipes and benchmark results (synthetic data
only) are in [`.okf/`](.okf/index.md).

## License

MIT OR Apache-2.0, at your option. The ported algorithm comes from faiss (MIT); see `NOTICE`.
