# emeans

Fast k-means for embeddings, in Rust. emeans is a port of faiss
[`contrib/e_means`](https://github.com/facebookresearch/faiss) (mini-batch k-means with scaling
laws, [arXiv 2611.09501](https://arxiv.org/abs/2611.09501)). It runs on the CPU, on Apple GPUs
(Metal) and on NVIDIA GPUs (CUDA), and you can call it from Rust, Python, Node.js, Go, C and
WebAssembly.

- **Same results as faiss.** emeans replays faiss's random draws. Its objective stays within
  seed noise of faiss: the worst gap of means over 7 parity cases is 0.43%
  ([results](.okf/results/core-engine.md#faiss-parity)).
- **Same results on every backend.** With the same seed, the default CPU kernel, Metal and exact
  CUDA training ended at the same final objective in every benchmark run
  ([Metal](.okf/results/metal-kernels.md#exactness), [CUDA](.okf/results/cuda-kernels.md#exactness)).
- **Faster on the same machine.** 200,000 x 768 rows, k = 1024, exact FP32 on an RTX 3060:
  **0.16 s**, against 0.31 s for faiss Triton FP32
  ([results](.okf/results/cuda-kernels.md)).

It also adds what clustering embeddings needs in practice: automatic k in two levels,
incremental updates with stable cluster ids, near-duplicate groups and exact kNN.

> Version 0.1. Nothing is published yet (crates.io, PyPI, npm): build from source. The API may change.

## Contents

[Quickstart](#quickstart) ·
[Fixed k or automatic k](#fixed-k-or-automatic-k) ·
[Features](#features) ·
[Backends and platforms](#backends-and-platforms) ·
[Benchmarks](#benchmarks) ·
[How it differs from faiss](#how-it-differs-from-faiss) ·
[Limits](#limits) ·
[Docs](#docs) ·
[License](#license-and-credit)

## Quickstart

All APIs take row-major `f32` data, `n x d`. Cosine similarity expects unit rows.
Every binding has a full quickstart program; the `mise run example:*` tasks build the binding and
run it on synthetic data.

### Rust

```toml
[dependencies]
emeans = { git = "https://github.com/samhvw8/emeans" }                     # CPU, plus Metal on macOS
# emeans = { git = "https://github.com/samhvw8/emeans", features = ["cuda"] }  # plus CUDA
```

```rust
use emeans::synth::{NestedSpec, nested};
use emeans::{AutoConfig, Backend, Config, Emeans, auto_cluster, make_assigner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (n, d, k) = (20_000, 64, 32);
    let x = nested(&NestedSpec { n, d, ..Default::default() }).x; // synthetic unit rows

    // Train k centroids with cosine similarity, then label every row.
    let mut cfg = Config::new(d, k);
    cfg.spherical = true;
    let mut model = Emeans::with_assigner(cfg, make_assigner(Backend::cpu())?)?;
    let trained = model.train(&x)?; // trained.centroids: k x d
    let a = model.assign(&x)?;      // a.labels, a.scores: one per row
    model.save("model")?;

    // Automatic k in two levels on the first 16,000 rows, then add the rest.
    let cfg = AutoConfig { backend: Backend::cpu(), ..Default::default() };
    let (mut h, labels) = auto_cluster(&x[..16_000 * d], d, &cfg)?; // labels.l1, labels.l2
    let report = h.add(&x, 16_000)?; // all rows so far; the first 16,000 are already in h
    h.save("hierarchy")?;

    // Near-duplicate groups and exact kNN.
    let groups = emeans::dedup(&x, d, &Default::default(), &Default::default(), Backend::cpu())?;
    let nb = emeans::knn_self(&x, d, 10, Backend::cpu())?;
    Ok(())
}
```

Use `"metal".parse()?` or `"cuda".parse()?` in place of `Backend::cpu()` for a GPU.
Full API: [`crates/emeans/src/lib.rs`](crates/emeans/src/lib.rs) and the module docs.

### Python

```sh
mise run bind:py:wheel                       # wheel in target/py/wheels (abi3, CPython 3.11+)
pip install target/py/wheels/emeans-*.whl
```

```python
import numpy as np
import emeans

x = np.random.default_rng(0).standard_normal((10_000, 64), dtype=np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)           # unit rows for cosine

t = emeans.train(x, 32, spherical=True, seed=7)          # d comes from the 2-D shape
a = emeans.assign(x, t.centroids, metric="inner_product")
labels = a.labels.numpy()                                 # uint32, no copy

h, l = emeans.auto_cluster(x)                             # automatic k, two levels
h.save("model"); h = emeans.Hierarchy.load("model")
report = h.add(np.vstack([x, x[:100]]))                   # all rows seen, then the new ones
groups = emeans.dedup(x, sim_threshold=0.97)              # near-duplicate groups
```

Full example: [`bindings/python/examples/quickstart.py`](bindings/python/examples/quickstart.py)
(`mise run example:python`). Guide: [`bindings/python`](bindings/python/README.md).

### Node.js

```sh
cd bindings/node && npm install && npm run build
cd /your/project && npm install /path/to/emeans/bindings/node
```

```js
// quickstart.mjs
import emeans from 'emeans'

const n = 10_000, d = 64
const x = new Float32Array(n * d).map(() => Math.random() - 0.5)
for (let i = 0; i < n; i++) {                            // unit rows for cosine
  const row = x.subarray(i * d, (i + 1) * d)
  const norm = Math.hypot(...row)
  row.forEach((v, j) => (row[j] = v / norm))
}

const t = await emeans.train(x, d, 32, { spherical: true, seed: 7 })
const a = await emeans.assign(x, t.centroids, d, { metric: 'inner_product' })  // a.labels: Uint32Array
const { hierarchy, labels } = await emeans.autoCluster(x, d)
await hierarchy.save('model')
const groups = await emeans.dedup(x, d, { sim_threshold: 0.97 })
```

Every compute call runs on a worker thread and returns a Promise.
Full example: [`bindings/node/examples/quickstart.mjs`](bindings/node/examples/quickstart.mjs)
(`mise run example:node`). Guide: [`bindings/node`](bindings/node/README.md).

### Go

```sh
mise run bind:go:lib          # builds the static library into bindings/go/lib/<GOOS>_<GOARCH>/
# in your go.mod:
#   replace github.com/samhvw8/emeans/bindings/go => /path/to/emeans/bindings/go
```

```go
import emeans "github.com/samhvw8/emeans/bindings/go"

// x: []float32, n x d unit rows
cfg := emeans.DefaultTrainConfig()
cfg.Seed, cfg.Spherical = 7, true
t, err := emeans.Train(x, d, 32, &cfg)
a, err := emeans.Assign(x, t.Centroids, d, emeans.InnerProduct, "") // "" = cpu

h, labels, err := emeans.AutoCluster(x, d, nil)      // nil = the library defaults
defer h.Close()
err = h.Save("model")
report, err := h.Add(xAll)                           // all rows seen, then the new ones
```

Needs cgo and Go 1.24+. Full example:
[`bindings/go/examples/quickstart`](bindings/go/examples/quickstart/main.go)
(`mise run example:go`). Guide: [`bindings/go`](bindings/go/README.md).

### C and C++

```sh
mise run c:dist     # dist/: emeans.h, libemeans.{a,dylib,so}, pkg-config, CMake package
cc app.c $(PKG_CONFIG_PATH=dist/lib/pkgconfig pkg-config --cflags --libs emeans) \
   -Wl,-rpath,$PWD/dist/lib -o app
```

```c
#include "emeans.h"

EmeansModel *m;
EmeansResult *r;
if (emeans_model_train(x, n, d, 32, "{\"spherical\": true}", &m, &r) != EMEANS_OK) {
  fprintf(stderr, "emeans: %s\n", emeans_last_error());
  return 1;
}
EmeansArray c;
emeans_result_array(r, "centroids", &c);   /* float, k * d, borrowed from r */
emeans_result_free(r);
emeans_model_free(m);
```

CMake: `find_package(emeans 0.1 REQUIRED)` and link `emeans::emeans`.
Full example: [`examples/c/quickstart.c`](examples/c/quickstart.c) (`mise run c:test`).
Guide: [`crates/emeans-ffi`](crates/emeans-ffi/README.md).

### WebAssembly (browser and Node.js)

```sh
rustup target add wasm32-unknown-unknown
cd bindings/wasm && mise run build     # pkg/emeans.js, pkg/emeans.d.ts, pkg/emeans_bg.wasm
```

```js
import init, { train } from "./pkg/emeans.js";
await init();                       // fetches emeans_bg.wasm next to emeans.js
const x = new Float32Array(n * d);  // row-major n x d
const model = train(x, d, 64, { seed: 7, nepochs: 5 });
const { labels, scores } = model.assign(x);   // Uint32Array, Float32Array
```

In Node.js, load the module with `initSync`. Full example:
[`bindings/wasm/examples/quickstart.mjs`](bindings/wasm/examples/quickstart.mjs)
(`mise run example:wasm` in `bindings/wasm`); browser demo: `mise run demo`.
Guide: [`bindings/wasm`](bindings/wasm/README.md).

## Fixed k or automatic k

You choose per call; nothing else changes. Both work on the same data, backends and languages.

| You want | Call | You get |
|---|---|---|
| Exactly k clusters | `train(data, d, k)` | a model with k centroids; `assign` labels new rows |
| k chosen for you | `auto_cluster(data, d)` | a two-level hierarchy (broad groups, then leaves) and its labels |

Use a fixed k when you know how many groups you need (a quantizer, a fixed number of buckets, a
comparison with another k-means). Use automatic k when you don't: it sweeps k, picks it with
silhouette (level 1) and Davies-Bouldin (leaves), splits leaves that are too large, and stays flat
when the data has no broad structure.

Automatic k can be bounded rather than left fully free:

| Option | Default | Effect |
|---|---|---|
| `k1_range` | (2, 40) | range of the level-1 sweep; `(8, 8)` fixes 8 broad groups and still picks the leaves |
| `k2_max` | 20 | largest number of leaves per group |
| `min_cluster_size` / `max_cluster_size` | 20 / 200 | the leaf sizes it prefers; larger leaves are split again |

```python
m = emeans.train(x, 64)                                   # exactly 64 clusters
h, labels = emeans.auto_cluster(x)                        # k chosen for you
h, labels = emeans.auto_cluster(x, k1_range=(5, 12), k2_max=8)  # bounded
```

The option names are the same in Node.js (`{ k1_range: [5, 12], k2_max: 8 }`), Go
(`&emeans.AutoConfig{K1Range: [2]int{5, 12}, K2Max: 8}`) and the C API (JSON); WebAssembly uses
camelCase (`{ k1Range: [5, 12], k2Max: 8 }`).

New data later: a fixed-k model labels new rows with `assign` (or keeps training from its saved
state); an automatic-k hierarchy takes them with `add`, which keeps cluster ids stable and can grow
a new leaf for a new group ([how it works](.okf/design/autok-incremental-dedup.md)).

## Features

| Feature | What it does | Entry point (Rust) |
|---|---|---|
| e-means training | Mini-batch k-means with the faiss recipe: batch and step size from scaling laws, cosine step decay, revival of dead clusters, exact closing means, spherical (cosine) or L2 | `Emeans::train` |
| Assignment | Nearest centroid and score for every row | `Emeans::assign` |
| Resume and persist | Save and load models; continue training a saved model | `Emeans::save`, `load`, `train_resume` |
| Automatic k | Picks k in two levels (broad groups, then leaves); stays flat when the data has no broad structure | `auto_cluster` |
| Incremental updates | New rows go to the nearest leaf and keep stable ids; coherent new groups become new leaves; a short update or a full retrain runs when enough rows arrive, and ids carry over | `Hierarchy::add` |
| Outlier flag | Per leaf, from the median and spread of its similarities, not a fixed percentile | `Labels::outlier` |
| Near-duplicate groups | Exact kNN, then groups around a stored anchor, optionally limited by a time gap | `dedup` |
| Exact kNN | Top-k neighbors on CPU, Metal or CUDA | `knn`, `knn_self` |
| Your own backend | Implement `Assigner` (and `core::Device` for the GPU training loop), then pass `Backend::Custom` | [recipe](.okf/recipes/adding-an-emeans-backend.md) |

Measured on synthetic data ([summary](.okf/results/improvements-over-faiss.md)):

- Incremental: 80% of rows trained, then 4 adds of 5% each. 100% of old rows kept their leaf
  id. Updates took 0.6 s in total, against 38.5 s for full retrains. The objective was 0.37% above
  a full retrain.
- Automatic k on nested synthetic data: level-1 ARI 1.000 and leaf ARI 0.993 to 1.000 on 10 of
  10 sets.
- Near-duplicate groups on data with injected bursts: the groups match the bursts (ARI 1.0).

## Backends and platforms

| Backend | Where | Precision | Notes |
|---|---|---|---|
| `cpu` | everywhere | exact FP32 | Picks the fastest CPU kernel: Apple Accelerate on macOS, `cpu-fused` (AVX2+FMA) on x86-64, `cpu-blocked` elsewhere |
| `cpu-fused`, `cpu-blocked`, `cpu-accelerate`, `cpu-gemm` | see [`backend.rs`](crates/emeans-cluster/src/backend.rs) | exact FP32 | Force one CPU kernel |
| `metal` | macOS, Apple GPUs | exact FP32 | The training loop runs on the GPU. Part of every macOS build |
| `cuda` | Linux, NVIDIA GPUs (feature `cuda`) | exact FP32 | Tensor-core screen plus an FP32 check; labels and scores equal the plain FP32 kernel. Builds without a CUDA toolkit; running needs a driver |
| `cuda-bf16` | same as `cuda` | BF16 operands, FP32 sums | **Opt-in, not exact.** Faster; changed 809 of 200,000 labels at k = 1024 on the bench data ([summary](.okf/results/improvements-over-faiss.md)) |
| WebAssembly | browsers, Node.js | FP32 | One thread, CPU only. Default kernel `gemm` is the fast one; `kernel: "fused"` gives the same bits as native `cpu-fused` but is much slower |

| Platform | Status |
|---|---|
| macOS arm64 | CPU and Metal; tested on an M1 Pro |
| Linux x86-64 | CPU and CUDA; CUDA tested on one RTX 3060 (sm_86) with CUDA 12.4 |
| Windows | not built or tested |

**What "exact" means here.** Every backend implements one op contract (top-1, top-k,
scatter-add, EMA update, split, norms), checked against an f64 reference by
`core::contract::check`. On the benchmark data:

- Metal against the CPU: identical final objectives in 45 of 45 runs, byte-identical centroid files
  ([results](.okf/results/metal-kernels.md#exactness)).
- CUDA FP32 against `cpu-fused` on the same machine: every epoch objective identical in 9 of 9 runs
  ([results](.okf/results/cpu-kernels.md#exactness)).
- CUDA against the Mac CPU: within 6e-11 relative, from the float sum of the evaluator on two machines
  ([results](.okf/results/cuda-kernels.md#exactness)).

`cpu-blocked` sums in another order, so a near-tie can flip a label: its objectives differed from
the default kernel by up to 4.5e-4 ([results](.okf/results/core-engine.md#mac-m1-pro)). On harder
data, a revival split can send two backends to different optima. The test suite requires the same
objective within 1e-4 relative on every backend (`crates/emeans/tests/backends.rs`).

**Bindings.** The Python, Node.js, Go and C APIs give byte-identical centroids and labels to the
Rust crate: the binding tests check it, and the benchmark driver fails otherwise
([results](.okf/results/bindings-bench.md#byte-equality)).

## Benchmarks

Baseline: faiss `contrib/e_means`, its fastest implementation on each machine (NumPy, torch or
Triton). Synthetic data, d = 768, `train` time in seconds, **lower is better**. Median of 3 to 5
seeds, faiss and emeans run interleaved on the same machine.

| 200,000 rows, k = 1024 | Baseline: faiss | emeans | emeans is faster by | Source |
|---|---|---|---|---|
| M1 Pro, CPU ¹ | 1.80 s (torch) | about 1.0 to 1.2 s (Accelerate) | about 1.5 to 1.8x ¹ | [summary](.okf/results/improvements-over-faiss.md), [CPU](.okf/results/cpu-kernels.md) |
| M1 Pro, GPU | does not run (MPS has no float64) | **0.74 s** (Metal) | runs where faiss cannot | [Metal](.okf/results/metal-kernels.md#results) |
| 2x Xeon E5-2683 v4, CPU | 3.73 s (torch, oneDNN) | **1.83 s** (`cpu-fused`) | **2.0x** | [CPU](.okf/results/cpu-kernels.md#final-sweeps) |
| RTX 3060, exact FP32 | 0.31 s (Triton FP32) | **0.16 s** | **1.9x** | [CUDA](.okf/results/cuda-kernels.md) |
| RTX 3060, BF16 (faiss default, emeans opt-in) | 0.15 s (Triton BF16) | **0.10 s** | **1.5x** | [CUDA](.okf/results/cuda-kernels.md) |

¹ A lighter check on a busy workstation: faiss from one seed, emeans from an interleaved A/B of
old and new builds. Treat it as a range, not a ratio.

| 1,000,000 rows, k = 1024 | Baseline: faiss | emeans | emeans is faster by | Source |
|---|---|---|---|---|
| 2x Xeon E5-2683 v4, CPU | 18.31 s (torch) | **6.92 s** | **2.6x** | [CPU](.okf/results/cpu-kernels.md#final-sweeps) |
| RTX 3060, exact FP32 | 1.38 s (Triton FP32) | **0.62 s** | **2.2x** | [CUDA](.okf/results/cuda-kernels.md) |
| RTX 3060, BF16 | 0.60 s (Triton BF16) | **0.39 s** | **1.5x** | [CUDA](.okf/results/cuda-kernels.md) |

What made it faster than the baseline: kernels that find the nearest centroid inside the matrix
multiply instead of writing the full score matrix, a training loop that keeps data and centroids on
the GPU, and on CUDA an FP16 tensor-core screen with an exact FP32 re-check (same labels as plain
FP32). Quality does not change: the objective stays within 0.43% of faiss
([how it differs](#how-it-differs-from-faiss)).

Through the bindings (200,000 x 768, Linux box): `train` from Python, Node.js and Go took -9% to
+6% of the time of native Rust ([results](.okf/results/bindings-bench.md)).
WebAssembly with the default `gemm` kernel took 2.8 to 3.4 times as long as native `cpu-fused` on
one thread of the Xeon ([results](.okf/results/wasm-bench.md#reading)).

All results: [`.okf/results/`](.okf/results/index.md).

## How it differs from faiss

| | faiss `contrib/e_means` | emeans |
|---|---|---|
| Language | Python (NumPy, torch) | Rust, with Python, Node.js, Go, C and WebAssembly bindings |
| Recipe and random draws | the original | the same recipe; replays NumPy's MT19937 stream draw for draw |
| Apple GPU | fails on MPS (float64 statistics) | Metal |
| CUDA precision by default | BF16 | exact FP32; BF16 is opt-in |
| Same seed on different hardware | not a stated goal | same objective on CPU, Metal and CUDA FP32 (see [exactness](#backends-and-platforms)) |
| Choosing k | you give k | `auto_cluster` finds it, in two levels |
| New rows after training | not included | `Hierarchy::add` with stable ids |
| Near-duplicates | not included | `dedup` |
| Quantized (int8, fp8), IVF, sharded and distributed variants | included | **not ported** |

## Limits

- **No published packages.** Build from source. The Go module does not ship the library or the
  header, so `go get` alone does not work yet.
- **CUDA through the bindings is untested.** Python, Node.js, Go and C build with
  `--features cuda`, but their tests ran on CPU and Metal only.
- **Windows is not supported.** Tested platforms: macOS arm64 (M1 Pro) and Linux x86-64 (one
  RTX 3060). Other GPUs and CUDA versions are untested.
- f32 only. Metrics: inner product (cosine for unit rows) and squared L2.
- `knn` and `dedup` are exact and cost `O(n^2 d)`: fine at tens of thousands of rows, not at
  millions.
- Automatic k, incremental updates and the grow step were validated on synthetic data only. Their
  thresholds are defaults; check them on your data.
- WebAssembly runs one thread on the CPU, and wasm32 memory stops at 4 GiB.

## Docs

| To | Read |
|---|---|
| Understand the crates and where to start | [`.okf/getting-started.md`](.okf/getting-started.md) |
| See how the port follows faiss, and the parity table | [`.okf/design/emeans-algorithm.md`](.okf/design/emeans-algorithm.md) |
| Learn the op contract and the device loop | [`.okf/design/core.md`](.okf/design/core.md) |
| Learn how auto-k, incremental updates and dedup work | [`.okf/design/autok-incremental-dedup.md`](.okf/design/autok-incremental-dedup.md) |
| Use a language binding | [Python](bindings/python/README.md), [Node.js](bindings/node/README.md), [Go](bindings/go/README.md), [C](crates/emeans-ffi/README.md), [WebAssembly](bindings/wasm/README.md) |
| Add a backend or a language | [backend recipe](.okf/recipes/adding-an-emeans-backend.md), [binding recipe](.okf/recipes/adding-a-language-binding.md) |
| Check a number | [`.okf/results/`](.okf/results/index.md) |

### Crates

| Crate | Holds |
|---|---|
| [`emeans`](crates/emeans) | The facade. Depend on this one |
| [`emeans-core`](crates/emeans-core) | `Emeans`, the `Assigner` trait, CPU kernels, the op contract and its checks, the device training loop. No GPU crates |
| [`emeans-metal`](crates/emeans-metal) | The Metal engine (macOS only) |
| [`emeans-cuda`](crates/emeans-cuda) | The CUDA engine (`cudarc`, loads the driver at run time) |
| [`emeans-cluster`](crates/emeans-cluster) | `Backend`, `auto_cluster`, `Hierarchy`, `dedup`, `knn`, metrics, synthetic data |
| [`emeans-ffi`](crates/emeans-ffi) | The C ABI (not published) |

## Development

```sh
mise run check       # fmt, clippy -D warnings, all tests (GPU tests skip without a device)
mise run bind:test   # Python, Node.js and Go bindings, byte for byte against Rust
mise run c:test      # the C library and examples
```

The faiss Python reference (`reference/faiss_e_means/`, `bench/py/`) generates the test fixtures:
`uv run --script bench/py/fixtures.py`.

## License and credit

MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).

The e-means algorithm, its recipe and the fused top-1 idea come from faiss `contrib/e_means`
(MIT) by Swann Bessa, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Hugo Touvron, David Picard,
Matthijs Douze and Hervé Jégou. See [NOTICE](NOTICE) and the paper,
[arXiv 2611.09501](https://arxiv.org/abs/2611.09501).
