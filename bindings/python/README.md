# emeans for Python

Python binding of [emeans](https://github.com/samhvw8/emeans): e-means clustering (mini-batch
k-means with scaling laws) on the CPU, Apple GPUs (Metal) and NVIDIA GPUs (CUDA), with automatic
k, incremental updates and near-duplicate groups. Built with PyO3 and maturin.

## Install

No release is on PyPI yet. Build from the repository (needs Rust and maturin; `mise install` in
the repo root gives both):

```sh
mise run bind:py:wheel              # wheel in target/py/wheels (abi3: CPython 3.11+)
pip install target/py/wheels/emeans-*.whl
# The task remaps $HOME out of the binary and strips debug info, then checks
# that no home path is left. A plain `maturin build --release` in bindings/python
# works too, but its .so records your home folder.
# Into the active virtualenv:  maturin develop --release
# with CUDA (Linux, NVIDIA driver at run time):  maturin build --release --features cuda
```

NumPy is optional. Install `emeans[numpy]` or NumPy yourself to use `Array.numpy()`.

## Example

```python
import numpy as np
import emeans

x = np.random.default_rng(0).standard_normal((10_000, 64), dtype=np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)           # unit rows for cosine

t = emeans.train(x, 32, spherical=True, seed=7)         # d comes from the 2-D shape
a = emeans.assign(x, t.centroids, metric="inner_product")
labels = a.labels.numpy()                                # uint32, no copy

h, l = emeans.auto_cluster(x, backend="metal")           # automatic k, two levels
h.save("model"); h = emeans.Hierarchy.load("model")
report = h.add(np.vstack([x, x[:100]]))                  # all rows seen, then the new ones
groups = emeans.dedup(x, sim_threshold=0.97)             # near-duplicate groups
```

## Quickstart

`examples/quickstart.py` makes 20,000 synthetic unit rows, then trains, assigns, runs `auto_cluster` on 16,000 rows, saves
and loads the hierarchy, adds the other 4,000 rows, and prints labels and timings. Run it from
the repo root with one command (it builds the binding first):

```sh
mise run example:python           # CPU
mise run example:python metal     # another backend
```

## API

| Call | Returns |
|---|---|
| `train(data, k, *, d=None, **config)` | `TrainResult`: `centroids` (k x d), `objective` (free: the last epoch's moving mean), `exact_objective` (only with `exact_objective=True`: one more pass), `stats` (per epoch), `labels`, `backend` |
| `assign(data, centroids, *, d=None, metric="l2", backend="cpu")` | `Assignment`: `labels` (uint32), `scores` (float32) |
| `auto_cluster(data, *, d=None, **config)` | `(Hierarchy, Labels)`; `Labels` has `l1`, `l2`, `sim1`, `sim`, `outlier` |
| `dedup(data, *, d=None, created=None, burst=None, priority=None, **config)` | `DedupResult`: `group_of`, `representatives`, `anchors`, `sizes`, `nn_sim`, `n_groups`, ... |
| `Hierarchy.add(data_all)` | `AddReport`: `action`, `labels` (all rows), `new_outliers`, `pending`, `new_leaves`, ... |
| `Hierarchy.save(path)`, `Hierarchy.load(path, backend="cpu")` | a directory, replaced as a whole |
| `default_config(kind)` | the defaults of `"train"`, `"auto"` or `"dedup"`; every key is a keyword argument |
| `backends()` | the backends this build can use on this machine |

- **Data in:** any C-contiguous float32 buffer: a NumPy array, `array.array('f')`, a
  `memoryview`, or `bytes` / `bytearray` read as native float32. It is read in place, without a
  copy. Do not change it from another thread during a call. A 2-D buffer gives `d`; pass `d` for
  a 1-D one.
- **Data out:** `emeans.Array`, a read-only buffer. `memoryview(a)`, `np.asarray(a)` and
  `a.numpy()` read it without a copy; `a.tolist()` copies.
- **Config:** keyword arguments override the defaults of `default_config(kind)`. An unknown key
  raises `EmeansError` (a `ValueError`) that names it. Nested `incremental={...}` merges the same way.
- **Threads:** every call releases the GIL while it computes.
- Type stubs ship in the package (`py.typed`).

## Backends

`backend=` takes `cpu` (the fastest CPU kernel: Accelerate on macOS), `cpu-fused`, `cpu-blocked`,
`cpu-accelerate` (macOS), `cpu-gemm` (not macOS), `metal` (macOS), and `cuda` / `cuda-bf16` (a
build with `--features cuda`). `backends()` lists the ones that work here. Training gives the same
objective on every backend within 1e-4; the CPU result is bit-identical to the Rust library.

## Limits

- float32 only; metrics are squared L2 and inner product (cosine for unit rows).
- `dedup` is exact, `O(n^2 d)`: fine for tens of thousands of rows, not millions.
- Version 0.1: the API may change. Built and tested with CPython 3.15 on macOS arm64 (CPU and
  Metal) and Linux x86-64 (Ubuntu, CPU); the abi3 wheel targets CPython 3.11+. No prebuilt
  wheels yet; CUDA is from source only and not tested through the binding.

## Benchmark

`examples/bench.py` is this binding's side of the benchmark against native Rust
(`bindings/bench/run.py`, `mise run bind:bench -- --data FILE --d D --out DIR`). Results:
`.okf/results/bindings-bench.md` in the repo.

## Tests

`mise run bind:py:test` in the repo root writes the Rust fixture, builds the module into
`bindings/python/.venv`, and runs `tests/test_emeans.py`. The tests check that `train`, `assign`,
`auto_cluster`, `Hierarchy.save/load/add` and `dedup` equal the Rust library's results byte for
byte, that inputs are read without a copy, and that misuse raises clear errors.
