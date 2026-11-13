# emeans for Go

Go binding of [emeans](https://github.com/samhvw8/emeans): e-means clustering (mini-batch k-means
with scaling laws) on the CPU, Apple GPUs (Metal) and NVIDIA GPUs (CUDA), with automatic k,
incremental updates and near-duplicate groups. It is a cgo package over the emeans C ABI
(`crates/emeans-ffi`, header `crates/emeans-ffi/include/emeans.h`, made by cbindgen).

## Install

No prebuilt library is published yet. Build the static library in the repository, then use the
package from there (needs Rust and Go; `mise install` in the repo root gives both):

```sh
mise run bind:go:lib     # cargo build -p emeans-ffi → bindings/go/lib/<GOOS>_<GOARCH>/libemeans_ffi.a
                         # (home paths remapped out, then checked)
# in your go.mod:  replace github.com/samhvw8/emeans/bindings/go => /path/to/emeans/bindings/go
```

CUDA (Linux): build the library with `cargo build --release -p emeans-ffi --features cuda`, copy
it to `lib/linux_amd64/`, and link the CUDA driver at run time.

## Example

```go
import emeans "github.com/samhvw8/emeans/bindings/go"

cfg := emeans.DefaultTrainConfig()
cfg.Seed, cfg.Spherical = 7, true
t, err := emeans.Train(x, d, 32, &cfg)               // x: []float32, n x d
a, err := emeans.Assign(x, t.Centroids, d, emeans.InnerProduct, "")

h, labels, err := emeans.AutoCluster(x, d, nil)      // nil = the library defaults
defer h.Close()
err = h.Save("model")
report, err := h.Add(xAll)                           // all rows seen, then the new ones
g, err := emeans.Dedup(x, d, nil, emeans.DedupSide{Created: times})
```

## Quickstart

`examples/quickstart/main.go` makes 20,000 synthetic unit rows, then trains, assigns, runs `AutoCluster` on 16,000 rows, saves
and loads the hierarchy, adds the other 4,000 rows, and prints labels and timings. Run it from
the repo root with one command (it builds the binding first):

```sh
mise run example:go           # CPU
mise run example:go metal     # another backend
```

## API

| Call | Returns |
|---|---|
| `Train(data, d, k, *TrainConfig)` | `*TrainResult{Centroids, K, D, Objective, ExactObjective, Stats, Labels, Backend}`; `Objective` is the last epoch's moving mean (free), `ExactObjective` only with `TrainConfig.ExactObjective` (one more pass) |
| `Assign(data, centroids, d, Metric, backend)` | `*Assignment{Labels []uint32, Scores []float32}` |
| `AutoCluster(data, d, *AutoConfig)` | `*Hierarchy`, `*Labels{L1, L2, Sim1, Sim, Outlier}` |
| `Dedup(data, d, *DedupConfig, DedupSide)` | `*DedupResult{GroupOf, Representatives, Anchors, Sizes, NNSim, NGroups, ...}` |
| `(*Hierarchy).Add(dataAll)` | `*AddReport{Action, Labels, NewOutliers, Pending, OutlierFraction, NewLeaves}` |
| `(*Hierarchy).Save(path)`, `LoadHierarchy(path, backend)` | a directory, replaced as a whole |
| `(*Hierarchy).Info()`, `(*Hierarchy).Close()` | `HierarchyInfo{D, NSeen, K1, K2}`; frees the model |
| `DefaultTrainConfig()`, `DefaultAutoConfig()`, `DefaultDedupConfig()` | the library's own defaults |
| `Backends()`, `Version()` | |

- **Data in:** `[]float32` row-major, passed to the library in place (no copy). Do not change it
  from another goroutine until the call returns.
- **Data out:** Go slices, copied once from the library's result, which is then freed.
- **Config:** get the defaults from `Default*Config()`, change fields, pass a pointer; `nil` means
  the defaults. A zero-value config struct is not the defaults (it is refused or means zeros).
- **Errors:** every error is an `*emeans.Error{Code, Msg}`; test with `errors.Is(err,
  emeans.ErrInvalidArgument | ErrFailed | ErrPanic | ErrClosed)`.
- **Memory:** `Close()` frees a `Hierarchy`; a `runtime.AddCleanup` frees it if you forget.
  A `Hierarchy` is safe for concurrent use; its calls run one at a time.
- Each call locks its goroutine to one OS thread while it runs, so the C ABI's thread-local
  error message belongs to that call.

## Backends

`backend` takes `cpu` (the fastest CPU kernel: Accelerate on macOS), `cpu-fused`, `cpu-blocked`,
`cpu-accelerate` (macOS), `cpu-gemm` (not macOS), `metal` (macOS), and `cuda` / `cuda-bf16` (a
library built with `--features cuda`). `Backends()` lists the ones that work here. Training gives
the same objective on every backend within 1e-4; the CPU result is bit-identical to the Rust library.

## Limits

- float32 only; metrics are squared L2 and inner product (cosine for unit rows).
- `Dedup` is exact, `O(n^2 d)`: fine for tens of thousands of rows, not millions.
- Needs cgo and Go 1.24+. Version 0.1: the API may change.
- Built and tested with Go 1.27 on macOS arm64 (CPU and Metal) and Linux x86-64 (Ubuntu, CPU).
  The `linux_arm64` link line is untested; CUDA is from source only. `go get` alone does not
  work yet: the module does not ship the library or the header.

## Benchmark

`examples/bench/main.go` is this binding's side of the benchmark against native Rust
(`bindings/bench/run.py`, `mise run bind:bench -- --data FILE --d D --out DIR`). Results:
`.okf/results/bindings-bench.md` in the repo.

## Tests

`mise run bind:go:test` in the repo root writes the Rust fixture, builds the library and runs
`go test`. The tests check that every call equals the Rust library's results byte for byte, that
the Go config structs match the library's config keys in both directions, that inputs are passed
in place, and that misuse returns the right sentinel errors.
