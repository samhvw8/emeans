---
type: Design
title: Language bindings — Python, Node.js and Go over one plain API
description: How the Python (PyO3), Node.js (napi-rs) and Go (cgo over a C ABI) bindings share one Rust layer, pass configs as JSON overrides, read inputs without a copy, and prove they equal the Rust facade byte for byte.
resource: bindings/
tags: [bindings, python, node, go, ffi, c-abi, api]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T08:00:00Z'
sources:
  - id: common
    resource: bindings/common/src/lib.rs
    title: emeans-bindings-common — the plain API every binding wraps
  - id: fixture
    resource: bindings/common/src/fixture.rs
    title: The cross-language fixture writer (calls the facade directly)
  - id: ffi
    resource: crates/emeans-ffi/src/lib.rs
    title: emeans-ffi — the C ABI and its cbindgen header
---

# Layout

| Path | What | Build |
|---|---|---|
| `bindings/common` (`emeans-bindings-common`, not published) | The plain API: `train`, `assign`, `auto_cluster`, `dedup`, `hierarchy_add/save/load/info`, `default_config`, `available_backends`; plain result structs; the fixture writer | root workspace member |
| `crates/emeans-ffi` (not published) | C ABI over the common layer: status codes, thread-local last error, opaque `EmeansResult` / `EmeansHierarchy`, JSON configs. Header `include/emeans.h` made by cbindgen | root workspace member; `staticlib`, `cdylib`, `rlib` |
| `bindings/python` | PyO3 0.30 extension `emeans._emeans` + package `emeans` (stubs, `py.typed`), maturin 1.16, abi3 for CPython 3.11+ | own Cargo workspace and lock |
| `bindings/node` | napi-rs 3 addon (`napi` 3.16, `@napi-rs/cli` 3.10), async tasks, generated `index.d.ts` with a hand-written config header | own Cargo workspace and lock |
| `bindings/go` | cgo package `emeans` over `emeans-ffi` (static library in `lib/<GOOS>_<GOARCH>/`, built by `mise run bind:go:lib`) | Go 1.24+ module |

The Python and Node crates are excluded from the root workspace, so `cargo test --workspace`
never needs a Python or Node toolchain, and their dependencies stay out of the root lock file.[^common]

# One API, one set of rules

Every binding calls the common layer, so these rules live in one place:[^common]

- **Configs are JSON objects of overrides.** A call starts from the library defaults
  (`default_config("train" | "auto" | "dedup")`, which serializes `Config`, `AutoConfig` and
  `DedupConfig` with serde) and replaces only the given keys, recursing into nested objects
  (`incremental`). An unknown key is an error that names it and lists the known keys. So no
  binding repeats the field list to apply defaults: Python takes `**kwargs`, Node a plain object,
  Go marshals its struct.
- **`backend` is one more config key** (`assign` and `Hierarchy.load` take it as an argument).
  It is parsed by `Backend::from_str`; a backend this build or machine lacks fails at the call.
- **Shapes:** `n` comes from the data length; `d` is an argument (Python also reads it from a
  2-D buffer). `train` refuses `d` or `k` inside the config.
- **`train` does the work of native `Emeans::train`, no more.** Its `objective` is free: the last
  epoch's moving objective as a mean (that epoch's `obj` over the rows it processed), a little
  above the exact value. The exact objective over the final centroids costs one more assign pass,
  so it is opt-in: the config key `exact_objective: true` (all languages) fills
  `exact_objective`, and the C model handle has `emeans_model_objective`. Until 2026-11-13 the
  layer always ran that pass, which made binding `train` slower than native and warmed the next
  `assign` (found by the benchmark).
- **The incremental model** is `Hierarchy`: `add(data_all)` passes `n_old = n_seen` itself, so
  callers cannot pass a wrong split; rows are append-only.

Typed config declarations exist only where a language needs them for types: the Python stub
`TypedDict`s, the Node `types-header.d.ts`, and the Go structs. The Go structs are checked in
both directions by tests (library defaults decode with unknown fields refused; the Go defaults
go back without an unknown-key error). The Python stubs (module members, result classes, config
`TypedDict`s) and the Node config interfaces are compared with the runtime and with
`default_config` by tests, so a new library key fails a test until the declaration has it.

# Zero copy

| | In | Out |
|---|---|---|
| Python | buffer protocol (`PyUntypedBuffer`): native float32 (`'f'`, NumPy, `array.array`, `memoryview`) or raw bytes (`bytes`, `bytearray`), C-contiguous and 4-byte aligned; read in place with the GIL released (`Python::detach`) | `emeans.Array`: a frozen pyclass that owns the `Vec` and exports `__getbuffer__` (read-only, 1-D or 2-D); `Array.numpy()` is `numpy.asarray` on it |
| Node | `Float32Array` moved into an `AsyncTask` and read on a libuv worker | `Float32Array::new(vec)` etc.: the Rust buffer becomes the typed array's memory |
| Go | `unsafe.SliceData` of the slice passed to C | copied once into Go slices, then the C result is freed |

The input rule in every language: do not change the data until the call returns. Side arrays of
`dedup` (`created`, `burst`, `priority`) are small: Python copies them, Node copies only
`burst` (Int32 to Int64), Go passes them in place.

# C ABI

Status codes `EMEANS_OK`, `EMEANS_INVALID_ARGUMENT` (NULL, overflow, bad UTF-8 or JSON),
`EMEANS_ERROR` (the library refused or failed), `EMEANS_PANIC` (every entry point runs under
`catch_unwind`). The message is thread-local (`emeans_last_error`); the Go wrapper locks the
goroutine to its OS thread for each call so it reads its own message. Results are one opaque
type with named arrays (`emeans_result_array(r, "centroids", &arr)`, with a dtype tag) plus a
JSON object of scalars, so the ABI has one result mechanism for every call.[^ffi] The Go
`Hierarchy` frees its handle in `Close()`, with `runtime.AddCleanup` as the safety net.

# Proof: byte for byte against the facade

`fixture::write_fixture` makes synthetic data (`synth::nested`, 1000 x 16, plus 40 bursts of
near-copies: 1143 rows) and calls the facade directly with explicit configs: `train` (k 8,
spherical, seed 7), `assign`, `auto_cluster` on the first 1000 rows, `Hierarchy::add` of the
rest, and `dedup` with times and priorities. It writes the raw outputs and a `fixture.json` that
holds the same configs as JSON overrides.[^fixture]

Each language runs the same calls through its binding and compares the bytes: centroids, labels,
scores, the per-epoch objectives, hierarchy labels (`l1`, `l2`, `sim1`, `sim`, `outlier`) after
`auto_cluster`, after `save` + `load` + `add`, and after `add`, and the dedup arrays. The common
layer's own Rust test runs the same check without a language in between. The fixture is written
at test time on the same machine, because the CPU kernel differs by target (Accelerate on macOS,
the fused kernel on x86-64), so a committed fixture would not match other machines.

| Check (macOS arm64, M1 Pro, 2026-11-13) | Result |
|---|---|
| Rust: common layer = facade (`bindings/common/tests/fixture.rs`), C ABI unit tests | pass |
| Python 3.15, PyO3 0.30.1, maturin 1.16.0: 12 tests, incl. Metal (`EMEANS_TEST_METAL=1`) and stub drift | pass |
| Node 26.13, napi 3.16.2, @napi-rs/cli 3.10.12: 8 tests, incl. Metal and the config-interface check | pass |
| Go 1.27.3, cgo, cbindgen 0.29.4: 7 tests, incl. Metal and the config structs in both directions | pass |
| TypeScript 5.9 `tsc --strict` on a usage file against the generated `index.d.ts` (one-off) | pass |
| Linux x86-64 (Ubuntu box, CPU, 2026-11-13): Rust tests + `mise run bind:test` (Python 12, Node 7, Go 7; Metal tests skip) | pass |

Metal runs are compared on the objective (1e-4 relative), as in the facade's backend test: Metal
training is not bit-identical to the CPU.

# Commands

| Task | Command |
|---|---|
| All bindings | `mise run bind:test` |
| One binding | `mise run bind:py:test`, `bind:node:test`, `bind:go:test` |
| Regenerate the C header after an ABI change | `mise run bind:go:header` (`bind:go:test` fails on a stale header: `cbindgen --verify`) |
| Metal checks | set `EMEANS_TEST_METAL=1` (each test file has one) |
| Quickstart per language (synthetic data: train, assign, auto-cluster, save/load, add, timings) | `mise run example:python`, `example:node`, `example:go` (append a backend, e.g. `metal`) |
| Benchmark against native Rust | `mise run bind:bench -- --data FILE --d D --ks 64,256,1024 --backends cpu --out DIR` (builds into `target/bench`; results in [bindings benchmark](/results/bindings-bench.md)) |

The Go link line for macOS is the `--print native-static-libs` list of the static library
(Metal, CoreFoundation, Foundation, Accelerate, objc). The `linux_amd64` line (`-lm -ldl
-lpthread`) links and passes the tests on the box; `linux_arm64` is unchecked.

# Shipped binaries carry no home paths

Panic locations and debug info record source paths, so a binary built in a home folder carries
`/Users/<name>/...` or `/home/<name>/...`. Before the fix (Mac, 2026-11-13): `libemeans.a` 89,
`libemeans.dylib` 38, the Python `.so` 96, the Node `.node` 79, the Go-linked `libemeans_ffi.a`
89 such paths. The builds that ship (`bind:py:test`, `bind:py:wheel`, `bind:node:test`,
`bind:go:lib`, `bind:bench:build`, `example:*`, `c:dist`) now set
`RUSTFLAGS=--remap-path-prefix=$HOME=~` (mapping `~/.cargo/registry/...` and the repo path) and
build release without debug info (`CARGO_PROFILE_RELEASE_DEBUG=0`, `STRIP=debuginfo`). The macOS
linker's debug map (paths of the toolchain's `.rlib` files) cannot be remapped, only stripped.
The Node addon also gets an `@rpath` install name (`build.rs`), because macOS writes the output's
absolute path there. `bindings/check-home-paths.sh` greps each artifact for `/Users/` and `/home/`
and fails on a hit; the tasks run it after every build: 0 hits on every artifact after the fix.
Plain `cargo build` and `cargo test` keep debug info and real paths.

# Benchmark harness

`bindings/bench/run.py` runs four programs in fresh processes under `/usr/bin/time` (peak RSS):
`bindings/common/examples/bench.rs` (the facade, native) and `examples/bench.*` of each binding.
Each loads the data into its own language first (one buffer: Vec, NumPy, Float32Array,
[]float32), then times `train` (spherical, seed 1234, 5 epochs) and `assign` of all rows. Every
k and backend runs `--reps` times (default 3) in a rotated language order, and the table shows
medians, so the first-run warm-up does not land on one language. A separate run per language
(mode `call`, `RAYON_NUM_THREADS=1`) times 2000 one-row assign calls against 64 centroids: the
per-call cost of crossing into Rust, without thread wake-ups. Every run prints FNV-1a 64 hashes of
the centroids and labels; the driver fails unless all of them equal Rust's. `--lock` holds a
flock for shared machines.

# Not done yet

- Prebuilt packages: PyPI wheels, npm platform packages, a Go module that ships the library and
  header (`go get` alone does not work). Publishing needs the user's go-ahead.
- Linux x86-64: the bindings are written for it (Go link line, napi target) but were not built
  or tested there.
- CUDA: every binding has a `cuda` feature that forwards to `emeans/cuda`; built from source
  only, not tested through the bindings.
- The plan also lists `encoder`'s `embed_images` / `embed_texts` for the bindings; they belong
  to the `encoder` repo and are not part of these bindings.
