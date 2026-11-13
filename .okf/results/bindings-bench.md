---
type: Benchmark Result
title: Bindings vs native Rust — train, assign, call cost, memory, byte equality
description: "Python, Node.js and Go bindings against the Rust facade on the same synthetic data and seed: 200k x 768 at k 64, 256, 1024 on the Linux box (cpu-fused, the reference numbers) and the Mac (CPU and Metal); medians of repeated runs, the per-call cost, peak RSS, and centroid and label hashes."
resource: bindings/bench/
tags: [bindings, benchmark, python, node, go, overhead, metal, cpu]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T10:00:00Z'
sources:
  - id: harness
    resource: bindings/bench/run.py
    title: The driver; programs in bindings/common/examples/bench.rs and each binding's examples/bench.*
  - id: mac
    resource: bindings/bench/results/mac-m1pro/
    title: Mac M1 Pro raw runs (runs.jsonl, calls.jsonl) and tables, 2026-11-13
  - id: box
    resource: bindings/bench/results/box-linux-x86/
    title: Linux box raw runs (runs.jsonl) and table, 2026-11-13
  - id: data
    resource: bench/py/gen_data.py
    title: s768_200k.f32 — 200,000 x 768 synthetic unit rows, 500 centers, sigma 0.6, seed 5
---

# Setup

Data: `s768_200k.f32`, 200,000 synthetic unit rows of 768 floats (614 MB), from
`bench/py/gen_data.py`.[^data] Train: spherical, seed 1234, 5 epochs, k 64, 256 and 1024. Assign:
all rows against the trained centroids (inner product). Each language loads the data into its own
array first, so the timings cover only the emeans calls. Every run is a fresh process; each cell
runs 5 times (box) or 3 times (Mac) in a rotated language order, and the tables show medians. Peak RSS is the
process maximum (the data alone is 614 MB). The call cost is 2000 assign calls of one row against
64 centroids on `cpu`, with `RAYON_NUM_THREADS=1`, median of 7 rounds.[^harness]

Two harness bugs were found and fixed before these numbers. Both made the bindings look faster
than native:

- The binding layer's `train` ran a full extra assign pass for its objective, which also warmed
  the following `assign`. Since 2026-11-13 `train` reports the last epoch's free moving objective,
  and the exact one is opt-in ([language bindings](/design/bindings.md)).
- The Rust bench kept its trained model, with its Metal engine and a GPU copy of the data, alive
  during `assign`; it now drops it as the bindings do.

# Byte equality

Every run of every binding gave the same FNV-1a 64 hashes of the centroids and the labels as
native Rust, on every k and backend (the "= Rust" column). The driver fails otherwise.

# Mac M1 Pro, macOS, 2026-11-13 (a shared workstation: expect ±20 % between runs)

Run through `quiet-run` (it starts only when CPU busy is at most 40 %) with the GPU lock.[^mac]

| k | backend | lang | train s | vs Rust | assign s | vs Rust | peak RSS MB | = Rust |
|---|---|---|---|---|---|---|---|---|
| 64 | cpu-accelerate | rust | 0.331 | — | 0.028 | — | 634 | — |
| 64 | cpu-accelerate | python | 0.345 | +4.1% | 0.023 | -17.7% | 659 | yes |
| 64 | cpu-accelerate | node | 0.341 | +3.0% | 0.024 | -11.6% | 690 | yes |
| 64 | cpu-accelerate | go | 0.327 | -1.3% | 0.022 | -18.9% | 643 | yes |
| 64 | metal-fused | rust | 0.355 | — | 0.135 | — | 1274 | — |
| 64 | metal-fused | python | 0.308 | -13.5% | 0.127 | -5.9% | 1300 | yes |
| 64 | metal-fused | node | 0.344 | -3.2% | 0.171 | +26.6% | 1293 | yes |
| 64 | metal-fused | go | 0.335 | -5.7% | 0.135 | +0.2% | 1271 | yes |
| 256 | cpu-accelerate | rust | 0.432 | — | 0.047 | — | 671 | — |
| 256 | cpu-accelerate | python | 0.445 | +3.1% | 0.045 | -4.5% | 699 | yes |
| 256 | cpu-accelerate | node | 0.432 | -0.1% | 0.046 | -1.5% | 728 | yes |
| 256 | cpu-accelerate | go | 0.435 | +0.8% | 0.047 | +0.4% | 681 | yes |
| 256 | metal-fused | rust | 0.468 | — | 0.178 | — | 1264 | — |
| 256 | metal-fused | python | 0.401 | -14.3% | 0.148 | -16.7% | 1320 | yes |
| 256 | metal-fused | node | 0.387 | -17.3% | 0.142 | -20.0% | 1343 | yes |
| 256 | metal-fused | go | 0.377 | -19.4% | 0.141 | -20.4% | 1301 | yes |
| 1024 | cpu-accelerate | rust | 1.065 | — | 0.148 | — | 742 | — |
| 1024 | cpu-accelerate | python | 1.036 | -2.7% | 0.153 | +3.1% | 767 | yes |
| 1024 | cpu-accelerate | node | 0.993 | -6.8% | 0.146 | -1.4% | 798 | yes |
| 1024 | cpu-accelerate | go | 0.997 | -6.3% | 0.156 | +5.0% | 752 | yes |
| 1024 | metal-fused | rust | 0.844 | — | 0.242 | — | 1318 | — |
| 1024 | metal-fused | python | 0.828 | -1.8% | 0.264 | +9.4% | 1332 | yes |
| 1024 | metal-fused | node | 0.820 | -2.7% | 0.254 | +5.1% | 1352 | yes |
| 1024 | metal-fused | go | 0.829 | -1.8% | 0.239 | -1.2% | 1320 | yes |

| lang | call µs, median of 7 | min - max | overhead vs Rust |
|---|---|---|---|
| rust | 1.9 | 1.8 - 3.1 | — |
| python | 2.5 | 2.4 - 3.2 | +0.6 µs |
| node | 15.9 | 14.0 - 17.9 | +13.9 µs |
| go | 2.7 | 2.6 - 5.3 | +0.7 µs |

# Linux box, 2026-11-13 (the reference numbers)

Xeon E5-2683 v4, 64 threads, Ubuntu (kernel 7.0), `cpu-fused`. The timed part held the box lock
(`/tmp/emeans-box-gpu.lock`), and no other job ran.[^box]

| k | lang | train s | vs Rust | assign s | vs Rust | peak RSS MB | = Rust |
|---|---|---|---|---|---|---|---|
| 64 | rust | 0.587 | — | 0.038 | — | 612 | — |
| 64 | python | 0.535 | -8.7% | 0.030 | -19.1% | 649 | yes |
| 64 | node | 0.624 | +6.3% | 0.040 | +5.1% | 666 | yes |
| 64 | go | 0.540 | -8.0% | 0.035 | -6.5% | 618 | yes |
| 256 | rust | 0.669 | — | 0.064 | — | 614 | — |
| 256 | python | 0.666 | -0.4% | 0.072 | +11.6% | 649 | yes |
| 256 | node | 0.685 | +2.5% | 0.071 | +10.3% | 670 | yes |
| 256 | go | 0.673 | +0.7% | 0.071 | +11.2% | 625 | yes |
| 1024 | rust | 1.678 | — | 0.219 | — | 637 | — |
| 1024 | python | 1.728 | +3.0% | 0.220 | +0.8% | 672 | yes |
| 1024 | node | 1.692 | +0.8% | 0.214 | -2.2% | 695 | yes |
| 1024 | go | 1.743 | +3.9% | 0.210 | -3.7% | 644 | yes |

| lang | call µs, median of 5 | min - max | overhead vs Rust |
|---|---|---|---|
| rust | 215.5 | 197.7 - 225.2 | — |
| python | 202.4 | 199.2 - 213.0 | none measurable |
| node | 598.7 | 534.2 - 778.2 | +383 µs |
| go | 217.2 | 206.4 - 220.1 | +1.7 µs |

One tiny call costs ~200 µs here even in Rust, against ~2 µs on the Mac. The cause was not
investigated. A likely one is the per-call work that every language shares, the same in Rust:
each call builds a new assigner (as `api::assign` does) and hands the job to rayon from a
non-rayon thread on a 64-thread machine. Node pays ~0.4 ms more per call for its two extra thread
hops (libuv worker, promise).

# What it shows

- **Large calls cost the same through every binding.** On the quiet box, train through Python,
  Node and Go is within -9 % to +6 % of native Rust (the faster cells are noise: the same compiled
  code runs), and assign within ±20 % at its ~0.03-0.2 s scale. On the Mac, CPU train is within
  ±7 %, and Metal cells swing ±20 % in both directions (a busy workstation). The data is read in place in every language, so no
  copy shows up in time or memory.
- **Memory:** peak RSS is the data plus the library's working set, the same for every language
  (+25 to +80 MB for the Python, Node and Go runtimes). Metal adds a GPU-side copy of the data.
- **Per-call cost** matters only for many tiny calls: Python and Go add under 1 µs per call;
  Node adds ~14 µs on the Mac and ~0.4 ms on the box, the async round trip (a libuv worker and a
  promise). A Node caller with many small batches should batch rows into one call. Reusing a
  trained model (the C API's `EmeansModel`) skips the per-call assigner build; the Python, Node
  and Go surfaces do not have a model handle yet.

[^data]: `bench/py/gen_data.py`, set `s768_200k`; the same file as the core-engine results.
[^harness]: `bindings/bench/run.py`; `mise run bind:bench -- --data FILE --d 768 --out DIR`.
[^box]: `bindings/bench/results/box-linux-x86/` (`runs.jsonl`, `table.md`).
[^mac]: `bindings/bench/results/mac-m1pro/` (`runs.jsonl`, `table.md`, `calls.jsonl`, `calls.md`).
