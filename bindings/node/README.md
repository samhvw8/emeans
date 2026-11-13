# emeans for Node.js

Node.js binding of [emeans](https://github.com/samhvw8/emeans): e-means clustering (mini-batch
k-means with scaling laws) on the CPU, Apple GPUs (Metal) and NVIDIA GPUs (CUDA), with automatic
k, incremental updates and near-duplicate groups. Built with napi-rs 3. TypeScript types included.

## Install

No release is on npm yet (the package is marked `private`). Build from the repository (needs Rust
and Node; `mise install` in the repo root gives both):

```sh
cd bindings/node
npm install                 # the napi CLI only
npm run build               # emeans.<platform>.node, index.js, index.d.ts
# To ship the .node, build it with `mise run bind:node:test` from the repo root: it
# remaps $HOME out of the binary, strips debug info and checks for home paths.
# with CUDA (Linux, NVIDIA driver at run time):  npm run build -- --features cuda
```

Then `require('<path>/bindings/node')`, or `npm install <path>/bindings/node` in your project.

## Example

```js
const emeans = require('emeans')

const d = 64
const x = new Float32Array(10_000 * d).map(() => Math.random() - 0.5)

const t = await emeans.train(x, d, 32, { seed: 7 })            // Float32Array in, out
const a = await emeans.assign(x, t.centroids, d)               // a.labels: Uint32Array
const { hierarchy, labels } = await emeans.autoCluster(x, d, { backend: 'metal' })
await hierarchy.save('model')
const h = await emeans.Hierarchy.load('model')
const report = await h.add(xAll)                                // all rows seen, then new ones
const groups = await emeans.dedup(x, d, { sim_threshold: 0.97 })
```

## Quickstart

`examples/quickstart.mjs` makes 20,000 synthetic unit rows, then trains, assigns, runs `autoCluster` on 16,000 rows, saves
and loads the hierarchy, adds the other 4,000 rows, and prints labels and timings. Run it from
the repo root with one command (it builds the binding first):

```sh
mise run example:node           # CPU
mise run example:node metal     # another backend
```

## API

| Call | Resolves to |
|---|---|
| `train(data, d, k, config?)` | `{ centroids, k, d, objective, exactObjective?, stats, labels?, backend }`; `objective` is the last epoch's moving mean (free), `exactObjective` only with `exact_objective: true` (one more pass) |
| `assign(data, centroids, d, { metric?, backend? })` | `{ labels: Uint32Array, scores: Float32Array }` |
| `autoCluster(data, d, config?)` | `{ hierarchy: Hierarchy, labels: { l1, l2, sim1, sim, outlier } }` |
| `dedup(data, d, config?, { created?, burst?, priority? })` | `{ groupOf, representatives, anchors, sizes, nnSim, nGroups, linkedPairs, splitComponents }` |
| `hierarchy.add(dataAll)` | `{ action, labels, newOutliers, pending, outlierFraction, newLeaves }` |
| `hierarchy.save(path)`, `Hierarchy.load(path, backend?)` | a directory, replaced as a whole |
| `hierarchy.d`, `.nSeen`, `.chosenK` | getters (they throw while an `add` runs) |
| `defaultConfig(kind)`, `backends()` | sync |

- **Async:** every compute call runs on a libuv worker and returns a Promise, so the event loop
  stays free. Calls on one `Hierarchy` run one at a time.
- **Data in:** `Float32Array` (row-major `n x d`), read in place without a copy. Do not change it
  until the promise settles.
- **Data out:** typed arrays that take over the Rust buffers without a copy.
- **Config:** a plain object of overrides of `defaultConfig(kind)`, with the library's snake_case
  keys (`sim_threshold`, `k1_range`, ...). An unknown key rejects the promise and names the key.
  Result fields are camelCase.
- **Errors:** a rejected promise with an `Error` whose message says why (`code` `InvalidArg`).

## Backends

`backend` takes `cpu` (the fastest CPU kernel: Accelerate on macOS), `cpu-fused`, `cpu-blocked`,
`cpu-accelerate` (macOS), `cpu-gemm` (not macOS), `metal` (macOS), and `cuda` / `cuda-bf16` (a
build with `--features cuda`). `backends()` lists the ones that work here. Training gives the same
objective on every backend within 1e-4; the CPU result is bit-identical to the Rust library.

## Limits

- float32 only; metrics are squared L2 and inner product (cosine for unit rows).
- `dedup` is exact, `O(n^2 d)`: fine for tens of thousands of rows, not millions.
- Burst ids are `Int32Array` (ids below 2^31).
- Version 0.1: the API may change. Built and tested with Node 26 on macOS arm64 (CPU and Metal)
  and Linux x86-64 (Ubuntu, CPU). No prebuilt packages yet; CUDA is from source only and not
  tested through the binding.

## Benchmark

`examples/bench.mjs` is this binding's side of the benchmark against native Rust
(`bindings/bench/run.py`, `mise run bind:bench -- --data FILE --d D --out DIR`). Results:
`.okf/results/bindings-bench.md` in the repo.

## Tests

`mise run bind:node:test` in the repo root writes the Rust fixture, builds the addon and runs
`node --test test/`. The tests check that every call equals the Rust library's results byte for
byte, that inputs are read in place, and that misuse rejects with a clear message.
