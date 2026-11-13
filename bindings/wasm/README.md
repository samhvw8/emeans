# emeans for the web

The emeans library compiled to WebAssembly, with a JavaScript/TypeScript API. One package
runs in the browser and in Node.js, on the CPU, in one thread. The default kernel (`gemm`)
is the fast one. `kernel: "fused"` is bit-identical to native `cpu-fused` on every machine.
The Node tests check both on synthetic data.

## Build

Tools come from `mise.toml` in this folder: the `wasm-bindgen` CLI (0.2.130, equal to the
crate version), `wasm-opt` (binaryen) and Node. Add the Rust target once:

```bash
rustup target add wasm32-unknown-unknown
cd bindings/wasm
mise run build          # pkg/emeans.js, pkg/emeans.d.ts, pkg/emeans_bg.wasm, size report
mise run test           # native unit tests, native golden outputs, Node tests
mise run example:wasm   # the Node quickstart (examples/quickstart.mjs)
mise run demo           # the browser demo at http://localhost:8765/demo/
mise run demo:check     # the demo in headless Chrome: timings, out/demo.png
mise run bench          # wasm vs native benchmark (scripts/bench-run.mjs --help)
```

`pkg/` is not checked in; `mise run build` makes it. This folder is its own Cargo
workspace, so `cargo test --workspace` at the repo root does not build it.

## Examples

- **Node.js:** `mise run example:wasm` runs `examples/quickstart.mjs`. It makes synthetic
  unit vectors, then trains, assigns, saves and loads the model, runs `autoCluster`, adds
  1,000 new rows to the hierarchy, saves and loads it, and prints labels and timings.
- **Browser:** `mise run demo`, then open `http://localhost:8765/demo/`. The page
  (`demo/index.html`, no bundler) clusters synthetic blobs, plots the first two dimensions
  and shows timings; pick the kernel to compare `gemm` and `fused`.
  `demo/bench.html?n=16000&d=768&k=256` runs one benchmark training.

## Use

`pkg/emeans.js` is an ES module (wasm-bindgen `--target web`).

Browser:

```js
import init, { train } from "./pkg/emeans.js";
await init();                       // fetches emeans_bg.wasm next to emeans.js
const x = new Float32Array(n * d);  // row-major n x d
const model = train(x, d, 64, { seed: 7, nepochs: 5 });
const { labels, scores } = model.assign(x);   // Uint32Array, Float32Array
```

Node.js (tested on 26; needs ES modules and `initSync`):

```js
import { readFileSync } from "node:fs";
import { initSync, train } from "./pkg/emeans.js";
initSync({ module: readFileSync(new URL("./pkg/emeans_bg.wasm", import.meta.url)) });
```

## API

Inputs are `Float32Array`s of `n x d` values, row-major; they are copied into wasm memory.
Results are new typed arrays and plain objects. Options are plain objects: keys are the
fields of the Rust config in camelCase (snake_case also works). A missing key keeps the
Rust default. An unknown key, a wrong type or a ragged input throws an `Error`.
`pkg/emeans.d.ts` has every type.

| Function or method | Returns | Rust |
|---|---|---|
| `train(x, d, k, options?: TrainOptions)` | `Model` | `Emeans::train` |
| `model.assign(x)` | `{ labels: Uint32Array, scores: Float32Array }` | `Emeans::assign` |
| `model.centroids()` | `Float32Array` (`k x d`) | |
| `model.trainLabels()` | `Int32Array` (last-pass labels, -1 = not sampled) or `undefined` | `Trained::labels` |
| `model.emaState()`, `model.stats()` | EMA positions and masses; per-epoch statistics | |
| `model.toBytes()`, `Model.fromBytes(bytes)` | `Uint8Array`; `Model` | the native model file format |
| `autoCluster(x, d, options?: AutoOptions)` | `{ hierarchy: Hierarchy, labels: Labels }` | `auto_cluster` |
| `hierarchy.assign(x)` | `Labels` | `Hierarchy::assign` |
| `hierarchy.add(xAll, nOld)` | `AddReport` (`action`, `labels` of all rows, ...) | `Hierarchy::add` |
| `hierarchy.toBytes()`, `Hierarchy.fromBytes(bytes)` | one byte blob; `Hierarchy` | `Hierarchy::to_bytes` / `from_bytes` |
| `hierarchy.chosenK()`, `diagnostics()`, `level1Centroids()`, `leafCentroids(g)`, `nSeen` | | |
| `dedup(x, d, options?: DedupOptions, inputs?: DedupInputs)` | `DedupResult` (`groupOf`, `representatives`, `nnSim`, ...) | `dedup` |
| `blobs(n, d, centers, seed)` | `Float32Array`: synthetic rows, the same bits on every platform | `api::blobs` |
| `version()` | crate version | |

`autoCluster`, `dedup` and spherical `train` expect unit rows (L2-normalized).

`Model` and `Hierarchy` live in wasm memory. Call `free()` when you are done with one, or
leave it to the garbage collector. A Rust panic prints its message to the console and
throws; the module is then unusable, so load it again.

The size of `pkg/emeans_bg.wasm` (`wasm-opt -O3`, simd128): 783 KiB raw, 295 KiB gzip,
230 KiB brotli.

## Kernels and exactness

`TrainOptions`, `AutoOptions`, `DedupOptions` and `fromBytes` take `kernel`:

| wasm `kernel` | Speed on wasm | Same bits as | Checked by |
|---|---|---|---|
| `gemm` (default; `CpuKernel::fastest()` on wasm) | Fast: the `gemm` crate's simd128 kernels | Every other wasm run (any engine, any machine). **Not** native `cpu-gemm`: the `gemm` crate picks its microkernels per instruction set, and the small differences change labels during training | `tests/golden.test.mjs` against x86-64 Linux; centroid hashes in Node and Chrome on macOS and Linux |
| `fused` | 12x slower than `gemm` at d = 32, 21-26x at d = 768 (wasm has no FMA instruction, so `mul_add` is a software call) | Native `cpu-fused` on every machine: the portable `mul_add` kernel on aarch64 macOS and the AVX2+FMA kernel on x86-64 Linux | `tests/golden.test.mjs` on both hosts; benchmark centroid hashes |
| `blocked` | Middle | Native `cpu-blocked` on x86-64 (the same portable multiply-add); not on aarch64 (NEON FMA tiles) | `tests/golden.test.mjs` on x86-64 Linux |

With `fused`, `train`, `assign`, `toBytes`/`fromBytes`, `autoCluster` and `hierarchy.add`
give the same bits as native in the tests. `autoCluster` also scores candidate k with
`gemm_nt` (Accelerate on macOS, the `gemm` crate elsewhere); the scores chose the same k in
the tests, but a near tie could choose differently. `dedup` gives the same groups, and its
`nnSim` similarities (from `gemm_nt`) are within 4 ulps (3 seen).

Benchmark (d = 768, k = 256 and 1024, n up to 200,000; Node and headless Chrome against
native Rust): `.okf/results/wasm-bench.md`. Design and the research on threads and WebGPU:
`.okf/design/web.md`.

## Threads and the GPU

- **Threads:** not used. Wasm threads need nightly Rust with `-Z build-std` and a page
  served with COOP/COEP headers (`SharedArrayBuffer`). See `.okf/design/web.md`.
- **WebGPU:** not built. See the research in `.okf/design/web.md`.
