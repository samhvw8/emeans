// Wasm side of the benchmark in Node.js: one training run, one JSON line.
// node scripts/bench-node.mjs <n> <d> <k> <kernel> <seed>
// Prints train time, the SHA-256 of the centroids, the wasm linear memory at the
// end (it only grows, so this is its peak) and the process's peak RSS.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { initSync, blobs, train } from "../pkg/emeans.js";

const [n, d, k, kernel, seed] = process.argv.slice(2);
const wasm = initSync({ module: readFileSync(new URL("../pkg/emeans_bg.wasm", import.meta.url)) });
const x = blobs(+n, +d, 512, +seed);
const t = performance.now();
const m = train(x, +d, +k, { kernel, seed: +seed, captureAssignments: false });
const trainMs = performance.now() - t;
const c = m.centroids();
console.log(JSON.stringify({
  n: +n, d: +d, k: +k, kernel: m.kernel, runtime: `node ${process.version}`,
  train_ms: trainMs,
  centroids_sha256: createHash("sha256").update(new Uint8Array(c.buffer, c.byteOffset, c.byteLength)).digest("hex"),
  wasm_memory_bytes: wasm.memory.buffer.byteLength,
  peak_rss_bytes: process.resourceUsage().maxRSS * 1024,
  epochs: m.stats().length,
}));
