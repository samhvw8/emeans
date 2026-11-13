// Node side of the binding benchmark (bindings/bench/run.py).
//
// Usage: node bench.mjs <data.f32> <d> <k> <backend> <mode>; prints one JSON line.
// Mode `full` times train and assign; `call` only the per-call probe.
// The data is read into a Float32Array first: timings cover only the emeans calls.

import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'

const emeans = createRequire(import.meta.url)('..')
const [path, dArg, kArg, backend, mode] = process.argv.slice(2)
const [d, k] = [Number(dArg), Number(kArg)]

/** FNV-1a 64 of the bytes (the same hash in every language bench). */
function fnv(a) {
  const b = new Uint8Array(a.buffer, a.byteOffset, a.byteLength)
  let h = 0xcbf29ce484222325n
  const p = 0x100000001b3n
  const m = 0xffffffffffffffffn
  for (let i = 0; i < b.length; i++) h = ((h ^ BigInt(b[i])) * p) & m
  return h.toString(16).padStart(16, '0')
}

const buf = readFileSync(path)
const x = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
const n = x.length / d

if (mode === 'call') {
  const row = x.subarray(0, d)
  const cents = x.subarray(64 * d, 128 * d)
  const reps = 2000
  const t = performance.now()
  for (let i = 0; i < reps; i++) await emeans.assign(row, cents, d, { metric: 'inner_product' })
  console.log(JSON.stringify({ lang: 'node', call_us: ((performance.now() - t) * 1000) / reps }))
  process.exit(0)
}

let t = performance.now()
const tr = await emeans.train(x, d, k, { spherical: true, backend })
const trainS = (performance.now() - t) / 1000

t = performance.now()
const a = await emeans.assign(x, tr.centroids, d, { metric: 'inner_product', backend })
const assignS = (performance.now() - t) / 1000
let sum = 0
for (const s of a.scores) sum += 2 - 2 * s
const objective = sum / n

console.log(
  JSON.stringify({
    lang: 'node', backend: tr.backend, n, d, k, train_s: trainS, assign_s: assignS, objective,
    centroids_fnv: fnv(tr.centroids), labels_fnv: fnv(a.labels),
  }),
)
