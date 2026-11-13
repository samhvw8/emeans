// emeans quickstart: train, assign, autoCluster, add new rows, save and load.
//
// Run: mise run example:node   (or: node examples/quickstart.mjs [backend])
// The data is synthetic (16 blobs of unit rows).

import { createRequire } from 'node:module'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const emeans = createRequire(import.meta.url)('..')
const backend = process.argv[2] ?? 'cpu'
const [n, d, blobs] = [20_000, 64, 16]

// A small seeded generator (mulberry32) and Box-Muller normals.
let s = 7
const rand = () => {
  s = (s + 0x6d2b79f5) | 0
  let t = Math.imul(s ^ (s >>> 15), 1 | s)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand())

const centers = Array.from({ length: blobs }, () => Array.from({ length: d }, gauss))
const x = new Float32Array(n * d)
for (let i = 0; i < n; i++) {
  const c = centers[Math.floor(rand() * blobs)]
  const row = c.map((v) => v + 0.35 * gauss())
  const norm = Math.hypot(...row)
  row.forEach((v, j) => (x[i * d + j] = v / norm))
}

async function timed(label, f) {
  const t = performance.now()
  const r = await f()
  console.log(`${label.padEnd(28)} ${(performance.now() - t).toFixed(1).padStart(8)} ms`)
  return r
}

console.log(`emeans: ${n} x ${d} rows, backend ${backend} (available: ${emeans.backends().join(', ')})`)
const t = await timed('train k=32', () => emeans.train(x, d, 32, { spherical: true, seed: 7, backend }))
console.log(`  objective ${t.objective.toFixed(4)} after ${t.stats.length} epochs on ${t.backend}`)
const a = await timed('assign', () => emeans.assign(x, t.centroids, d, { metric: 'inner_product', backend }))
console.log(`  first labels ${[...a.labels.subarray(0, 10)]}`)

const nOld = 16_000
let { hierarchy: h, labels } = await timed('autoCluster (16k rows)', () =>
  emeans.autoCluster(x.subarray(0, nOld * d), d, { seed: 7, k1_range: [2, 24], restarts: 2 }),
)
const { k1, k2 } = h.chosenK
console.log(`  level 1: ${k1} groups, ${k2.reduce((p, q) => p + q, 0)} leaves; first l1 labels ${[...labels.l1.subarray(0, 10)]}`)

const dir = mkdtempSync(join(tmpdir(), 'emeans-'))
await timed('save', () => h.save(join(dir, 'model')))
h = await timed('load', () => emeans.Hierarchy.load(join(dir, 'model')))
rmSync(dir, { recursive: true })

const r = await timed('add 4k new rows', () => h.add(x)) // all rows seen so far, then the new ones
console.log(`  action ${r.action}, ${r.newOutliers} new outliers, model has seen ${h.nSeen} rows`)
console.log(`  labels of the new rows: ${[...r.labels.l1.subarray(nOld, nOld + 10)]} ...`)
