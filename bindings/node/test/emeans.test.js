// Node binding tests: results equal the Rust facade byte for byte.
//
// Needs the fixture written by the Rust side (`mise run bind:node:test` does both):
//   cargo run --release -p emeans-bindings-common --example write_fixture -- DIR
//   EMEANS_FIXTURE=DIR node --test test/

'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { test, describe } = require('node:test')

const emeans = require('..')

const FIXTURE = process.env.EMEANS_FIXTURE || ''
const hasFixture = FIXTURE !== '' && fs.existsSync(path.join(FIXTURE, 'fixture.json'))

const read = (name) => fs.readFileSync(path.join(FIXTURE, name))
/** A typed array over a fresh, aligned copy of a fixture file. */
const typed = (Type, name) => {
  const b = read(name)
  const out = new Type(b.byteLength / Type.BYTES_PER_ELEMENT)
  new Uint8Array(out.buffer).set(b)
  return out
}
const bytesOf = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength)

function same(name, arr) {
  assert.ok(bytesOf(arr).equals(read(name)), `${name} differs from Rust`)
}

function sameLabels(prefix, l) {
  same(`${prefix}.l1.u32`, l.l1)
  same(`${prefix}.l2.u32`, l.l2)
  same(`${prefix}.sim1.f32`, l.sim1)
  same(`${prefix}.sim.f32`, l.sim)
  same(`${prefix}.outlier.u8`, l.outlier)
}

describe('same as Rust', { skip: !hasFixture && 'set EMEANS_FIXTURE' }, () => {
  const meta = hasFixture ? JSON.parse(read('fixture.json')) : {}
  const x = hasFixture ? typed(Float32Array, 'x.f32') : null
  const d = meta.d

  test('train and assign', async () => {
    const t = await emeans.train(x, d, meta.k, meta.train.config)
    same('train.centroids.f32', t.centroids)
    same('train.labels.i32', t.labels)
    assert.equal(t.objective, meta.train.objective)
    assert.equal(t.exactObjective, undefined)
    const te = await emeans.train(x, d, meta.k, { ...meta.train.config, exact_objective: true })
    same('train.centroids.f32', te.centroids)
    assert.equal(te.exactObjective, meta.train.exact_objective)
    assert.deepEqual(
      t.stats.map((s) => s.obj),
      meta.train.obj,
    )
    assert.ok(t.centroids instanceof Float32Array)
    const a = await emeans.assign(x, t.centroids, d, { metric: meta.assign.metric })
    same('assign.labels.u32', a.labels)
    same('assign.scores.f32', a.scores)
  })

  test('autoCluster, save, load, add', async () => {
    const n0 = meta.n0
    const { hierarchy: h, labels } = await emeans.autoCluster(x.subarray(0, n0 * d), d, meta.auto.config)
    sameLabels('auto', labels)
    assert.deepEqual(h.chosenK, { k1: meta.auto.k1, k2: meta.auto.k2 })
    assert.equal(h.nSeen, n0)
    assert.equal(h.d, d)
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'emeans-node-'))
    try {
      await h.save(path.join(dir, 'model'))
      const h2 = await emeans.Hierarchy.load(path.join(dir, 'model'), 'cpu')
      sameLabels('add', (await h2.add(x)).labels)
    } finally {
      fs.rmSync(dir, { recursive: true, force: true })
    }
    const r = await h.add(x)
    sameLabels('add', r.labels)
    assert.equal(r.action, meta.add.action)
    assert.equal(h.nSeen, meta.n)
    await assert.rejects(h.add(x), /rows the model has seen/)
  })

  test('dedup', async () => {
    const g = await emeans.dedup(x, d, meta.dedup.config, {
      created: typed(Float64Array, 'created.f64'),
      priority: typed(Float32Array, 'priority.f32'),
    })
    same('dedup.group_of.u32', g.groupOf)
    same('dedup.representatives.u32', g.representatives)
    same('dedup.sizes.u32', g.sizes)
    same('dedup.nn_sim.f32', g.nnSim)
    assert.equal(g.nGroups, meta.dedup.n_groups)
    assert.equal(g.linkedPairs, meta.dedup.linked_pairs)
  })
})

describe('misuse', () => {
  const x = new Float32Array(40)
  test('errors reject with the reason', async () => {
    await assert.rejects(emeans.train(x, 4, 2, { sead: 1 }), /"sead"/)
    await assert.rejects(emeans.train(x, 3, 2), /multiple of d/)
    await assert.rejects(emeans.train(x, 4, 2, { backend: 'tpu' }), /unknown backend/)
    await assert.rejects(emeans.assign(x, x.subarray(0, 8), 4, { metric: 'manhattan' }), /metric/)
    assert.throws(() => emeans.defaultConfig('nope'), /unknown config kind/)
    assert.throws(() => emeans.train(new Float64Array(40), 4, 2), /Float32Array|TypedArray|expect/i)
  })
})

describe('zero copy', () => {
  test('inputs are read in place', () => {
    const buf = new Float32Array(64)
    const view = new Float32Array(buf.buffer, 16, 8) // 16 bytes in
    // A copy would land at an unrelated address; a view is 16 bytes past its buffer.
    assert.equal(emeans._bufferAddress(view) - emeans._bufferAddress(buf), 16)
  })
})

describe('info', () => {
  test('defaults and backends', () => {
    const c = emeans.defaultConfig('auto')
    assert.equal(c.backend, 'cpu')
    assert.deepEqual(c.k1_range, [2, 40])
    assert.ok(!('d' in emeans.defaultConfig('train')))
    assert.equal(emeans.backends()[0], 'cpu')
  })
})

describe('types', () => {
  test('config interfaces list the library keys', () => {
    const src = fs.readFileSync(path.join(__dirname, '..', 'types-header.d.ts'), 'utf8')
    const keys = (name) => {
      const body = src.match(new RegExp(`export interface ${name} \\{([^}]*)\\}`))[1]
      return [...body.matchAll(/^\s+(\w+)\?:/gm)].map((m) => m[1]).sort()
    }
    assert.deepEqual(keys('TrainConfig'), Object.keys(emeans.defaultConfig('train')).sort())
    assert.deepEqual(keys('AutoConfig'), Object.keys(emeans.defaultConfig('auto')).sort())
    assert.deepEqual(keys('DedupConfig'), Object.keys(emeans.defaultConfig('dedup')).sort())
    assert.deepEqual(keys('IncrementalConfig'), Object.keys(emeans.defaultConfig('auto').incremental).sort())
  })
})

describe('metal', { skip: process.env.EMEANS_TEST_METAL !== '1' && 'set EMEANS_TEST_METAL=1 (holds the GPU)' }, () => {
  test('trains to the cpu objective', async () => {
    const meta = JSON.parse(read('fixture.json'))
    const x = typed(Float32Array, 'x.f32')
    const t = await emeans.train(x, meta.d, meta.k, { ...meta.train.config, backend: 'metal' })
    assert.equal(t.backend, 'metal-fused')
    assert.ok(Math.abs(t.objective - meta.train.objective) / meta.train.objective < 1e-4)
  })
})
