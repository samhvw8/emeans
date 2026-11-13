// The JavaScript API: shapes, types, loud errors, size budget.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { load } from "./load.mjs";

const em = load();

/** n x d rows around `centers` random points (deterministic LCG). */
function blobs(n, d, centers, seed = 1) {
  let s = seed >>> 0;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) - 0.5;
  const c = Array.from({ length: centers * d }, () => rnd() * 8);
  const x = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const j = i % centers;
    for (let t = 0; t < d; t++) x[i * d + t] = c[j * d + t] + rnd();
  }
  return x;
}

test("train and assign return typed arrays of the right length", () => {
  const [n, d, k] = [3000, 8, 12];
  const x = blobs(n, d, 12);
  const m = em.train(x, d, k, { seed: 3, nepochs: 2 });
  assert.equal(m.d, d);
  assert.equal(m.k, k);
  assert.equal(m.kernel, "cpu-gemm", "gemm is the default");
  assert.ok(m.centroids() instanceof Float32Array);
  assert.equal(m.centroids().length, k * d);
  assert.ok(m.trainLabels() instanceof Int32Array);
  const a = m.assign(x);
  assert.ok(a.labels instanceof Uint32Array && a.scores instanceof Float32Array);
  assert.equal(a.labels.length, n);
  assert.ok(a.labels.every((l) => l < k));
  assert.equal(m.stats().length, 2);
  const again = em.train(x, d, k, { seed: 3, nepochs: 2 });
  assert.deepEqual(again.centroids(), m.centroids(), "same seed, same model");
  const b = em.Model.fromBytes(m.toBytes());
  assert.deepEqual(b.assign(x).labels, a.labels);
  assert.equal(b.trainLabels(), undefined);
  for (const o of [m, again, b]) o.free();
});

test("errors are thrown, not swallowed", () => {
  const x = blobs(200, 4, 3);
  assert.throws(() => em.train(x, 4, 3, { nepoch: 2 }), /unknown option "nepoch"/);
  assert.throws(() => em.train(x, 4, 3, { kernel: "accelerate" }), /unknown kernel/);
  assert.throws(() => em.train(x, 4, 3, { d: 4 }), /pass d and k as arguments/);
  assert.throws(() => em.train(x.subarray(0, 7), 4, 3), /not a multiple of d/);
  assert.throws(() => em.train(x, 4, 500), /need at least k/);
  assert.throws(() => em.Model.fromBytes(new Uint8Array(20)), /not an emeans model/);
  const m = em.train(x, 4, 3);
  const bytes = m.toBytes();
  assert.throws(() => em.Model.fromBytes(bytes.subarray(0, bytes.length - 1)), /bytes, expected/);
  assert.throws(() => m.assign(x.subarray(0, 5)), /not a multiple of d/);
  assert.throws(() => em.Hierarchy.fromBytes(bytes), /not an emeans hierarchy/);
  m.free();
});

test("dedup accepts plain arrays and checks their length", () => {
  const d = 4;
  const x = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]);
  const r = em.dedup(x, d, { k: 2 }, { priority: [0, 1, 0] });
  assert.deepEqual([...r.groupOf], [0, 0, 1]);
  assert.deepEqual([...r.representatives], [1, 2], "priority picks row 1");
  assert.throws(() => em.dedup(x, d, {}, { burst: [1, 2] }), /burst has 2 entries for 3 rows/);
});

test("the wasm holds no absolute home path", () => {
  const text = readFileSync(new URL("../pkg/emeans_bg.wasm", import.meta.url)).toString("latin1");
  for (const p of ["/Users/", "/home/"]) assert.ok(!text.includes(p), `found ${p} in the wasm`);
});

test("wasm size budget: under 1 MiB raw, 400 KiB gzip", () => {
  const b = readFileSync(new URL("../pkg/emeans_bg.wasm", import.meta.url));
  assert.ok(b.length < 1024 * 1024, `${b.length} bytes`);
  assert.ok(gzipSync(b, { level: 9 }).length < 400 * 1024);
});
