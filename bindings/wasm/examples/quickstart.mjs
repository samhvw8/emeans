// emeans quickstart for Node.js: synthetic data -> train -> assign -> autoCluster
// -> add new rows -> toBytes / fromBytes. Prints labels and timings.
//
// Build pkg/ first (`mise run build`), then: `mise run example:wasm`
// (or `node examples/quickstart.mjs` in bindings/wasm).
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { initSync, train, autoCluster, Model, Hierarchy, version } from "../pkg/emeans.js";

initSync({ module: readFileSync(new URL("../pkg/emeans_bg.wasm", import.meta.url)) });

/** n x d rows around `groups` random unit directions, each row L2-normalized. */
function unitBlobs(n, d, groups, seed) {
  let s = seed >>> 0;
  const u = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) + 1) / 2 ** 32;
  const g = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
  const centers = Float32Array.from({ length: groups * d }, g);
  const x = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const j = i % groups;
    let norm = 0;
    for (let t = 0; t < d; t++) {
      const v = centers[j * d + t] + 0.35 * g();
      x[i * d + t] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    for (let t = 0; t < d; t++) x[i * d + t] /= norm;
  }
  return x;
}

const time = (label, f) => {
  const t = performance.now();
  const r = f();
  console.log(`${label.padEnd(34)} ${(performance.now() - t).toFixed(1).padStart(8)} ms`);
  return r;
};
const head = (a, n = 12) => `[${Array.from(a.slice(0, n)).join(", ")}, ...]`;

const [n, d, groups, k] = [6000, 64, 8, 8];
const x = unitBlobs(n, d, groups, 42);
console.log(`emeans ${version()}: ${n} rows x ${d} dims, ${groups} groups\n`);

// 1. Train k centroids (cosine: rows are unit vectors) and label every row.
const model = time(`train (k = ${k}, spherical)`, () => train(x, d, k, { spherical: true, seed: 7 }));
const { labels, scores } = time("assign", () => model.assign(x));
console.log(`  kernel ${model.kernel}; labels ${head(labels)}`);
console.log(`  mean cosine to the centroid ${(scores.reduce((a, b) => a + b, 0) / n).toFixed(4)}\n`);

// 2. Save and load the model as bytes (the native model file format).
const bytes = time("model toBytes + fromBytes", () => Model.fromBytes(model.toBytes()));
console.log(`  ${model.toBytes().length} bytes; same labels after reload: ${
  bytes.assign(x).labels.every((l, i) => l === labels[i])}\n`);

// 3. Automatic k (two levels) on the first 5,000 rows.
const n0 = 5000;
const { hierarchy, labels: l } = time("autoCluster (first 5,000 rows)", () =>
  autoCluster(x.subarray(0, n0 * d), d, { k1Range: [2, 12], k2Max: 6, restarts: 2, seed: 7 }));
const ck = hierarchy.chosenK();
console.log(`  level-1 k = ${ck.k1}, level-2 k per group = [${ck.k2}]`);
console.log(`  level-1 ids ${head(l.l1)}; leaf ids ${head(l.l2)}\n`);

// 4. Add the other 1,000 rows: they get labels, and the model updates by its policy.
const report = time("hierarchy.add (1,000 new rows)", () => hierarchy.add(x, n0));
console.log(`  action ${report.action}; ${report.newOutliers} new outliers; rows labelled ${report.labels.l1.length}`);
console.log(`  new rows' level-1 ids ${head(report.labels.l1.subarray(n0))}\n`);

// 5. Save and load the whole hierarchy as one byte blob.
const blob = hierarchy.toBytes();
const again = time("hierarchy toBytes + fromBytes", () => Hierarchy.fromBytes(blob));
const same = again.assign(x).l2.every((v, i) => v === report.labels.l2[i]);
console.log(`  ${blob.length} bytes; ${again.nSeen} rows seen; same leaf ids after reload: ${same}`);

for (const o of [model, bytes, hierarchy, again]) o.free();
