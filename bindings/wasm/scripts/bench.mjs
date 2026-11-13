// Wasm timings per CPU kernel on the golden inputs (out/golden): median of 3 runs.
// node scripts/bench.mjs  ->  prints JSON, writes out/bench.json
import { readFileSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { load } from "../tests/load.mjs";

const em = load();
const dir = new URL("../out/golden/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("cases.json", dir)));
const f32 = (name) => {
  const b = readFileSync(new URL(name, dir));
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const median = (v) => v.sort((a, b) => a - b)[Math.floor(v.length / 2)];
const out = { node: process.version, cases: {} };
for (const c of meta.cases.filter((c) => c.kind === "train")) {
  const x = f32(c.x);
  out.cases[c.name] = { n: x.length / c.d, d: c.d, k: c.k };
  for (const kernel of ["fused", "blocked", "gemm"]) {
    const tr = [], as = [];
    for (let r = 0; r < 3; r++) {
      let t = performance.now();
      const m = em.train(x, c.d, c.k, { ...c.options, kernel });
      tr.push(performance.now() - t);
      t = performance.now();
      m.assign(x);
      as.push(performance.now() - t);
      m.free();
    }
    out.cases[c.name][kernel] = { train_ms: median(tr), assign_ms: median(as) };
  }
}
writeFileSync(new URL("../out/bench.json", import.meta.url), JSON.stringify(out, null, 2));
console.log(JSON.stringify(out, null, 2));
