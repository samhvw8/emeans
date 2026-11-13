// Bit equality of the wasm build with native Rust (out/golden from `mise run golden`).
//
// For every kernel the native golden run built, run the same wasm export on the same
// inputs and options and compare every output byte for byte. The result matrix goes
// to out/golden-report.json. What must match:
// - `fused`: always (one mul_add chain per dot product on every instruction set);
// - `blocked`: when the native host is not aarch64 (aarch64 uses NEON FMA tiles,
//   elsewhere the same portable multiply-add as wasm);
// - `gemm`: recorded, not required (the gemm crate picks its kernels per instruction set).
// `dedup` similarities come from gemm_nt (Accelerate on macOS), so for them only
// the groups must match, and the similarities within 4 ulps.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { load } from "./load.mjs";

const em = load();
const dir = new URL("../out/golden/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("cases.json", dir)));
const C = meta.cases;
const raw = (name) => {
  const b = readFileSync(new URL(name, dir));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const typed = { f32: Float32Array, f64: Float64Array, u32: Uint32Array, i32: Int32Array, u8: Uint8Array };
const read = (name) => new typed[name.split(".").pop()](raw(name));

/** null when the bytes match, else a description of the first difference. */
function diff(actual, expected) {
  const a = new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength);
  const e = new Uint8Array(expected.buffer, expected.byteOffset, expected.byteLength);
  if (a.length !== e.length) return `length ${a.length} vs ${e.length}`;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== e[i]) {
      const el = Math.floor(i / actual.BYTES_PER_ELEMENT);
      return `element ${el}: wasm ${actual[el]} vs native ${expected[el]}`;
    }
  }
  return null;
}

/** Largest distance in units in the last place between two Float32Arrays. */
function maxUlps(a, b) {
  const ia = new Int32Array(a.buffer, a.byteOffset, a.length);
  const ib = new Int32Array(b.buffer, b.byteOffset, b.length);
  const ord = (v) => (v < 0 ? -2147483648 - v : v);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(ord(ia[i]) - ord(ib[i])));
  return m;
}

/** Model file bytes with the wall-clock `time` of every epoch removed from the header. */
function withoutTimes(bytes) {
  const b = new Uint8Array(bytes);
  const h = Number(new DataView(b.buffer, b.byteOffset).getBigUint64(8, true));
  const header = JSON.parse(new TextDecoder().decode(b.subarray(16, 16 + h)));
  for (const s of header.iteration_stats) delete s.time;
  return { header: JSON.stringify(header), payload: b.subarray(16 + h) };
}

/** Run every case on wasm with `kernel`; return {case: {output: null | difference}}. */
function compare(kernel) {
  const p = (name) => `${kernel}/${name}`;
  const opts = (o) => ({ ...o, kernel });
  const r = {};
  for (const name of ["l2", "sph"]) {
    const c = C[name];
    const x = read(c.x);
    const m = em.train(x, c.d, c.k, opts(c.options));
    const a = m.assign(x);
    const ema = m.emaState();
    const mine = withoutTimes(m.toBytes());
    const theirs = withoutTimes(raw(p(`${name}.model.bin`)));
    const back = em.Model.fromBytes(new Uint8Array(raw(p(`${name}.model.bin`))), { kernel });
    r[name] = {
      centroids: diff(m.centroids(), read(p(`${name}.centroids.f32`))),
      train_labels: diff(m.trainLabels(), read(p(`${name}.train_labels.i32`))),
      ema_positions: diff(ema.positions, read(p(`${name}.positions.f32`))),
      ema_masses: diff(ema.usage, read(p(`${name}.usage.f32`))),
      assign_labels: diff(a.labels, read(p(`${name}.assign_labels.u32`))),
      assign_scores: diff(a.scores, read(p(`${name}.assign_scores.f32`))),
      model_header: mine.header === theirs.header ? null : "header differs",
      model_payload: diff(mine.payload, theirs.payload),
      native_file_assign: diff(back.assign(x).scores, read(p(`${name}.assign_scores.f32`))),
    };
    back.free();
    m.free();
  }
  const ac = C.auto;
  const x = read(ac.x);
  const { hierarchy, labels } = em.autoCluster(x.subarray(0, ac.n0 * ac.d), ac.d, opts(ac.options));
  const h = em.Hierarchy.fromBytes(hierarchy.toBytes(), { kernel });
  hierarchy.free();
  const add = h.add(x, ac.n0);
  const native = meta.native[kernel].auto;
  r.auto = {
    chosen_k: JSON.stringify(h.chosenK()) === JSON.stringify(native.chosen_k) ? null : "chosen k differs",
    action: add.action === native.action ? null : `action ${add.action} vs ${native.action}`,
  };
  for (const [stage, l] of [["fit", labels], ["add", add.labels]]) {
    for (const f of ["l1", "l2", "sim1", "sim", "outlier", "slot1", "slot2"]) {
      const ext = { sim1: "f32", sim: "f32", outlier: "u8" }[f] ?? "u32";
      r.auto[`${stage}.${f}`] = diff(l[f], read(p(`auto.${stage}.${f}.${ext}`)));
    }
  }
  h.free();
  const dc = C.dedup;
  const dd = em.dedup(read(dc.x), dc.d, opts(dc.options), { createdUnix: read(dc.created) });
  const nn = read(p("dedup.nn_sim.f32"));
  r.dedup = {
    group_of: diff(dd.groupOf, read(p("dedup.group_of.u32"))),
    representatives: diff(dd.representatives, read(p("dedup.representatives.u32"))),
    linked_pairs: dd.linkedPairs === meta.native[kernel].dedup.linked_pairs ? null : "linked pairs differ",
    nn_sim: diff(dd.nnSim, nn),
    nn_sim_max_ulps: maxUlps(dd.nnSim, nn),
  };
  return r;
}

const report = { native_host: meta.host, kernels: {} };
const aarch64 = meta.host.startsWith("aarch64");

for (const kernel of meta.kernels) {
  const required = kernel === "fused" || (kernel === "blocked" && !aarch64);
  test(`${kernel}: ${required ? "same bits as" : "compared with"} native ${meta.host}`, () => {
    const r = compare(kernel);
    const diffs = [];
    for (const [c, outs] of Object.entries(r)) {
      for (const [o, d] of Object.entries(outs)) {
        if (o === "nn_sim_max_ulps") continue;
        if (d !== null) diffs.push(`${c}.${o}: ${d}`);
      }
    }
    report.kernels[kernel] = { required, same_bits: diffs.length === 0, differences: diffs, nn_sim_max_ulps: r.dedup.nn_sim_max_ulps };
    if (required) {
      // dedup similarities may differ in the last bits (gemm_nt); everything else must not.
      const hard = diffs.filter((d) => !d.startsWith("dedup.nn_sim"));
      assert.deepEqual(hard, [], `${kernel} must give the same bits as native`);
      assert.ok(r.dedup.nn_sim_max_ulps <= 4, `nnSim differs by ${r.dedup.nn_sim_max_ulps} ulps`);
    }
  });
}

test("write the report", () => {
  writeFileSync(new URL("../out/golden-report.json", import.meta.url), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
});
