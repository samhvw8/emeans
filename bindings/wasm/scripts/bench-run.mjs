// Wasm vs native benchmark: the same `blobs` data and seed in native Rust, Node.js
// and headless Chrome; train time, peak memory and the SHA-256 of the centroids.
//
// node scripts/bench-run.mjs [--shapes 16000x256,16000x1024,...] [--d 768] [--seed 1234]
//   [--native fused,gemm] [--threads 1,all] [--wasm gemm,fused] [--runtimes node,chrome]
//   [--slow-max-n 16000] [--slow-jobs 1] [--out out/bench.json]
//
// Needs pkg/ (mise run build). Builds the native `bench` example itself. Every run is
// its own process (clean peak RSS). wasm `fused` is 10x slower (software FMA):
// `--slow-max-n` caps its n in the sequential runs; with `--slow-jobs N` (N > 1) its
// Node runs leave the sequential loop and run at the end, N processes (one thread
// each) at a time, on every shape. On a shared machine, hold its timing lock for the
// whole run (on the Ubuntu box: `flock /tmp/emeans-box-gpu.lock node scripts/bench-run.mjs ...`).
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { availableParallelism, cpus, hostname, platform, arch } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const argv = process.argv.slice(2);
if (argv.includes("--help")) {
  console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(0, 10).join("\n"));
  process.exit(0);
}
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const list = (name, dflt) => opt(name, dflt).split(",").filter(Boolean);
const here = fileURLToPath(new URL("..", import.meta.url));
const mac = platform() === "darwin";
const d = Number(opt("d", 768));
const seed = Number(opt("seed", 1234));
const shapes = list("shapes", "16000x256,16000x1024,100000x256,100000x1024,200000x256,200000x1024")
  .map((s) => s.split("x").map(Number));
const nativeKernels = list("native", mac ? "fused,fastest" : "fused,gemm");
const threads = list("threads", "1,all");
const wasmKernels = list("wasm", "gemm,fused");
const runtimes = list("runtimes", "node,chrome");
const slowMaxN = Number(opt("slow-max-n", 16000));
const slowJobs = Number(opt("slow-jobs", 1));
const outFile = join(here, opt("out", `out/bench-${hostname().split(".")[0]}.json`));
mkdirSync(join(here, "out/bench"), { recursive: true });

const sha = (file) => createHash("sha256").update(readFileSync(file)).digest("hex");
const run = (cmd, args, env = {}) =>
  new Promise((ok, fail) => {
    const p = spawn(cmd, args, { cwd: here, env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] });
    let out = "", err = "";
    p.stdout.on("data", (b) => (out += b));
    p.stderr.on("data", (b) => (err += b));
    p.on("close", (code) => {
      const line = out.trim().split("\n").pop();
      if (code !== 0) return fail(new Error(`${cmd} ${args.join(" ")} exited ${code}: ${err.slice(-800)}`));
      try { ok(JSON.parse(line)); } catch { fail(new Error(`no JSON from ${cmd}: ${out.slice(-400)} ${err.slice(-400)}`)); }
    });
  });

console.error("building the native bench ...");
execFileSync("cargo", ["build", "--release", "--quiet", "--example", "bench"], { cwd: here, stdio: "inherit" });
const targetDir = process.env.CARGO_TARGET_DIR ?? join(here, "target");
const nativeBin = join(targetDir, "release/examples/bench");

const host = {
  host: hostname().split(".")[0], platform: platform(), arch: arch(),
  cpu: cpus()[0]?.model, logical_cpus: availableParallelism(), node: process.version,
  wasm_bytes: readFileSync(join(here, "pkg/emeans_bg.wasm")).length, d, seed,
};
const rows = [];
const record = (r) => {
  rows.push(r);
  console.error(`${r.n}x${r.k} ${r.runtime.padEnd(14)} ${String(r.kernel).padEnd(15)} ${String(r.threads).padEnd(3)} ` +
    `${(r.train_ms / 1000).toFixed(2).padStart(8)} s  ${r.centroids_sha256.slice(0, 12)}`);
  writeFileSync(outFile, JSON.stringify({ ...host, rows }, null, 2));
};

for (const [n, k] of shapes) {
  for (const kernel of nativeKernels) {
    for (const t of threads) {
      const out = join(here, `out/bench/native-${n}x${k}-${kernel}-${t}.f32`);
      const env = t === "all" ? {} : { RAYON_NUM_THREADS: t };
      const r = await run(nativeBin, [n, d, k, kernel, seed, out].map(String), env);
      record({ ...r, runtime: "native", threads: t === "all" ? availableParallelism() : Number(t), centroids_sha256: sha(out) });
    }
  }
  for (const kernel of wasmKernels) {
    if (kernel === "fused" && n > slowMaxN) continue;
    if (runtimes.includes("node") && !(kernel === "fused" && slowJobs > 1)) {
      const r = await run(process.execPath, ["scripts/bench-node.mjs", n, d, k, kernel, seed].map(String));
      record({ ...r, runtime: "node", threads: 1 });
    }
    if (runtimes.includes("chrome")) {
      const q = new URLSearchParams({ n, d, k, kernel, seed });
      const r = await run(process.execPath, ["scripts/chrome.mjs", `demo/bench.html?${q}`, "--timeout", "3600"]);
      record({ ...r.result, runtime: r.result.runtime, threads: 1 });
    }
  }
}

// Slow wasm `fused` Node runs in parallel, one process (one thread) each.
if (slowJobs > 1 && runtimes.includes("node") && wasmKernels.includes("fused")) {
  const jobs = [...shapes];
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(slowJobs, jobs.length) }, async () => {
    while (next < jobs.length) {
      const [n, k] = jobs[next++];
      const r = await run(process.execPath, ["scripts/bench-node.mjs", n, d, k, "fused", seed].map(String));
      record({ ...r, runtime: "node", threads: 1, parallel_jobs: slowJobs });
    }
  }));
}

// Bit equality: every run's centroids against native runs of the same kernel.
const nativeHash = new Map();
for (const r of rows.filter((r) => r.runtime === "native")) {
  const key = `${r.n}x${r.k}/${r.kernel}`;
  if (!nativeHash.has(key)) nativeHash.set(key, new Set());
  nativeHash.get(key).add(r.centroids_sha256);
}
for (const r of rows) {
  const s = nativeHash.get(`${r.n}x${r.k}/${r.kernel}`);
  r.same_as_native = s ? s.has(r.centroids_sha256) && s.size === 1 : null;
}
writeFileSync(outFile, JSON.stringify({ ...host, rows }, null, 2));

const gib = (b) => (b ? `${(b / 2 ** 30).toFixed(2)}` : "");
console.log(`\n${host.cpu}, ${host.logical_cpus} logical CPUs, ${host.platform}-${host.arch}, Node ${host.node}; d = ${d}, seed ${seed}; wasm ${(host.wasm_bytes / 1024).toFixed(0)} KiB\n`);
console.log("| n | k | runtime | kernel | threads | train s | peak RSS GiB | wasm memory GiB | centroids SHA-256 | = native |");
console.log("|---|---|---|---|---|---|---|---|---|---|");
for (const r of rows) {
  console.log(`| ${r.n} | ${r.k} | ${r.runtime} | ${r.kernel} | ${r.threads}${r.parallel_jobs ? ` (${r.parallel_jobs} jobs)` : ""} | ` +
    `${(r.train_ms / 1000).toFixed(2)} | ${gib(r.peak_rss_bytes)} | ${gib(r.wasm_memory_bytes)} | ` +
    `\`${r.centroids_sha256.slice(0, 12)}\` | ${r.same_as_native === null ? "n/a" : r.same_as_native ? "yes" : "NO"} |`);
}
console.log(`\nwrote ${outFile}`);
