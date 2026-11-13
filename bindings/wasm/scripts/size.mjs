// Size report of pkg/emeans_bg.wasm: raw, gzip and brotli bytes.
import { readFileSync } from "node:fs";
import { gzipSync, brotliCompressSync } from "node:zlib";

const path = new URL("../pkg/emeans_bg.wasm", import.meta.url);
const b = readFileSync(path);
const kib = (n) => `${(n / 1024).toFixed(1)} KiB`;
console.log(
  `emeans_bg.wasm: ${kib(b.length)} raw, ${kib(gzipSync(b, { level: 9 }).length)} gzip, ` +
    `${kib(brotliCompressSync(b).length)} brotli`,
);
