// Load pkg/ in Node.js: the `--target web` module, initialized from the file's bytes.
import { readFileSync } from "node:fs";
import * as em from "../pkg/emeans.js";

export function load() {
  em.initSync({ module: readFileSync(new URL("../pkg/emeans_bg.wasm", import.meta.url)) });
  return em;
}
