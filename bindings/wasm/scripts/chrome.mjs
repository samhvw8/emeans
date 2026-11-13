// Open a page of bindings/wasm in headless Chrome, wait for `window.__emeansResult`,
// print it as JSON. Exit 1 when the page reports an error or times out.
//
// node scripts/chrome.mjs "<path?query>" [--shot out/page.png] [--timeout 1800]
// CHROME=/path/to/chrome overrides the browser (default: Google Chrome on macOS,
// google-chrome / chromium on Linux).
import { spawn, execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const argv = process.argv.slice(2);
const page = argv.find((a, i) => !a.startsWith("--") && !argv[i - 1]?.startsWith("--"));
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
if (!page) throw new Error('usage: chrome.mjs "<path?query>" [--shot file] [--timeout seconds]');
const here = fileURLToPath(new URL("..", import.meta.url));
const timeoutMs = Number(opt("timeout", 1800)) * 1000;

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  if (existsSync(mac)) return mac;
  for (const bin of ["google-chrome", "chromium", "chromium-browser"]) {
    try {
      return execFileSync("which", [bin], { encoding: "utf8" }).trim();
    } catch {}
  }
  throw new Error("no Chrome found; set CHROME");
}

const port = 8790 + Math.floor(Math.random() * 100);
const server = spawn(process.execPath, [join(here, "scripts/serve.mjs"), String(port)], { stdio: "pipe" });
await new Promise((ok) => server.stdout.once("data", ok));
const profile = mkdtempSync(join(tmpdir(), "emeans-chrome-"));
const browser = spawn(findChrome(), [
  "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--window-size=1000,900", "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
const cleanup = () => {
  browser.kill();
  server.kill();
  try { rmSync(profile, { recursive: true, force: true }); } catch {}
};

try {
  const wsUrl = await new Promise((ok, fail) => {
    let buf = "";
    browser.stderr.on("data", (d) => {
      buf += d;
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) ok(m[1]);
    });
    setTimeout(() => fail(new Error(`Chrome did not start: ${buf.slice(-500)}`)), 30000);
  });
  const ws = new WebSocket(wsUrl);
  await new Promise((ok) => ws.addEventListener("open", ok, { once: true }));
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { ok, fail } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? fail(new Error(JSON.stringify(m.error))) : ok(m.result);
    }
  });
  const send = (method, params = {}, sessionId) =>
    new Promise((ok, fail) => {
      const msg = { id: ++id, method, params };
      if (sessionId) msg.sessionId = sessionId;
      pending.set(msg.id, { ok, fail });
      ws.send(JSON.stringify(msg));
    });
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const url = `http://localhost:${port}/${page.replace(/^\//, "")}`;
  await send("Page.navigate", { url }, sessionId);
  const deadline = Date.now() + timeoutMs;
  let result = null;
  while (Date.now() < deadline) {
    // A long synchronous wasm call blocks the page; the evaluate waits for it.
    const r = await send("Runtime.evaluate", {
      expression: "JSON.stringify(window.__emeansResult ?? null)", returnByValue: true,
    }, sessionId);
    result = JSON.parse(r.result.value);
    if (result) break;
    await new Promise((ok) => setTimeout(ok, 250));
  }
  const shot = opt("shot");
  if (shot) {
    const png = await send("Page.captureScreenshot", { format: "png" }, sessionId);
    const file = join(here, shot);
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, Buffer.from(png.data, "base64"));
  }
  const ua = await send("Runtime.evaluate", { expression: "navigator.userAgent", returnByValue: true }, sessionId);
  console.log(JSON.stringify({ url, userAgent: ua.result.value, result }));
  if (!result || result.error) process.exitCode = 1;
  ws.close();
} finally {
  cleanup();
}
