#!/usr/bin/env python3
"""Benchmark every binding against native Rust on the same data and seed.

    python3 bindings/bench/run.py --data DATA.f32 --d 768 --ks 64,256,1024 \
        --backends cpu-fused [--langs rust,python,node,go] [--reps 3] [--lock FILE] --out DIR

Build first (`mise run bind:bench:build`). For each k and backend it runs
every language `--reps` times, in a rotated order each round, each run in a
fresh process under `/usr/bin/time` (peak RSS). Each program prints one JSON
line (bindings/common/examples/bench.rs and the `examples/bench.*` of each
binding). The table shows medians. Writes DIR/runs.jsonl and DIR/table.md.

Timings start after the data is in the host language (NumPy array,
Float32Array, []float32, Vec<f32>). The call cost is a separate run per
language (mode `call`, `RAYON_NUM_THREADS=1` so thread wake-ups do not hide
it): the mean of 2000 assign calls of one row against 64 centroids on `cpu`.
Results must be byte-identical: every run's centroid and label hashes are
compared with Rust's.
"""

import argparse
import fcntl
import json
import os
import platform
import statistics
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def commands(target: Path) -> dict[str, list[str]]:
    return {
        "rust": [str(target / "release/examples/bench")],
        "python": [str(ROOT / "bindings/python/.venv/bin/python"), str(ROOT / "bindings/python/examples/bench.py")],
        "node": ["node", str(ROOT / "bindings/node/examples/bench.mjs")],
        "go": [str(target / "go-bench")],
    }


def peak_rss_mb(stderr: str) -> float | None:
    if m := re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr):  # GNU time
        return int(m.group(1)) / 1024
    if m := re.search(r"(\d+)\s+maximum resident set size", stderr):  # BSD time (bytes)
        return int(m.group(1)) / 2**20
    return None


def run_one(cmd: list[str], data: str, d: int, k: int, backend: str, mode: str = "full") -> dict:
    flag = "-v" if platform.system() == "Linux" else "-l"
    env = dict(os.environ, RAYON_NUM_THREADS="1") if mode == "call" else None
    p = subprocess.run(["/usr/bin/time", flag, *cmd, data, str(d), str(k), backend, mode],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)} k={k} {backend}\n{p.stderr[-3000:]}")
    line = [l for l in p.stdout.splitlines() if l.startswith("{")][-1]
    r = json.loads(line)
    r["peak_rss_mb"] = peak_rss_mb(p.stderr)
    r["requested_backend"] = backend
    return r


def med(rs: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in rs)


def table(runs: list[dict], calls: list[dict], langs: list[str]) -> str:
    out = [] if not runs else [
        "| k | backend | lang | train s (median) | vs Rust | assign s (median) | vs Rust | peak RSS MB | runs | = Rust |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (k, b) in dict.fromkeys((r["k"], r["requested_backend"]) for r in runs):
        group = [r for r in runs if r["k"] == k and r["requested_backend"] == b]
        rust = [r for r in group if r["lang"] == "rust"]
        for lang in langs:
            rs = [r for r in group if r["lang"] == lang]
            pct = lambda key: "—" if lang == "rust" else f"{100 * (med(rs, key) / med(rust, key) - 1):+.1f}%"
            same = all(r["centroids_fnv"] == rust[0]["centroids_fnv"] and r["labels_fnv"] == rust[0]["labels_fnv"] for r in rs)
            out.append(
                f"| {k} | {rs[0]['backend']} | {lang} | {med(rs, 'train_s'):.3f} | {pct('train_s')} | {med(rs, 'assign_s'):.3f} "
                f"| {pct('assign_s')} | {max(r['peak_rss_mb'] for r in rs):.0f} | {len(rs)} | {'yes' if same else '**NO**'} |"
            )
    if calls:
        rust_call = med([c for c in calls if c["lang"] == "rust"], "call_us")
        out += ["", "| lang | call µs, median (1 row x 64 centroids, 1 thread) | min - max | overhead vs Rust µs |",
                "|---|---|---|---|"]
        for lang in langs:
            cs = [c for c in calls if c["lang"] == lang]
            m = med(cs, "call_us")
            over = "—" if lang == "rust" else f"{m - rust_call:+.1f}"
            out.append(f"| {lang} | {m:.1f} | {min(c['call_us'] for c in cs):.1f} - {max(c['call_us'] for c in cs):.1f} | {over} |")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--ks", default="64,256,1024")
    p.add_argument("--backends", default="cpu")
    p.add_argument("--langs", default="rust,python,node,go")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--target", default=str(ROOT / "target/bench"))
    p.add_argument("--lock", help="hold this lock file (flock) for the whole run")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cmds = commands(Path(a.target))
    lock = open(a.lock, "a") if a.lock else None
    if lock:
        print(f"waiting for {a.lock}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
    langs = a.langs.split(",")
    runs, calls = [], []
    with open(out / "runs.jsonl", "w") as f:
        for rep in range(a.reps):
            for lang in langs[rep % len(langs):] + langs[: rep % len(langs)]:
                c = run_one(cmds[lang], a.data, a.d, 64, "cpu", "call")
                calls.append(c)
                f.write(json.dumps(c) + "\n")
                print(f"call {lang} #{rep}: {c['call_us']:.1f}us", flush=True)
        for k in (int(k) for k in a.ks.split(",") if k):
            for b in a.backends.split(","):
                for rep in range(a.reps):
                    for lang in langs[rep % len(langs):] + langs[: rep % len(langs)]:
                        t = time.time()
                        r = run_one(cmds[lang], a.data, a.d, k, b)
                        runs.append(r)
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        print(f"k={k} {b} {lang} #{rep}: train {r['train_s']:.3f}s assign {r['assign_s']:.3f}s "
                              f"rss {r['peak_rss_mb']:.0f}MB ({time.time() - t:.0f}s wall)", flush=True)
    text = table(runs, calls, langs)
    (out / "table.md").write_text(text)
    print(text)
    if not runs:
        return
    ref = {(r["k"], r["requested_backend"]): r for r in runs if r["lang"] == "rust"}
    bad = [r for r in runs if any(r[h] != ref[(r["k"], r["requested_backend"])][h] for h in ("centroids_fnv", "labels_fnv"))]
    if bad:
        sys.exit(f"NOT byte-identical to Rust: {[(r['lang'], r['k'], r['requested_backend']) for r in bad]}")


if __name__ == "__main__":
    main()
