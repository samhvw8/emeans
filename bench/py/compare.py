# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare Python (py_*.json) and Rust (rs_*.json) parity runs; print a Markdown table.

Usage: compare.py RESULTS_DIR
"""

import json
import pathlib
import statistics as st
import sys

RECIPE_KEYS = ["batch_size", "alpha0", "steps_per_pass", "total_steps", "epoch_rows"]


def ms(v):
    return f"{st.mean(v):.6f} ± {st.stdev(v) if len(v) > 1 else 0:.6f}"


def main():
    res = pathlib.Path(sys.argv[1])
    print("| Case | seeds | Python final obj (mean ± sd) | Rust final obj (mean ± sd) | Rel. diff of means | Max per-seed rel. diff | Recipe equal | Splits py / rs |")
    print("|---|---|---|---|---|---|---|---|")
    worst = 0.0
    for py in sorted(res.glob("py_*.json")):
        rs = py.with_name("rs_" + py.name[3:])
        if not rs.exists():
            continue
        p, r = json.loads(py.read_text()), json.loads(rs.read_text())
        pf = [x["final_objective"] for x in p["runs"]]
        rf = [x["final_objective"] for x in r["runs"]]
        rel = abs(st.mean(rf) - st.mean(pf)) / st.mean(pf)
        per = max(abs(a - b) / a for a, b in zip(pf, rf))
        worst = max(worst, rel)
        same = all(
            all(x["recipe"][key] == y["recipe"][key] for key in RECIPE_KEYS)
            for x, y in zip(p["runs"], r["runs"])
        )
        splits = "/".join(str(x["n_splits"]) for x in p["runs"]) + " vs " + "/".join(str(x["n_splits"]) for x in r["runs"])
        case = py.stem[3:]
        print(f"| {case} | {len(pf)} | {ms(pf)} | {ms(rf)} | {rel:.2e} | {per:.2e} | {same} | {splits} |")
    print(f"\nworst relative difference of means: {worst:.2e}")


if __name__ == "__main__":
    main()
