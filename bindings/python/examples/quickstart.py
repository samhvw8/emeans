"""emeans quickstart: train, assign, auto_cluster, add new rows, save and load.

Run: mise run example:python   (or: python examples/quickstart.py [backend])
Needs only the emeans package; the data is synthetic (16 blobs of unit rows).
"""

import array
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import emeans

backend = sys.argv[1] if len(sys.argv) > 1 else "cpu"
n, d, blobs = 20_000, 64, 16
rng = random.Random(7)


def unit_rows(count: int) -> array.array:
    """`count` unit rows near one of `blobs` random centers."""
    centers = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(blobs)]
    out = array.array("f")
    for _ in range(count):
        c = centers[rng.randrange(blobs)]
        row = [v + rng.gauss(0, 0.35) for v in c]
        norm = math.sqrt(sum(v * v for v in row))
        out.extend(v / norm for v in row)
    return out


def timed(label, f):
    t = time.perf_counter()
    r = f()
    print(f"{label:<28} {1000 * (time.perf_counter() - t):8.1f} ms")
    return r


x = unit_rows(n)  # array.array('f'): read by emeans without a copy
print(f"emeans {emeans.__version__}: {n} x {d} rows, backend {backend} (available: {', '.join(emeans.backends())})")

t = timed("train k=32", lambda: emeans.train(x, 32, d=d, spherical=True, seed=7, backend=backend))
print(f"  objective {t.objective:.4f} after {len(t.stats)} epochs on {t.backend}")
a = timed("assign", lambda: emeans.assign(x, t.centroids, metric="inner_product", backend=backend))
print(f"  first labels {a.labels.tolist()[:10]}")

n_old = 16_000
h, labels = timed("auto_cluster (16k rows)", lambda: emeans.auto_cluster(memoryview(x)[: n_old * d], d=d, seed=7, k1_range=[2, 24], restarts=2))
k1, k2 = h.chosen_k
print(f"  level 1: {k1} groups, {sum(k2)} leaves; first l1 labels {labels.l1.tolist()[:10]}")

with tempfile.TemporaryDirectory() as tmp:
    timed("save", lambda: h.save(Path(tmp) / "model"))
    h = timed("load", lambda: emeans.Hierarchy.load(Path(tmp) / "model"))

r = timed("add 4k new rows", lambda: h.add(x))  # all rows seen so far, then the new ones
new_l1 = r.labels.l1.tolist()[n_old:]
print(f"  action {r.action}, {r.new_outliers} new outliers, model has seen {h.n_seen} rows")
print(f"  labels of the new rows: {new_l1[:10]} ...")
