"""Python side of the binding benchmark (bindings/bench/run.py).

Usage: python bench.py <data.f32> <d> <k> <backend> <mode>; prints one JSON line.
Mode `full` times train and assign; `call` only the per-call probe.
The data is loaded into NumPy first: timings cover only the emeans calls.
"""

import json
import sys
import time

import numpy as np

import emeans


def fnv(buf) -> str:
    """FNV-1a 64 of the bytes (the same hash in every language bench)."""
    h = 0xCBF29CE484222325
    for b in bytes(memoryview(buf)):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


path, d, k, backend, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
x = np.fromfile(path, dtype=np.float32).reshape(-1, d)

if mode == "call":
    row, cents = x[:1], x[64:128]
    reps = 2000
    t = time.perf_counter()
    for _ in range(reps):
        emeans.assign(row, cents, metric="inner_product")
    print(json.dumps({"lang": "python", "call_us": (time.perf_counter() - t) * 1e6 / reps}))
    sys.exit(0)

t = time.perf_counter()
tr = emeans.train(x, k, spherical=True, backend=backend)
train_s = time.perf_counter() - t

t = time.perf_counter()
a = emeans.assign(x, tr.centroids, metric="inner_product", backend=backend)
assign_s = time.perf_counter() - t
objective = float(np.mean(2.0 - 2.0 * a.scores.numpy().astype(np.float64)))

print(json.dumps({
    "lang": "python", "backend": tr.backend, "n": x.shape[0], "d": d, "k": k,
    "train_s": train_s, "assign_s": assign_s, "objective": objective,
    "centroids_fnv": fnv(tr.centroids), "labels_fnv": fnv(a.labels),
}))
