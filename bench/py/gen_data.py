# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0"]
# ///
"""Write synthetic Gaussian-mixture vectors in the project data contract format.

Output: bench/data/<name>.f32 (raw little-endian f32, row-major, n x d) and
bench/data/<name>.json ({n, d, spherical, centers, sigma, seed}).

Spherical sets: centers are random unit vectors, each point is
center + sigma * N(0, I/d) projected back onto the unit sphere.
Euclidean sets: centers ~ N(0, I), points = center + sigma * N(0, I).
"""

import argparse
import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def mixture(n, d, centers, sigma, spherical, seed):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((centers, d)).astype(np.float32)
    if spherical:
        c /= np.linalg.norm(c, axis=1, keepdims=True)
    # Unequal cluster sizes: Dirichlet weights, like a real photo library.
    w = rng.dirichlet(np.full(centers, 0.7))
    lab = rng.choice(centers, size=n, p=w)
    x = np.empty((n, d), dtype=np.float32)
    step = 16384
    for b in range(0, n, step):
        e = min(b + step, n)
        noise = rng.standard_normal((e - b, d)).astype(np.float32)
        if spherical:
            noise *= np.float32(sigma / np.sqrt(d))
            rows = c[lab[b:e]] + noise
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        else:
            rows = c[lab[b:e]] + np.float32(sigma) * noise
        x[b:e] = rows
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--centers", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--euclidean", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    x = mixture(a.n, a.d, a.centers, a.sigma, not a.euclidean, a.seed)
    path = DATA / f"{a.name}.f32"
    x.astype("<f4").tofile(path)
    meta = dict(n=a.n, d=a.d, spherical=not a.euclidean, centers=a.centers,
                sigma=a.sigma, seed=a.seed)
    (DATA / f"{a.name}.json").write_text(json.dumps(meta))
    print(path, x.shape)


if __name__ == "__main__":
    main()
