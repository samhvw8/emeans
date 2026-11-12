# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "faiss-cpu>=1.9", "scipy>=1.13"]
# ///
"""Run the vendored faiss NumpyEmeans on a bench/data file and print JSON.

The vendored package lives in reference/faiss_e_means (read-only); it is
imported as the package `faiss_e_means`.

Output JSON: recipe, per-epoch iteration_stats, n_splits, and the final
objective: mean squared L2 distance (Euclidean) or mean(2 - 2 cos)
(spherical) of every row to its nearest final centroid.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
from faiss_e_means import NumpyEmeans  # noqa: E402


def load(name):
    meta = json.loads((ROOT / "bench/data" / f"{name}.json").read_text())
    x = np.fromfile(ROOT / "bench/data" / f"{name}.f32", dtype="<f4")
    return x.reshape(meta["n"], meta["d"]), meta


def final_objective(model, x):
    score, _ = model.assign(x)
    score = score.astype(np.float64)
    if model.spherical:
        return float(np.mean(2.0 - 2.0 * score))
    return float(np.mean(score))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--nepochs", type=float, default=5)
    p.add_argument("--batch-law", default="auto")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--alpha0", type=float)
    p.add_argument("--seeds", default="1234")
    p.add_argument("--spherical", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--max-points-per-centroid", type=int)
    p.add_argument("--dump-centroids")
    a = p.parse_args()
    x, meta = load(a.data)
    spherical = meta["spherical"] if a.spherical == "auto" else a.spherical == "yes"
    out = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        m = NumpyEmeans(
            x.shape[1], a.k, nepochs=a.nepochs, batch_size=a.batch_size,
            alpha0=a.alpha0, batch_law=a.batch_law, seed=seed,
            spherical=spherical, max_points_per_centroid=a.max_points_per_centroid,
        )
        t = time.perf_counter()
        m.train(x)
        secs = time.perf_counter() - t
        r = {k: v for k, v in m.recipe.items()}
        out.append(dict(
            seed=seed, seconds=secs, final_objective=final_objective(m, x),
            obj=[float(s["obj"]) for s in m.iteration_stats],
            imbalance=[float(s["imbalance_factor"]) for s in m.iteration_stats],
            nsplit=[int(s["nsplit"]) for s in m.iteration_stats],
            n_splits=int(m.n_splits), recipe=r,
        ))
        if a.dump_centroids:
            m.centroids.astype("<f4").tofile(f"{a.dump_centroids}.{seed}.f32")
    print(json.dumps(dict(data=a.data, k=a.k, nepochs=a.nepochs,
                          batch_law=a.batch_law, spherical=spherical, runs=out)))


if __name__ == "__main__":
    main()
