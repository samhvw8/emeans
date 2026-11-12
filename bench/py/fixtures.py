# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "faiss-cpu>=1.9"]
# ///
"""Write exact-match fixtures for the Rust unit tests.

- recipes: _resolve_recipe / _epoch_rows / steps for many (n, k, nepochs, law, d)
- alphas: _scheduled_alpha samples
- mt19937: numpy legacy RandomState(seed).permutation(n) and random_sample(k)
- runs: NumpyEmeans on two tiny data sets (raw f32 files next to the JSON):
  per-epoch objective, imbalance, splits, final centroids and last-pass labels

Writes crates/emeans-core/tests/fixtures/reference.json and tiny_*.f32.
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
from faiss_e_means import NumpyEmeans  # noqa: E402
from faiss_e_means.base import (  # noqa: E402
    _epoch_rows, _epoch_seed, _resolve_batch_law, _resolve_recipe,
    _resolve_unused_threshold, _scheduled_alpha,
)

recipes = []
for n in [20_000, 17_000, 50_000, 100_000, 200_000, 1_000, 37, 1_000_003]:
    for k in [8, 64, 256, 1024, 4096]:
        for nepochs in [1, 2.5, 3, 5, 0.4, 10]:
            for law in ["auto", "throughput", "quality"]:
                for d in [16, 768]:
                    resolved, reason = _resolve_batch_law(law, d)
                    b, a0 = _resolve_recipe(n, k, nepochs, resolved, None, None)
                    rows = _epoch_rows(n, nepochs)
                    steps = [(r + b - 1) // b for r in rows]
                    recipes.append(dict(
                        n=n, k=k, nepochs=nepochs, batch_law=law, d=d,
                        resolved_law=resolved, batch_size=b, alpha0=a0,
                        steps_per_pass=(n + b - 1) // b, epoch_rows=rows,
                        epoch_steps=steps, total_steps=sum(steps),
                        unused_threshold=_resolve_unused_threshold(None, d),
                    ))
# Given batch size, with and without alpha0.
for n, k, bs, a0 in [(20_000, 64, 1000, None), (20_000, 64, 50_000, None),
                     (200_000, 1024, 4096, 0.3)]:
    b, alpha = _resolve_recipe(n, k, 5, "throughput", bs, a0)
    recipes.append(dict(n=n, k=k, nepochs=5, batch_law="given", d=768,
                        given_batch_size=bs, given_alpha0=a0,
                        batch_size=b, alpha0=alpha))

alphas = []
for a0 in [0.9, 0.05, 1.0]:
    for total in [1, 7, 100]:
        for step in range(0, total + 2, max(1, total // 9)):
            alphas.append(dict(alpha0=a0, step=step, total=total,
                               alpha=_scheduled_alpha(a0, step, total)))

mt = []
for seed in [0, 1234, _epoch_seed(1234, 3), 2**32 - 1]:
    for n in [1, 2, 10, 1000, 70_000]:
        perm = np.random.RandomState(seed).permutation(n)
        mt.append(dict(seed=seed, n=n, head=perm[:64].tolist(),
                       sum=int(perm[: min(n, 5000)].astype(np.int64) @ np.arange(min(n, 5000)))))
    rs = np.random.RandomState(seed)
    mt.append(dict(seed=seed, random_sample=rs.random_sample(12).tolist()))

FIX = ROOT / "crates/emeans-core/tests/fixtures"
FIX.mkdir(parents=True, exist_ok=True)


def tiny(name, n, d, centers, spherical, seed):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((centers, d)).astype(np.float32) * 3
    x = c[rng.integers(0, centers, n)] + rng.standard_normal((n, d)).astype(np.float32)
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = x.astype("<f4")
    x.tofile(FIX / f"{name}.f32")
    return x


runs = []
for name, n, d, centers, spherical, k, nepochs, law, seeds in [
    ("tiny_e8", 1500, 8, 9, False, 12, 3, "auto", [1234, 7]),
    ("tiny_s24", 1000, 24, 6, True, 10, 2.5, "throughput", [1234, 99]),
    ("tiny_e8", 1500, 8, 9, False, 40, 1.5, "quality", [5]),
]:
    x = tiny(name, n, d, centers, spherical, 17)
    for seed in seeds:
        m = NumpyEmeans(d, k, nepochs=nepochs, batch_law=law, seed=seed, spherical=spherical)
        cent, labels = m.train(x)
        score, alabels = m.assign(x)
        runs.append(dict(
            data=name, n=n, d=d, k=k, nepochs=nepochs, batch_law=law, seed=seed,
            spherical=spherical, batch_size=m.recipe["batch_size"], alpha0=m.recipe["alpha0"],
            obj=[float(s["obj"]) for s in m.iteration_stats],
            imbalance=[float(s["imbalance_factor"]) for s in m.iteration_stats],
            nsplit=[int(s["nsplit"]) for s in m.iteration_stats],
            centroids=[float(v) for v in cent.ravel()],
            labels=[int(v) for v in labels],
            assign_labels=[int(v) for v in alabels],
            assign_scores=[float(v) for v in score],
        ))

out = FIX / "reference.json"
out.write_text(json.dumps(dict(recipes=recipes, alphas=alphas, mt19937=mt, runs=runs), indent=0))
print(out, len(recipes), len(alphas), len(mt), len(runs), [r["nsplit"] for r in runs])
