# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "scikit-learn>=1.5"]
# ///
"""Write scikit-learn reference values for crates/emeans-cluster/src/metrics.rs.

Three small labelled data sets of unit rows (raw f32 next to the JSON) with
silhouette (cosine), Calinski-Harabasz, Davies-Bouldin, and ARI / NMI between
the labels and a second labelling.

Writes crates/emeans-cluster/tests/fixtures/metrics.json and metrics_*.f32.
"""

import json
import pathlib

import numpy as np
from sklearn import metrics

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/emeans-cluster/tests/fixtures"

cases = []
for name, n, d, k, seed in [("a", 300, 24, 5, 1), ("b", 517, 64, 9, 2), ("c", 120, 8, 2, 3)]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    labels = rng.integers(0, k, size=n)
    x = centers[labels] + 0.8 * rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = x.astype(np.float32)
    other = np.where(rng.random(n) < 0.2, rng.integers(0, k + 2, size=n), labels)
    (OUT / f"metrics_{name}.f32").write_bytes(x.astype("<f4").tobytes())
    xd = x.astype(np.float64)
    cases.append(dict(
        name=name, n=n, d=d, k=k,
        labels=labels.tolist(), other=other.tolist(),
        silhouette=float(metrics.silhouette_score(xd, labels, metric="cosine")),
        calinski_harabasz=float(metrics.calinski_harabasz_score(xd, labels)),
        davies_bouldin=float(metrics.davies_bouldin_score(xd, labels)),
        ari=float(metrics.adjusted_rand_score(labels, other)),
        nmi=float(metrics.normalized_mutual_info_score(labels, other)),
    ))

(OUT / "metrics.json").write_text(json.dumps(dict(cases=cases), indent=1))
print(f"wrote {len(cases)} cases to {OUT / 'metrics.json'}")
