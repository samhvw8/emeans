"""e-means clustering on CPU, Metal and CUDA.

Inputs are float32 buffers (NumPy arrays, ``array.array('f')``, ``memoryview``,
``bytes``), read without a copy. Outputs are :class:`Array` buffers; call
``.numpy()`` on one for a NumPy view (needs NumPy).
"""

from ._emeans import (
    AddReport,
    Array,
    Assignment,
    DedupResult,
    EmeansError,
    EpochStats,
    Hierarchy,
    Labels,
    TrainResult,
    __version__,
    assign,
    auto_cluster,
    backends,
    dedup,
    default_config,
    train,
)

__all__ = [
    "AddReport",
    "Array",
    "Assignment",
    "DedupResult",
    "EmeansError",
    "EpochStats",
    "Hierarchy",
    "Labels",
    "TrainResult",
    "__version__",
    "assign",
    "auto_cluster",
    "backends",
    "dedup",
    "default_config",
    "train",
]
