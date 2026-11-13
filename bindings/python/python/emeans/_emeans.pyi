"""Type stubs of the native module. Config keys mirror `default_config(kind)`."""

import os
from typing import Literal, TypedDict, final

from typing_extensions import Buffer, Unpack

__version__: str

Backend = Literal[
    "cpu", "cpu-blocked", "cpu-gemm", "cpu-accelerate", "cpu-fused", "metal", "cuda", "cuda-bf16"
]

class EmeansError(ValueError):
    """A call emeans refused or failed: bad shape, config, backend or file."""

@final
class Array(Buffer):
    """Read-only result array. `numpy.asarray(a)` and `memoryview(a)` read it without a copy."""

    @property
    def shape(self) -> list[int]: ...
    @property
    def dtype(self) -> Literal["float32", "uint32", "int32", "uint8"]: ...
    def __len__(self) -> int: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...
    def tolist(self) -> list[float] | list[int] | list[list[float]]: ...
    def numpy(self):  # -> numpy.ndarray (read-only view; needs NumPy)
        ...

@final
class EpochStats:
    obj: float
    time: float
    imbalance_factor: float
    nsplit: int

@final
class TrainResult:
    centroids: Array
    """(k, d) float32."""
    objective: float
    """Free: the last epoch's moving objective as a mean. That epoch's `obj` (the sum of the squared
    distance, or 2 - 2 cos when spherical, from each row it processed to its centroid when the row's
    batch was assigned) divided by the rows it processed. A little above `exact_objective`."""
    exact_objective: float | None
    """The mean over all rows against the final centroids (one extra assign pass); None unless
    `train(..., exact_objective=True)`."""
    stats: list[EpochStats]
    labels: Array | None
    """int32 last-pass label per row (-1 outside the training subset); None when capture_assignments is off."""
    backend: str

@final
class Assignment:
    labels: Array
    """uint32 nearest centroid per row."""
    scores: Array
    """float32 squared distance (l2) or inner product per row."""

@final
class Labels:
    l1: Array
    """uint32 stable level-1 id."""
    l2: Array
    """uint32 stable level-2 id inside the level-1 group."""
    sim1: Array
    sim: Array
    """float32 cosine to the leaf centroid."""
    outlier: Array
    """uint8, 1 when below the leaf's outlier floor."""

@final
class AddReport:
    action: Literal["assign_only", "ema_update", "full_retrain"]
    labels: Labels
    new_outliers: int
    pending: int
    outlier_fraction: float
    new_leaves: list[tuple[int, int]]

@final
class DedupResult:
    group_of: Array
    representatives: Array
    anchors: Array
    sizes: Array
    nn_sim: Array
    n_groups: int
    linked_pairs: int
    split_components: int

class TrainConfig(TypedDict, total=False):
    backend: Backend
    exact_objective: bool
    nepochs: float
    batch_size: int | None
    alpha0: float | None
    batch_law: Literal["auto", "throughput", "quality"]
    seed: int
    spherical: bool
    unused_threshold: float | None
    max_points_per_centroid: int | None
    assignment_chunk_size: int | None
    capture_assignments: bool
    verbose: bool

class IncrementalConfig(TypedDict, total=False):
    outlier_mad_k: float
    outlier_min_sim: float | None
    grow_min_size: int
    update_min_outliers: int
    update_added_fraction: float
    update_epochs: float
    retrain_added_fraction: float
    retrain_outlier_fraction: float
    match_min_sim: float

Criterion = Literal["silhouette", "calinski_harabasz", "davies_bouldin"]

class AutoConfig(TypedDict, total=False):
    backend: Backend
    k1_range: tuple[int, int] | list[int]
    k2_max: int
    criterion1: Criterion
    criterion2: Criterion
    level1_tolerance: float
    level2_tolerance: float
    min_cluster_size: int
    max_cluster_size: int
    min_split_size: int
    split_min_silhouette: float
    restarts: int
    nepochs: float
    init: Literal["k_means_plus_plus", "random_rows"]
    level1_method: Literal["bottom_up", "top_down"]
    resplit_passes: int
    level1_max_share: float
    seed: int
    incremental: IncrementalConfig
    verbose: bool

class DedupConfig(TypedDict, total=False):
    backend: Backend
    k: int
    sim_threshold: float
    rep_threshold: float | None
    max_gap_secs: float | None

@final
class Hierarchy:
    """Two-level clustering that labels new rows and updates itself."""

    @property
    def d(self) -> int: ...
    @property
    def n_seen(self) -> int: ...
    @property
    def chosen_k(self) -> tuple[int, list[int]]: ...
    def add(self, data_all: Buffer, /) -> AddReport:
        """Label rows after the first `n_seen` of `data_all` (all rows seen so far, then the new ones)."""
    def save(self, path: str | os.PathLike[str]) -> None: ...
    @staticmethod
    def load(path: str | os.PathLike[str], backend: Backend = "cpu") -> Hierarchy: ...

def train(data: Buffer, k: int, *, d: int | None = None, **config: Unpack[TrainConfig]) -> TrainResult:
    """Train e-means with k centroids on data (n x d float32). d comes from a 2-D buffer."""

def assign(
    data: Buffer,
    centroids: Buffer,
    *,
    d: int | None = None,
    metric: Literal["l2", "inner_product", "ip", "cosine"] = "l2",
    backend: Backend = "cpu",
) -> Assignment: ...
def auto_cluster(
    data: Buffer, *, d: int | None = None, **config: Unpack[AutoConfig]
) -> tuple[Hierarchy, Labels]: ...
def dedup(
    data: Buffer,
    *,
    d: int | None = None,
    created: Buffer | list[float] | None = None,
    burst: Buffer | list[int] | None = None,
    priority: Buffer | list[float] | None = None,
    **config: Unpack[DedupConfig],
) -> DedupResult: ...
def default_config(kind: Literal["train", "auto", "dedup"]) -> dict[str, object]: ...
def backends() -> list[str]: ...
def _buffer_address(data: Buffer) -> int: ...
