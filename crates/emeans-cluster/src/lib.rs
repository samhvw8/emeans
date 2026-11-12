//! Algorithms on top of e-means (`emeans-core`), on any backend.
//!
//! - [`Backend`], [`make_assigner`]: pick and build a backend (CPU, Metal on
//!   macOS, CUDA with the `cuda` feature, or [`Backend::Custom`]).
//! - [`auto`]: two-level clustering with automatic k ([`auto_cluster`], [`Hierarchy`]).
//! - [`incremental`]: label new rows, EMA updates, retrains, stable ids, save/load.
//! - [`dedup`]: near-duplicate groups (bursts, near-identical shots).
//! - [`knn`]: exact top-k neighbour search on the chosen backend.
//! - [`metrics`]: ARI, NMI, cosine silhouette, Calinski-Harabasz, Davies-Bouldin.
//! - [`synth`]: synthetic nested mixtures and bursts for validation.

pub mod auto;
pub mod backend;
pub mod dedup;
pub mod incremental;
pub mod knn;
pub mod metrics;
pub mod synth;

// Core items the modules use as `crate::...`.
#[allow(unused_imports)]
use emeans_core::{Assigner, CpuAssigner, CpuKernel, Metric, Rows, assign, io, model, recipe, rng};

pub use auto::{AutoConfig, ChosenK, Criterion, Hierarchy, Init, Labels, auto_cluster};
pub use backend::{Backend, CustomBackend, make_assigner};
pub use dedup::{DedupConfig, DedupInput, DedupResult, dedup};
pub use incremental::{Action, AddReport, IncrementalConfig, PhotoAddReport};
pub use knn::{knn, knn_self};
