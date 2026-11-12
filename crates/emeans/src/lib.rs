//! e-means: mini-batch k-means with scaling laws, ported from faiss
//! `contrib/e_means` (arXiv 2611.09501), on CPU, Metal and CUDA.
//!
//! This crate is the facade over the `emeans-*` crates. It re-exports their
//! public API, so most users depend on `emeans` alone:
//!
//! - [`Emeans`]: train, assign, save / load, continue training (`emeans-core`).
//! - [`auto_cluster`] and [`Hierarchy`]: two-level clustering with automatic k,
//!   incremental `add`, save / load (`emeans-cluster`).
//! - [`dedup()`]: near-duplicate groups; [`knn`] / [`knn_self`]: exact top-k.
//! - [`Backend`] / [`make_assigner`]: CPU kernels, Metal (every macOS build),
//!   CUDA (feature `cuda`), or [`Backend::Custom`] for a backend of your own.
//!
//! Inputs are plain row-major `f32` slices (`n x d`), outputs plain structs and
//! vectors; nothing in the API is generic, so language bindings can wrap it
//! as is.
//!
//! ```
//! use emeans::{Backend, Config, Emeans, make_assigner};
//! use emeans::synth::{NestedSpec, nested};
//!
//! let (n, d, k) = (2000, 16, 8);
//! let x = nested(&NestedSpec { n, d, ..Default::default() }).x;
//! let mut cfg = Config::new(d, k);
//! cfg.spherical = true;
//! let mut model = Emeans::with_assigner(cfg, make_assigner(Backend::cpu())?)?;
//! let trained = model.train(&x)?;
//! assert_eq!(trained.centroids.len(), k * d);
//! assert_eq!(model.assign(&x)?.labels.len(), n);
//! # Ok::<(), anyhow::Error>(())
//! ```

pub use emeans_core::{
    Accumulated, Assigner, Assignment, BatchLaw, Config, CpuAssigner, CpuKernel, Emeans,
    EpochStats, Metric, Neighbors, Recipe, Rows, SharedAssigner, SharedData, Trained, assign, core,
    io, model, recipe, rng,
};

pub use emeans_cluster::{
    Action, AddReport, AutoConfig, Backend, ChosenK, Criterion, CustomBackend, DedupConfig,
    DedupInput, DedupResult, Hierarchy, IncrementalConfig, Init, Labels, PhotoAddReport, auto,
    auto_cluster, backend, dedup, incremental, knn, knn_self, make_assigner, metrics, synth,
};

/// The Metal engine (macOS).
#[cfg(target_os = "macos")]
pub use emeans_metal as metal;

/// The CUDA engine (feature `cuda`).
#[cfg(feature = "cuda")]
pub use emeans_cuda as cuda;
