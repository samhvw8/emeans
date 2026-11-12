//! e-means core: mini-batch k-means with scaling laws, ported from faiss
//! `contrib/e_means` (arXiv 2611.09501).
//!
//! This crate holds everything a backend needs and nothing heavy (`rayon`,
//! `serde`, `anyhow`; `gemm` off macOS):
//!
//! - [`Emeans`]: `train`, the streaming API (`begin` / `accumulate` /
//!   `update` / `end_epoch` / `finalize`), `assign`, `save` / `load`, and
//!   continued training from saved EMA state (`resume`, `train_resume`).
//! - [`assign`]: the [`Assigner`] trait and the CPU kernels; [`assign::gemm_nt`].
//! - [`crate::core`]: the core layer. Layer 1 is the op contract every backend
//!   implements ([`core::Device`]); [`core::reference`] is its f64 reference and
//!   [`core::contract`] checks a backend against it. Layer 2 is the
//!   device-resident training loop ([`core::train`]) that [`Emeans::train`] runs
//!   through [`core::Engine`].
//! - [`recipe`]: batch and step laws, epoch plan, cosine schedule.
//! - [`rng`]: numpy legacy `RandomState`, for runs that match faiss draw for draw.
//! - [`knn`]: the CPU exact top-k search and [`Neighbors`].
//! - [`io`]: save and load a model.
//!
//! Backends live in `emeans-metal` and `emeans-cuda`; the algorithms on top
//! (automatic k, dedup, incremental) in `emeans-cluster`; the `emeans` crate
//! is the facade over all of them.

pub mod assign;
pub mod core;
pub mod io;
pub mod knn;
pub mod model;
pub mod recipe;
pub mod rng;

pub use assign::{Assigner, CpuAssigner, CpuKernel, Metric, Rows, SharedAssigner, SharedData};
pub use knn::Neighbors;
pub use model::{Accumulated, Assignment, Config, Emeans, EpochStats, Trained};
pub use recipe::{BatchLaw, Recipe};
