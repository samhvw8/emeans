//! Exact k-nearest-neighbour search by inner product (cosine for unit rows),
//! on the backend you pick.
//!
//! At n = 16k the `n^2` dot products (d = 768) cost about 0.4 TFLOP, so exact
//! search is affordable and no approximate index is needed. GPU engines run a
//! fused GEMM + top-k kernel (`Device::topk`, `k <= GPU_KMAX`); the CPU path
//! ([`knn_cpu`]) multiplies row blocks with `gemm_nt` and selects the top k per
//! row with rayon. Every backend orders each row best first: larger score,
//! then lower id.

use anyhow::{Context, Result, ensure};

use crate::Backend;
pub use emeans_core::knn::{GPU_KMAX, NONE, Neighbors, knn_cpu};

/// Neighbours of every row of `data` among the other rows (self excluded).
pub fn knn_self(data: &[f32], d: usize, k: usize, backend: Backend) -> Result<Neighbors> {
    search(data, data, d, k, true, backend)
}

/// Neighbours of every row of `queries` among the rows of `data`.
pub fn knn(
    queries: &[f32],
    data: &[f32],
    d: usize,
    k: usize,
    backend: Backend,
) -> Result<Neighbors> {
    search(queries, data, d, k, false, backend)
}

fn search(
    queries: &[f32],
    data: &[f32],
    d: usize,
    k: usize,
    exclude_self: bool,
    backend: Backend,
) -> Result<Neighbors> {
    ensure!(d > 0 && queries.len().is_multiple_of(d) && data.len().is_multiple_of(d));
    ensure!(k > 0, "k must be positive");
    if matches!(backend, Backend::Cpu(_)) || k > GPU_KMAX {
        return knn_cpu(queries, data, d, k, exclude_self);
    }
    // GPU backends: the engine's top-k op (layer 1). BF16 is for top-1 only;
    // its exact FP32 engine runs the search.
    let engine = match backend {
        Backend::CudaBf16 => Backend::Cuda,
        b => b,
    };
    let mut a = crate::make_assigner(engine)?;
    let mut out = None;
    let ran = a.with_engine(&mut |e| {
        out = Some(e.run_topk(queries, data, d, k, exclude_self)?);
        Ok(())
    });
    match ran {
        Some(r) => r?,
        // A custom backend without an engine: the CPU search.
        None => return knn_cpu(queries, data, d, k, exclude_self),
    }
    let (ids, sims) = out.context("top-k did not run")?;
    Ok(Neighbors { k, ids, sims })
}
