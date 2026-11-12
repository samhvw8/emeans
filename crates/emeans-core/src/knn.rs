//! Exact k-nearest-neighbour search by inner product (cosine for unit rows): the
//! CPU path and the result type. GPU backends implement the same search as
//! [`crate::core::Device::topk`]; `emeans-cluster` picks the backend.
//!
//! At n = 16k the `n^2` dot products (d = 768) cost about 0.4 TFLOP, so exact
//! search is affordable and no approximate index is needed. The CPU path
//! multiplies row blocks with [`gemm_nt`] and selects the top k per row with
//! rayon. Every backend orders each row best first: larger score, then lower id.

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::assign::gemm_nt;

/// Largest `k` the GPU top-k kernels keep; larger `k` runs on the CPU.
pub const GPU_KMAX: usize = 16;

/// Padding id for rows with fewer than `k` candidates.
pub const NONE: u32 = u32::MAX;

/// The `k` nearest data rows of every query row, best first.
#[derive(Clone, Debug)]
pub struct Neighbors {
    pub k: usize,
    /// `n x k` data-row ids; `NONE` for padding.
    pub ids: Vec<u32>,
    /// `n x k` inner products; `-inf` for padding.
    pub sims: Vec<f32>,
}

impl Neighbors {
    pub fn len(&self) -> usize {
        self.ids.len() / self.k.max(1)
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Ids and similarities of row `i`'s neighbours.
    pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let r = i * self.k..(i + 1) * self.k;
        (&self.ids[r.clone()], &self.sims[r])
    }
}

#[inline]
fn better(a: f32, ja: u32, b: f32, jb: u32) -> bool {
    a > b || (a == b && ja < jb)
}

/// CPU search: Accelerate GEMM on row blocks, then a per-row top-k.
pub fn knn_cpu(
    queries: &[f32],
    data: &[f32],
    d: usize,
    k: usize,
    exclude_self: bool,
) -> Result<Neighbors> {
    let (m, n) = (queries.len() / d, data.len() / d);
    ensure!(
        !exclude_self || m == n,
        "exclude_self needs queries == data"
    );
    let mut ids = vec![NONE; m * k];
    let mut sims = vec![f32::NEG_INFINITY; m * k];
    // About 32 MB of dot products per block.
    let block = ((8 << 20) / n.max(1)).clamp(16, 4096);
    let mut dots = Vec::new();
    for b0 in (0..m).step_by(block) {
        let b1 = (b0 + block).min(m);
        dots.resize((b1 - b0) * n, 0f32);
        gemm_nt(&queries[b0 * d..b1 * d], data, d, &mut dots)?;
        ids[b0 * k..b1 * k]
            .par_chunks_mut(k)
            .zip(sims[b0 * k..b1 * k].par_chunks_mut(k))
            .enumerate()
            .for_each(|(r, (id, sc))| {
                let i = b0 + r;
                let row = &dots[r * n..(r + 1) * n];
                let mut w = 0usize;
                for (j, &v) in row.iter().enumerate() {
                    if exclude_self && j == i {
                        continue;
                    }
                    if v > sc[w] {
                        sc[w] = v;
                        id[w] = j as u32;
                        w = 0;
                        for t in 1..k {
                            if better(sc[w], id[w], sc[t], id[t]) {
                                w = t;
                            }
                        }
                    }
                }
                let mut pairs: Vec<(f32, u32)> =
                    sc.iter().copied().zip(id.iter().copied()).collect();
                pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                for (t, (s, j)) in pairs.into_iter().enumerate() {
                    sc[t] = s;
                    id[t] = j;
                }
            });
    }
    Ok(Neighbors { k, ids, sims })
}
