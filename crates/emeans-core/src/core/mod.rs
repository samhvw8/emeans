//! The core layer: build each block once, run it on every backend.
//!
//! - Layer 1, [`Device`]: the primitive op contract every backend implements
//!   (fused top-1 assignment, top-k, scatter-add of rows into per-cluster sums,
//!   row norms and normalization, the EMA update), on buffers that live on the
//!   backend. [`reference`] holds one f64 CPU implementation of every op;
//!   [`contract`] checks a backend against it. Adding a backend = implementing
//!   [`Device`] and passing [`contract::check`].
//! - Layer 2, [`train`]: the e-means loop written once against layer 1. Data,
//!   centroids, batch sums and counts stay on the device; the host syncs only for
//!   a host decision (revival every `CHECK_UNUSED_EVERY` updates, epoch ends).
//!
//! See `.okf/design/core.md`.

pub mod contract;
pub mod cpu;
pub mod reference;
pub mod train;

use std::ops::Range;

use anyhow::{Result, ensure};

use crate::assign::{Metric, Rows};
use crate::rng::RandomState;

/// An `n x d` matrix as a kernel input: the device copy, and the host slice it
/// came from. The CPU backend reads `host` (its `dev` is empty); GPU backends
/// read `dev`.
pub struct Mat<'a, F> {
    pub host: &'a [f32],
    pub dev: &'a F,
    pub d: usize,
}

impl<F> Clone for Mat<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<F> Copy for Mat<'_, F> {}

/// Rows `rows` of a selection: `x[ids[r]]` when `ids` is given, else `x[r]`.
pub struct Select<'a, U> {
    pub ids: Option<&'a U>,
    pub rows: Range<usize>,
}

impl<U> Clone for Select<'_, U> {
    fn clone(&self) -> Self {
        Select {
            ids: self.ids,
            rows: self.rows.clone(),
        }
    }
}

/// Layer 1: the ops a backend implements. Buffers are the backend's own types.
///
/// Exactness contract (checked by [`contract::check`]):
///
/// | Op | Must match |
/// |---|---|
/// | `top1` | f64 reference label, except near-ties (reference scores within `TIE_EPS`); score within `SCORE_EPS` |
/// | `topk` | same, per rank |
/// | `scatter_add` | counts exact; sums bit-equal to adding the rows in row order in f32 |
/// | `ema_update`, `split` | bit-equal to the CPU backend (same f32 operations, no FMA contraction) |
/// | `row_norms`, `normalize_rows` | equal to the CPU backend (sequential f32 sums, IEEE divide and sqrt); f64 reference within `NORM_EPS` |
///
/// Ops are queued; only `read_*` (and `sync`) wait for the device.
pub trait Device {
    /// f32 buffer on the device.
    type F;
    /// u32 buffer on the device.
    type U;

    fn name(&self) -> String;

    /// Put the data set `x` (`n x d`) on the device. The CPU backend returns an
    /// empty buffer and reads the host slice instead.
    fn bind(&mut self, x: &[f32]) -> Result<Self::F>;
    fn upload_f32(&mut self, v: &[f32]) -> Result<Self::F>;
    fn upload_u32(&mut self, v: &[u32]) -> Result<Self::U>;
    fn zeros_f32(&mut self, n: usize) -> Result<Self::F>;
    fn zeros_u32(&mut self, n: usize) -> Result<Self::U>;
    /// Overwrite `b[offset..offset + v.len()]`, after queued work that uses `b`.
    fn write_f32(&mut self, b: &mut Self::F, offset: usize, v: &[f32]) -> Result<()>;
    fn write_u32(&mut self, b: &mut Self::U, offset: usize, v: &[u32]) -> Result<()>;
    /// Wait for queued work and copy `b[range]` to the host.
    fn read_f32(&mut self, b: &Self::F, range: Range<usize>) -> Result<Vec<f32>>;
    fn read_u32(&mut self, b: &Self::U, range: Range<usize>) -> Result<Vec<u32>>;
    /// Submit queued work without waiting.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Fused score + argmax: for every selected row `r`, `labels[r]` = the best
    /// of the `k` centroids and `scores[r]` its score (inner product, or squared
    /// L2 distance). Ties keep the lower id. `cnorm` (squared centroid norms) is
    /// required for L2. Outputs are written at the selection's positions `rows`.
    #[allow(clippy::too_many_arguments)]
    fn top1(
        &mut self,
        x: Mat<'_, Self::F>,
        sel: Select<'_, Self::U>,
        centroids: &Self::F,
        cnorm: Option<&Self::F>,
        k: usize,
        metric: Metric,
        labels: &mut Self::U,
        scores: &mut Self::F,
    ) -> Result<()>;

    /// Best `kk` rows of `data` for every row of `queries` by inner product,
    /// best first (larger score, then lower id). With `exclude_self`, query `i`
    /// is data row `i` and is skipped. Missing neighbours: id `u32::MAX`, score -inf.
    fn topk(
        &mut self,
        queries: &[f32],
        data: &[f32],
        d: usize,
        kk: usize,
        exclude_self: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)>;

    /// `sums` (`k x d`) = the selected rows added per label in row order,
    /// `counts` (`k`) = rows per label; both are overwritten. With `acc`,
    /// also `acc += sums` (the closing pass's segmented sums).
    #[allow(clippy::too_many_arguments)]
    fn scatter_add(
        &mut self,
        x: Mat<'_, Self::F>,
        sel: Select<'_, Self::U>,
        labels: &Self::U,
        k: usize,
        sums: &mut Self::F,
        counts: &mut Self::U,
        acc: Option<&mut Self::F>,
    ) -> Result<()>;

    /// Squared norm of each of the `rows` rows (`rows x d`) of `m`.
    fn row_norms(&mut self, m: &Self::F, rows: usize, d: usize, out: &mut Self::F) -> Result<()>;

    /// faiss `normalize_into`: divide each row by its max magnitude, then by its norm.
    fn normalize_rows(
        &mut self,
        src: &Self::F,
        rows: usize,
        d: usize,
        dst: &mut Self::F,
    ) -> Result<()>;

    /// Apply a revival plan (see [`plan_revival`]): for each (donor, dead)
    /// pair, donor and child get `c (1 -/+ 1e-4 s)` and half the donor's mass.
    /// Bit-equal to [`apply_splits`].
    fn split(
        &mut self,
        positions: &mut Self::F,
        usage: &mut Self::F,
        plan: &[(usize, usize)],
        d: usize,
    ) -> Result<()>;

    /// The EMA step for one batch (faiss `update`): masses decay by `decay`
    /// (unless `replace`); each cluster with rows gets `N += alpha n_j` and
    /// `c += (alpha n_j / N)(s_j / n_j - c)`.
    #[allow(clippy::too_many_arguments)]
    fn ema_update(
        &mut self,
        positions: &mut Self::F,
        usage: &mut Self::F,
        sums: &Self::F,
        counts: &Self::U,
        k: usize,
        d: usize,
        step: EmaStep,
    ) -> Result<()>;
}

/// Scalars of one EMA step, computed on the host in f64 and cast like numpy.
#[derive(Clone, Copy, Debug)]
pub struct EmaStep {
    pub alpha: f32,
    pub decay: f32,
    /// alpha == 1 exactly: masses are replaced, not decayed.
    pub replace: bool,
}

impl EmaStep {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha as f32,
            decay: (1.0 - alpha) as f32,
            replace: alpha == 1.0,
        }
    }
}

/// A backend that can run the device loop: the object-safe face of a [`Device`],
/// handed to [`crate::Assigner::with_engine`]'s callback.
///
/// Every [`Device`] is an `Engine` (blanket impl), so a backend crate gets the
/// training loop and top-k by implementing [`Device`] and returning
/// `Some(f(self))` from `with_engine`. The generic loop is compiled once per
/// backend, in the crate that coerces its device to `&mut dyn Engine`.
pub trait Engine {
    /// The device's name, as [`Device::name`].
    fn engine_name(&self) -> String;

    /// Layer 2: one training run ([`train::run`]) on this device.
    fn run_train(&mut self, job: train::Job<'_>) -> Result<train::Outcome>;

    /// [`Device::topk`].
    fn run_topk(
        &mut self,
        queries: &[f32],
        data: &[f32],
        d: usize,
        kk: usize,
        exclude_self: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)>;
}

impl<D: Device> Engine for D {
    fn engine_name(&self) -> String {
        Device::name(self)
    }

    fn run_train(&mut self, job: train::Job<'_>) -> Result<train::Outcome> {
        train::run(self, job)
    }

    fn run_topk(
        &mut self,
        queries: &[f32],
        data: &[f32],
        d: usize,
        kk: usize,
        exclude_self: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        Device::topk(self, queries, data, d, kk, exclude_self)
    }
}

/// `Assigner::assign` on any [`Device`]: upload, `top1`, read back. For backend
/// crates; not a stable API.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn assign_on<D: Device>(
    dev: &mut D,
    bound: Option<(*const f32, usize, &D::F)>,
    rows: Rows<'_>,
    d: usize,
    centroids: &[f32],
    metric: Metric,
    labels: &mut [u32],
    scores: &mut [f32],
) -> Result<()> {
    let k = crate::assign::check_shapes(&rows, d, centroids, labels, scores)?;
    let m = rows.len(d);
    if m == 0 {
        return Ok(());
    }
    let c = dev.upload_f32(centroids)?;
    let cn = match metric {
        Metric::L2 => {
            let mut o = dev.zeros_f32(k)?;
            dev.row_norms(&c, k, d, &mut o)?;
            Some(o)
        }
        Metric::InnerProduct => None,
    };
    let mut lab = dev.zeros_u32(m)?;
    let mut sco = dev.zeros_f32(m)?;
    match rows {
        Rows::Indexed { data, ids }
            if bound.is_some_and(|(p, l, _)| p == data.as_ptr() && l == data.len()) =>
        {
            let dv = bound.unwrap().2;
            let idb = dev.upload_u32(ids)?;
            dev.top1(
                Mat {
                    host: data,
                    dev: dv,
                    d,
                },
                Select {
                    ids: Some(&idb),
                    rows: 0..m,
                },
                &c,
                cn.as_ref(),
                k,
                metric,
                &mut lab,
                &mut sco,
            )?;
        }
        _ => {
            let mut g;
            let host: &[f32] = match rows {
                Rows::Dense(x) => x,
                Rows::Indexed { .. } => {
                    g = Vec::with_capacity(m * d);
                    for i in 0..m {
                        g.extend_from_slice(rows.row(i, d));
                    }
                    &g
                }
            };
            let xb = dev.bind(host)?;
            dev.top1(
                Mat { host, dev: &xb, d },
                Select {
                    ids: None,
                    rows: 0..m,
                },
                &c,
                cn.as_ref(),
                k,
                metric,
                &mut lab,
                &mut sco,
            )?;
        }
    }
    labels.copy_from_slice(&dev.read_u32(&lab, 0..m)?);
    scores.copy_from_slice(&dev.read_f32(&sco, 0..m)?);
    Ok(())
}

// ------------------------------------------------------------- host blocks

/// Revival plan (Algorithm 1, line 9; Appendix J): (donor, dead) pairs.
///
/// Clusters below `tau * mean(N)` (numpy float32 pairwise sum) take a donor
/// drawn by Gumbel top-k on `log N`. Draws `k` uniforms from `rng` only when
/// at least one split happens, as faiss does. Equal keys: higher index first.
pub fn plan_revival(usage: &[f32], tau: f64, rng: &mut RandomState) -> Vec<(usize, usize)> {
    let k = usage.len();
    let total = crate::model::numpy_sum_f32(usage);
    let threshold = (total * tau as f32) / k as f32;
    let unused: Vec<bool> = usage.iter().map(|&u| u < threshold).collect();
    let n_unused = unused.iter().filter(|&&u| u).count();
    let donors: Vec<bool> = usage
        .iter()
        .zip(&unused)
        .map(|(&u, &dead)| !dead && u > 0.0)
        .collect();
    let n_donors = donors.iter().filter(|&&x| x).count();
    let nsplit = n_unused.min(n_donors);
    if nsplit == 0 {
        return Vec::new();
    }
    let uniform: Vec<f64> = (0..k)
        .map(|_| rng.random_sample().max(f64::MIN_POSITIVE))
        .collect();
    let mut keyed: Vec<(f64, usize)> = (0..k)
        .filter(|&j| donors[j])
        .map(|j| ((usage[j].ln() as f64) - (-(uniform[j].ln())).ln(), j))
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
    let target = (0..k).filter(|&j| unused[j]);
    keyed
        .iter()
        .take(nsplit)
        .map(|&(_, j)| j)
        .zip(target)
        .collect()
}

/// Apply a revival plan: donor and child get `c (1 -/+ 1e-4 s)` with an
/// alternating sign `s`, and half the donor's mass each.
pub fn apply_splits(positions: &mut [f32], usage: &mut [f32], d: usize, plan: &[(usize, usize)]) {
    let eps = crate::recipe::SPLIT_EPSILON as f32;
    for &(s, t) in plan {
        let c = positions[s * d..(s + 1) * d].to_vec();
        let mut donor = vec![0f32; d];
        let mut child = vec![0f32; d];
        for i in 0..d {
            let off = eps * if i % 2 == 0 { 1.0 } else { -1.0 };
            donor[i] = c[i] * (1.0 - off);
            child[i] = c[i] * (1.0 + off);
        }
        if donor == child {
            let up = child[0].next_up();
            child[0] = if up.is_finite() {
                up
            } else {
                child[0].next_down()
            };
        }
        let half = usage[s] / 2.0;
        usage[s] = half;
        positions[s * d..(s + 1) * d].copy_from_slice(&donor);
        usage[t] = half;
        positions[t * d..(t + 1) * d].copy_from_slice(&child);
    }
}

/// Closing means (Algorithm 1, line 12): last-epoch sums / counts; clusters
/// with no last-epoch rows keep their EMA position; spherical means are
/// projected onto the unit sphere.
pub fn closing_means(
    positions: &[f32],
    sums: &[f32],
    counts: &[f64],
    d: usize,
    spherical: bool,
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = positions.to_vec();
    out.par_chunks_mut(d)
        .zip(sums.par_chunks(d))
        .zip(counts.par_iter())
        .for_each(|((c, s), &n)| {
            if n > 0.0 {
                // Euclidean: sums / counts.astype(f32); spherical: sums / max(counts, 1).
                let nf = n as f32;
                for (ci, &si) in c.iter_mut().zip(s) {
                    *ci = si / nf;
                }
            }
            if spherical {
                crate::model::normalize_rows(c, d);
            }
        });
    out
}

#[doc(hidden)]
pub fn check_select<U>(sel: &Select<'_, U>, ids_len: Option<usize>, n: usize) -> Result<()> {
    ensure!(sel.rows.start <= sel.rows.end, "bad row range");
    match ids_len {
        Some(l) => ensure!(sel.rows.end <= l, "row range past the ids"),
        None => ensure!(sel.rows.end <= n, "row range past the data"),
    }
    Ok(())
}
