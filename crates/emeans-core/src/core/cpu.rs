//! The CPU backend of the op contract: host `Vec`s as buffers, the existing
//! CPU assignment kernels for `top1`, and the reference float order for
//! everything else (these functions are also the streaming API's math).

use std::ops::Range;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use super::{Device, EmaStep, Mat, Select, check_select};
use crate::assign::{Assigner, CpuAssigner, Metric, Rows, on_cpu_pool};

fn host<'a>(m: &Mat<'a, Vec<f32>>) -> &'a [f32] {
    if m.dev.is_empty() { m.host } else { m.dev }
}

fn rows_of<'a>(x: &'a [f32], d: usize, sel: &Select<'a, Vec<u32>>) -> Rows<'a> {
    match sel.ids {
        Some(ids) => Rows::Indexed {
            data: x,
            ids: &ids[sel.rows.clone()],
        },
        None => Rows::Dense(&x[sel.rows.start * d..sel.rows.end * d]),
    }
}

impl Device for CpuAssigner {
    type F = Vec<f32>;
    type U = Vec<u32>;

    fn name(&self) -> String {
        Assigner::name(self)
    }

    fn bind(&mut self, _x: &[f32]) -> Result<Vec<f32>> {
        Ok(Vec::new())
    }

    fn upload_f32(&mut self, v: &[f32]) -> Result<Vec<f32>> {
        Ok(v.to_vec())
    }

    fn upload_u32(&mut self, v: &[u32]) -> Result<Vec<u32>> {
        Ok(v.to_vec())
    }

    fn zeros_f32(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(vec![0.0; n])
    }

    fn zeros_u32(&mut self, n: usize) -> Result<Vec<u32>> {
        Ok(vec![0; n])
    }

    fn write_f32(&mut self, b: &mut Vec<f32>, offset: usize, v: &[f32]) -> Result<()> {
        ensure!(offset + v.len() <= b.len(), "write past the buffer");
        b[offset..offset + v.len()].copy_from_slice(v);
        Ok(())
    }

    fn write_u32(&mut self, b: &mut Vec<u32>, offset: usize, v: &[u32]) -> Result<()> {
        ensure!(offset + v.len() <= b.len(), "write past the buffer");
        b[offset..offset + v.len()].copy_from_slice(v);
        Ok(())
    }

    fn read_f32(&mut self, b: &Vec<f32>, range: Range<usize>) -> Result<Vec<f32>> {
        ensure!(range.end <= b.len(), "read past the buffer");
        Ok(b[range].to_vec())
    }

    fn read_u32(&mut self, b: &Vec<u32>, range: Range<usize>) -> Result<Vec<u32>> {
        ensure!(range.end <= b.len(), "read past the buffer");
        Ok(b[range].to_vec())
    }

    fn top1(
        &mut self,
        x: Mat<'_, Vec<f32>>,
        sel: Select<'_, Vec<u32>>,
        centroids: &Vec<f32>,
        _cnorm: Option<&Vec<f32>>,
        _k: usize,
        metric: Metric,
        labels: &mut Vec<u32>,
        scores: &mut Vec<f32>,
    ) -> Result<()> {
        let data = host(&x);
        check_select(&sel, sel.ids.map(|i| i.len()), data.len() / x.d)?;
        let r = sel.rows.clone();
        ensure!(
            r.end <= labels.len() && r.end <= scores.len(),
            "outputs too short"
        );
        let rows = rows_of(data, x.d, &sel);
        self.assign(
            rows,
            x.d,
            centroids,
            metric,
            &mut labels[r.clone()],
            &mut scores[r],
        )
    }

    fn topk(
        &mut self,
        queries: &[f32],
        data: &[f32],
        d: usize,
        kk: usize,
        exclude_self: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let nb = on_cpu_pool(|| crate::knn::knn_cpu(queries, data, d, kk, exclude_self))?;
        Ok((nb.ids, nb.sims))
    }

    fn scatter_add(
        &mut self,
        x: Mat<'_, Vec<f32>>,
        sel: Select<'_, Vec<u32>>,
        labels: &Vec<u32>,
        k: usize,
        sums: &mut Vec<f32>,
        counts: &mut Vec<u32>,
        acc: Option<&mut Vec<f32>>,
    ) -> Result<()> {
        let data = host(&x);
        let d = x.d;
        check_select(&sel, sel.ids.map(|i| i.len()), data.len() / d)?;
        ensure!(
            sums.len() == k * d && counts.len() == k,
            "sums k x d, counts k"
        );
        counts.fill(0);
        let rows = rows_of(data, d, &sel);
        if let Some(a) = acc.as_deref() {
            ensure!(a.len() == k * d, "acc must be k x d");
        }
        // Zero, add the rows (row order) and add into `acc` in one pass per centroid.
        let labels = &labels[sel.rows.clone()];
        on_cpu_pool(|| scatter_rows(sums, counts, rows, labels, k, d, acc.map(|a| &mut a[..])))
    }

    fn row_norms(&mut self, m: &Vec<f32>, rows: usize, d: usize, out: &mut Vec<f32>) -> Result<()> {
        ensure!(m.len() >= rows * d && out.len() >= rows, "shapes");
        let norms = on_cpu_pool(|| crate::assign::row_norms(&m[..rows * d], d));
        out[..rows].copy_from_slice(&norms);
        Ok(())
    }

    fn normalize_rows(
        &mut self,
        src: &Vec<f32>,
        rows: usize,
        d: usize,
        dst: &mut Vec<f32>,
    ) -> Result<()> {
        ensure!(src.len() >= rows * d && dst.len() >= rows * d, "shapes");
        on_cpu_pool(|| {
            dst[..rows * d]
                .par_chunks_mut(d)
                .zip(src[..rows * d].par_chunks(d))
                .with_min_len(small_task_rows(d))
                .for_each(|(r, s)| {
                    r.copy_from_slice(s);
                    normalize_row(r);
                })
        });
        Ok(())
    }

    fn split(
        &mut self,
        positions: &mut Vec<f32>,
        usage: &mut Vec<f32>,
        plan: &[(usize, usize)],
        d: usize,
    ) -> Result<()> {
        let k = usage.len();
        ensure!(positions.len() == k * d, "positions must be k x d");
        ensure!(
            plan.iter().all(|&(s, t)| s < k && t < k),
            "split ids out of range"
        );
        super::apply_splits(positions, usage, d, plan);
        Ok(())
    }

    fn ema_update(
        &mut self,
        positions: &mut Vec<f32>,
        usage: &mut Vec<f32>,
        sums: &Vec<f32>,
        counts: &Vec<u32>,
        k: usize,
        d: usize,
        step: EmaStep,
    ) -> Result<()> {
        ensure!(
            positions.len() == k * d
                && usage.len() == k
                && sums.len() == k * d
                && counts.len() == k,
            "shapes"
        );
        ema_host(positions, usage, sums, counts, d, step);
        Ok(())
    }
}

/// Rows per rayon task in the small per-batch ops (normalize, EMA, scatter):
/// about 16k floats, so a 1024 x 768 op wakes ~50 threads, not one per row.
pub(crate) fn small_task_rows(d: usize) -> usize {
    (16384 / d.max(1)).max(1)
}

/// faiss `normalize_into` for one row, in [`crate::model::normalize_rows`]'s float order.
fn normalize_row(r: &mut [f32]) {
    let max = r.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let s = if max > 0.0 { max } else { 1.0 };
    for v in r.iter_mut() {
        *v /= s;
    }
    let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
    let s = if norm > 0.0 { norm } else { 1.0 };
    for v in r.iter_mut() {
        *v /= s;
    }
}

/// Counting sort of `labels`: (start offsets `k + 1`, row order grouped by label).
fn group_by_label(labels: &[u32], k: usize) -> Result<(Vec<usize>, Vec<usize>)> {
    ensure!(
        labels.iter().all(|&l| (l as usize) < k),
        "label out of range"
    );
    let mut start = vec![0usize; k + 1];
    for &l in labels {
        start[l as usize + 1] += 1;
    }
    for j in 0..k {
        start[j + 1] += start[j];
    }
    let mut fill = start.clone();
    let mut order = vec![0usize; labels.len()];
    for (i, &l) in labels.iter().enumerate() {
        order[fill[l as usize]] = i;
        fill[l as usize] += 1;
    }
    Ok((start, order))
}

/// Add the rows in `mine` (row order) to `part`, columns `c0..` of each row.
#[inline]
fn add_group(part: &mut [f32], rows: Rows<'_>, mine: &[usize], c0: usize, d: usize) {
    let w = part.len();
    for (n, &i) in mine.iter().enumerate() {
        if let Some(&next) = mine.get(n + PREFETCH_AHEAD) {
            prefetch(&rows.row(next, d)[c0..c0 + w]);
        }
        let row = &rows.row(i, d)[c0..c0 + w];
        for (a, &b) in part.iter_mut().zip(row) {
            *a += b;
        }
    }
}

/// Rows ahead that `add_group` asks the memory system for (rows are scattered).
const PREFETCH_AHEAD: usize = 2;

/// Hint the cache lines of `v` into L1 (x86-64; a no-op elsewhere).
#[inline(always)]
fn prefetch(v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    for line in v.chunks(16) {
        // SAFETY: a prefetch is a hint; the address is inside `v`.
        unsafe {
            std::arch::x86_64::_mm_prefetch(line.as_ptr().cast(), std::arch::x86_64::_MM_HINT_T0)
        };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = v;
}

/// Column width of the scatter tasks: with few centroids, split each sum's
/// columns so that every thread has work. Each sum still adds its rows in row
/// order, so the bits do not change.
fn column_width(k: usize, d: usize) -> usize {
    let parts = (4 * rayon::current_num_threads())
        .div_ceil(k)
        .min(d.div_ceil(16))
        .max(1);
    d.div_ceil(parts).next_multiple_of(16).min(d)
}

/// Add rows to per-centroid sums in row order, and their counts.
///
/// Parallel over centroids, sequential within one: the same additions, in the
/// same order, as numpy's `np.add.at` / the scipy product in faiss.
pub(crate) fn add_rows(
    sums: &mut [f32],
    counts: &mut [u32],
    rows: Rows<'_>,
    labels: &[u32],
    k: usize,
    d: usize,
) -> Result<()> {
    on_cpu_pool(|| add_rows_here(sums, counts, rows, labels, k, d))
}

fn add_rows_here(
    sums: &mut [f32],
    counts: &mut [u32],
    rows: Rows<'_>,
    labels: &[u32],
    k: usize,
    d: usize,
) -> Result<()> {
    let (start, order) = group_by_label(labels, k)?;
    for j in 0..k {
        counts[j] += (start[j + 1] - start[j]) as u32;
    }
    let width = column_width(k, d);
    sums.par_chunks_mut(d).enumerate().for_each(|(j, s)| {
        let mine = &order[start[j]..start[j + 1]];
        s.par_chunks_mut(width)
            .enumerate()
            .for_each(|(q, part)| add_group(part, rows, mine, q * width, d));
    });
    Ok(())
}

/// The device `scatter_add` on the host: per centroid, zero its sums, add its
/// rows in row order, then add the sums into `acc`. Same bits as `fill(0)`,
/// [`add_rows`], then `acc += sums`; one pass and one thread wake-up.
fn scatter_rows(
    sums: &mut [f32],
    counts: &mut [u32],
    rows: Rows<'_>,
    labels: &[u32],
    k: usize,
    d: usize,
    acc: Option<&mut [f32]>,
) -> Result<()> {
    let (start, order) = group_by_label(labels, k)?;
    for j in 0..k {
        counts[j] = (start[j + 1] - start[j]) as u32;
    }
    let width = column_width(k, d);
    let one = |j: usize, s: &mut [f32], a: Option<&mut [f32]>| {
        let mine = &order[start[j]..start[j + 1]];
        s.fill(0.0);
        s.par_chunks_mut(width)
            .enumerate()
            .for_each(|(q, part)| add_group(part, rows, mine, q * width, d));
        if let Some(a) = a {
            for (p, &v) in a.iter_mut().zip(s.iter()) {
                *p += v;
            }
        }
    };
    match acc {
        Some(a) => sums
            .par_chunks_mut(d)
            .zip(a.par_chunks_mut(d))
            .enumerate()
            .for_each(|(j, (s, a))| one(j, s, Some(a))),
        None => sums
            .par_chunks_mut(d)
            .enumerate()
            .for_each(|(j, s)| one(j, s, None)),
    }
    Ok(())
}

/// The EMA step in f32, faiss's float order (see [`Device::ema_update`]).
pub(crate) fn ema_host(
    positions: &mut [f32],
    usage: &mut [f32],
    sums: &[f32],
    counts: &[u32],
    d: usize,
    step: EmaStep,
) {
    on_cpu_pool(|| ema_here(positions, usage, sums, counts, d, step))
}

fn ema_here(
    positions: &mut [f32],
    usage: &mut [f32],
    sums: &[f32],
    counts: &[u32],
    d: usize,
    step: EmaStep,
) {
    let EmaStep {
        alpha: a32,
        decay,
        replace,
    } = step;
    if !replace {
        for u in usage.iter_mut() {
            *u *= decay;
        }
    }
    positions
        .par_chunks_mut(d)
        .zip(usage.par_iter_mut())
        .zip(sums.par_chunks(d))
        .zip(counts.par_iter())
        .with_min_len(small_task_rows(d))
        .for_each(|(((pos, usage), sums), &count)| {
            if count == 0 {
                return;
            }
            let count = count as f32;
            let retained = if replace { 0.0 } else { *usage };
            let step = a32 * count;
            let mass = retained + step;
            let weight = step / mass;
            for (c, &s) in pos.iter_mut().zip(sums) {
                let mean = s / count;
                *c = if weight == 1.0 {
                    mean
                } else {
                    *c + weight * (mean - *c)
                };
            }
            *usage = mass;
        });
}
