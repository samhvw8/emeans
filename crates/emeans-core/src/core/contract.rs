//! The exactness contract of layer 1: run every op on a backend and compare it
//! with [`super::reference`] (f64) or, for the exact ops, with the f32 float
//! order the CPU backend uses. A new backend passes [`check`] before it trains.

use anyhow::{Result, bail, ensure};

use super::{Device, EmaStep, Mat, Select, reference};
use crate::assign::Metric;
use crate::rng::RandomState;

/// Two centroids whose f64 reference scores differ by at most this (times the
/// magnitude) are a near-tie: either label is accepted.
pub const TIE_EPS: f64 = 1e-5;
/// A reported score may differ from the f64 score of its label by this (times the magnitude).
pub const SCORE_EPS: f64 = 1e-5;
/// Row norms and normalized rows: relative error against f64.
pub const NORM_EPS: f64 = 1e-5;

/// Largest errors seen, per op.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub backend: String,
    /// Rows checked by top-1 and labels that differed from the reference (all near-ties).
    pub top1_rows: usize,
    pub top1_ties: usize,
    pub top1_score_err: f64,
    pub topk_rows: usize,
    pub topk_ties: usize,
    pub topk_score_err: f64,
    pub norm_err: f64,
    pub normalize_err: f64,
    /// Scatter-add sums and EMA/split results that were not bit-equal (must be 0).
    pub scatter_mismatch: usize,
    pub ema_mismatch: usize,
    /// Row norms and normalized values not equal to the CPU float order (must be 0).
    pub norm_mismatch: usize,
}

/// Gaussian clusters, `n x d`, deterministic.
pub fn mixture(n: usize, d: usize, centers: usize, sigma: f64, unit: bool, seed: u32) -> Vec<f32> {
    let mut rng = RandomState::new(seed);
    let mut gauss = move || {
        let u1 = rng.random_sample().max(1e-300);
        let u2 = rng.random_sample();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let cs: Vec<f64> = (0..centers * d).map(|_| gauss()).collect();
    let mut x = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = &cs[(i % centers) * d..(i % centers + 1) * d];
        let row: Vec<f64> = c.iter().map(|&v| v + sigma * gauss()).collect();
        let s = if unit {
            row.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12)
        } else {
            1.0
        };
        x.extend(row.iter().map(|v| (v / s) as f32));
    }
    x
}

fn sq(r: &[f32]) -> f64 {
    r.iter().map(|&v| v as f64 * v as f64).sum()
}

/// Score magnitude used to scale the tolerances.
fn magnitude(x: &[f32], c: &[f32], metric: Metric) -> f64 {
    match metric {
        Metric::InnerProduct => (sq(x) * sq(c)).sqrt().max(1.0),
        Metric::L2 => (sq(x) + sq(c)).max(1.0),
    }
}

/// Run every op on `dev`; fail on the first violation.
pub fn check<D: Device>(dev: &mut D) -> Result<Report> {
    let mut rep = Report {
        backend: dev.name(),
        ..Default::default()
    };
    for &(n, d, k) in &[
        (1000, 768, 64),
        (777, 768, 100),
        (513, 13, 70),
        (2000, 20, 300),
        (300, 64, 1),
        (600, 768, 1024),
    ] {
        for metric in [Metric::InnerProduct, Metric::L2] {
            let unit = metric == Metric::InnerProduct;
            let x = mixture(n, d, 11, 0.6, unit, (3 * n + k) as u32);
            let c: Vec<f32> = x[..k.min(n) * d]
                .iter()
                .cycle()
                .take(k * d)
                .map(|v| v * 0.95)
                .collect();
            check_top1(dev, &mut rep, &x, d, &c, metric, 1)?;
        }
        let x = mixture(n, d, 7, 0.5, false, (n + d) as u32);
        check_scatter_ema(dev, &mut rep, &x, d, k.min(n))?;
        check_norms(dev, &mut rep, &x, d)?;
    }
    for &(m, n, d, kk) in &[(300, 300, 768, 16), (200, 900, 20, 5), (130, 7, 33, 10)] {
        let data = mixture(n, d, 9, 0.3, true, (m + n) as u32);
        let q = mixture(m, d, 9, 0.3, true, (m * 7 + n) as u32);
        check_topk(dev, &mut rep, &q, &data, d, kk, false)?;
        if m == n {
            check_topk(dev, &mut rep, &data, &data, d, kk, true)?;
        }
    }
    Ok(rep)
}

/// The production batch shape: 16,384 rows, d 768, k 1024 and 1000 (not a
/// multiple of any tile), with exact duplicate centroids in other centroid
/// tiles; top-k of 64 queries against 16,384 rows with duplicate data rows.
/// The f64 reference runs on every 64th selected row (one per row tile, and
/// the last 70); the tie rule and the "outside the selection" checks cover
/// every row.
pub fn check_production<D: Device>(dev: &mut D) -> Result<Report> {
    let mut rep = Report {
        backend: dev.name(),
        ..Default::default()
    };
    let (n, d) = (16_384, 768);
    for k in [1024, 1000] {
        for metric in [Metric::InnerProduct, Metric::L2] {
            let unit = metric == Metric::InnerProduct;
            let x = mixture(n, d, 64, 0.6, unit, (5 * n + k) as u32);
            // The last 256 centroids copy earlier ones (other tiles, other
            // fragment positions): the lower id must win the exact tie.
            let base = k - 256;
            let mut c = vec![0f32; k * d];
            for j in 0..k {
                let row: Vec<f32> = if j < base {
                    let r = (j * 37) % n;
                    x[r * d..(r + 1) * d].iter().map(|v| v * 0.95).collect()
                } else {
                    let o = (j * 7) % base;
                    c[o * d..(o + 1) * d].to_vec()
                };
                c[j * d..(j + 1) * d].copy_from_slice(&row);
            }
            check_top1(dev, &mut rep, &x, d, &c, metric, 64)?;
        }
    }
    let mut data = mixture(n, d, 9, 0.3, true, 77);
    for j in 8192..8192 + 512 {
        let src = data[(j - 8000) * d..(j - 8000 + 1) * d].to_vec();
        data[j * d..(j + 1) * d].copy_from_slice(&src);
    }
    // Queries: rows with a duplicate (exact ties at rank 0) and plain rows.
    let q: Vec<f32> = (0..64)
        .flat_map(|i| {
            let r = if i % 2 == 0 { 192 + i } else { (i * 61) % n };
            data[r * d..(r + 1) * d].to_vec()
        })
        .collect();
    check_topk(dev, &mut rep, &q, &data, d, crate::knn::GPU_KMAX, false)?;
    Ok(rep)
}

/// For each row of `m` (`rows x d`), the lowest row id with the same bits.
fn first_duplicate(m: &[f32], d: usize) -> Vec<usize> {
    let mut seen = std::collections::HashMap::new();
    m.chunks(d)
        .enumerate()
        .map(|(j, r)| {
            let key: Vec<u32> = r.iter().map(|v| v.to_bits()).collect();
            *seen.entry(key).or_insert(j)
        })
        .collect()
}

/// `stride` > 1: the f64 reference (labels and scores) runs only on every
/// `stride`-th selected position and the last 70; the tie rule runs on all.
fn check_top1<D: Device>(
    dev: &mut D,
    rep: &mut Report,
    x: &[f32],
    d: usize,
    c: &[f32],
    metric: Metric,
    stride: usize,
) -> Result<()> {
    let (n, k) = (x.len() / d, c.len() / d);
    let first = first_duplicate(c, d);
    let xb = dev.bind(x)?;
    let cb = dev.upload_f32(c)?;
    let cn = match metric {
        Metric::L2 => {
            let mut o = dev.zeros_f32(k)?;
            dev.row_norms(&cb, k, d, &mut o)?;
            Some(o)
        }
        Metric::InnerProduct => None,
    };
    // Indexed (a permutation, a sub-range) and dense selections.
    let perm: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 13) % n as u32).collect();
    let ids = dev.upload_u32(&perm)?;
    let lo = n / 7;
    let hi = n - n / 9;
    for indexed in [true, false] {
        let mut lab = dev.upload_u32(&vec![u32::MAX; n])?;
        let mut sco = dev.upload_f32(&vec![f32::NAN; n])?;
        dev.top1(
            Mat {
                host: x,
                dev: &xb,
                d,
            },
            Select {
                ids: indexed.then_some(&ids),
                rows: lo..hi,
            },
            &cb,
            cn.as_ref(),
            k,
            metric,
            &mut lab,
            &mut sco,
        )?;
        let l = dev.read_u32(&lab, 0..n)?;
        let s = dev.read_f32(&sco, 0..n)?;
        for p in 0..n {
            if !(lo..hi).contains(&p) {
                ensure!(
                    l[p] == u32::MAX && s[p].is_nan(),
                    "top1 wrote position {p} outside the selection"
                );
                continue;
            }
            let i = if indexed { perm[p] as usize } else { p };
            let r = &x[i * d..(i + 1) * d];
            let got = l[p] as usize;
            ensure!(got < k, "top1 label {got} >= k = {k}");
            // Ties keep the lower id: an exact duplicate centroid scores the same.
            ensure!(
                first[got] == got,
                "{}: top1 {metric:?} n={n} d={d} k={k} row {i}: label {got}, but centroid {} is the same vector with a lower id",
                rep.backend,
                first[got]
            );
            if !(p - lo).is_multiple_of(stride) && p + 70 < hi {
                continue;
            }
            let cg = &c[got * d..(got + 1) * d];
            let mag = magnitude(r, cg, metric);
            let sg = reference::score(r, cg, metric);
            let (want, _) = reference::top1(r, d, c, metric);
            if got != want[0] as usize {
                let w = want[0] as usize;
                let sw = reference::score(r, &c[w * d..(w + 1) * d], metric);
                if (sg - sw).abs() > TIE_EPS * mag {
                    bail!(
                        "{}: top1 {metric:?} n={n} d={d} k={k} row {i}: label {got} (score {sg}) vs reference {w} ({sw}), not a tie",
                        rep.backend
                    );
                }
                rep.top1_ties += 1;
            }
            let err = (s[p] as f64 - sg).abs() / mag;
            ensure!(
                err <= SCORE_EPS,
                "{}: top1 {metric:?} n={n} d={d} k={k} row {i}: score {} vs {sg}",
                rep.backend,
                s[p]
            );
            rep.top1_score_err = rep.top1_score_err.max(err);
            rep.top1_rows += 1;
        }
    }
    Ok(())
}

fn check_topk<D: Device>(
    dev: &mut D,
    rep: &mut Report,
    q: &[f32],
    data: &[f32],
    d: usize,
    kk: usize,
    exclude_self: bool,
) -> Result<()> {
    let (ids, sims) = dev.topk(q, data, d, kk, exclude_self)?;
    let (wid, wsim) = reference::topk(q, data, d, kk, exclude_self);
    let m = q.len() / d;
    // Ties keep the lower id: a returned row's exact duplicate with a lower id
    // (other than the query itself under exclude_self) must rank before it.
    let first = first_duplicate(data, d);
    for i in 0..m {
        let row = &ids[i * kk..(i + 1) * kk];
        for (t, &j) in row.iter().enumerate() {
            if j == u32::MAX {
                continue;
            }
            let f = first[j as usize];
            if f != j as usize && !(exclude_self && f == i) {
                ensure!(
                    row[..t].contains(&(f as u32)),
                    "{}: topk row {i} rank {t}: id {j}, but row {f} is the same vector with a lower id and does not rank before it",
                    rep.backend
                );
            }
        }
    }
    ensure!(
        ids.len() == m * kk && sims.len() == m * kk,
        "topk output shape"
    );
    for i in 0..m {
        for t in 0..kk {
            let p = i * kk + t;
            if wid[p] == u32::MAX {
                ensure!(
                    ids[p] == u32::MAX && sims[p] == f32::NEG_INFINITY,
                    "{}: topk row {i} rank {t}: padding expected",
                    rep.backend
                );
                continue;
            }
            let j = ids[p] as usize;
            ensure!(
                j < data.len() / d && !(exclude_self && j == i),
                "{}: topk row {i} rank {t}: bad id {j}",
                rep.backend
            );
            let qi = &q[i * d..(i + 1) * d];
            let sj = reference::score(qi, &data[j * d..(j + 1) * d], Metric::InnerProduct);
            let mag = magnitude(qi, qi, Metric::InnerProduct);
            if ids[p] != wid[p] {
                ensure!(
                    (sj - wsim[p]).abs() <= TIE_EPS * mag,
                    "{}: topk row {i} rank {t}: id {j} ({sj}) vs reference {} ({}), not a tie",
                    rep.backend,
                    wid[p],
                    wsim[p]
                );
                rep.topk_ties += 1;
            }
            let err = (sims[p] as f64 - sj).abs() / mag;
            ensure!(
                err <= SCORE_EPS,
                "{}: topk score {} vs {sj}",
                rep.backend,
                sims[p]
            );
            rep.topk_score_err = rep.topk_score_err.max(err);
        }
        rep.topk_rows += 1;
    }
    Ok(())
}

fn check_scatter_ema<D: Device>(
    dev: &mut D,
    rep: &mut Report,
    x: &[f32],
    d: usize,
    k: usize,
) -> Result<()> {
    let n = x.len() / d;
    // Skewed labels with empty clusters: label = hash % (k - k/5).
    let live = (k - k / 5).max(1);
    let labels: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 7) % live as u32)
        .collect();
    let perm: Vec<u32> = (0..n as u32).map(|i| (i * 31 + 5) % n as u32).collect();
    let (lo, hi) = (n / 5, n);
    let xb = dev.bind(x)?;
    let ids = dev.upload_u32(&perm)?;
    let lab = dev.upload_u32(&labels)?;
    let mut sums = dev.upload_f32(&vec![7.0; k * d])?;
    let mut counts = dev.upload_u32(&vec![3; k])?;
    let acc0: Vec<f32> = (0..k * d).map(|i| (i % 13) as f32 * 0.25).collect();
    let mut acc = dev.upload_f32(&acc0)?;
    dev.scatter_add(
        Mat {
            host: x,
            dev: &xb,
            d,
        },
        Select {
            ids: Some(&ids),
            rows: lo..hi,
        },
        &lab,
        k,
        &mut sums,
        &mut counts,
        Some(&mut acc),
    )?;
    let got_s = dev.read_f32(&sums, 0..k * d)?;
    let got_c = dev.read_u32(&counts, 0..k)?;
    let got_a = dev.read_f32(&acc, 0..k * d)?;
    // Expected: rows added in selection order, in f32.
    let mut want_s = vec![0f32; k * d];
    let mut want_c = vec![0u32; k];
    let rows: Vec<&[f32]> = (lo..hi)
        .map(|p| &x[perm[p] as usize * d..(perm[p] as usize + 1) * d])
        .collect();
    for (r, &l) in rows.iter().zip(&labels[lo..hi]) {
        let l = l as usize;
        want_c[l] += 1;
        for (s, &v) in want_s[l * d..(l + 1) * d].iter_mut().zip(r.iter()) {
            *s += v;
        }
    }
    let (ref_s, ref_c) = reference::scatter_add(&rows, &labels[lo..hi], k, d);
    ensure!(
        got_c == want_c && ref_c == want_c,
        "{}: scatter counts",
        rep.backend
    );
    for i in 0..k * d {
        if got_s[i].to_bits() != want_s[i].to_bits() {
            rep.scatter_mismatch += 1;
        }
        let a = acc0[i] + want_s[i];
        if got_a[i].to_bits() != a.to_bits() {
            rep.scatter_mismatch += 1;
        }
        let tol = 1e-5 * (1.0 + ref_s[i].abs()) * (want_c[i / d] as f64).max(1.0).sqrt();
        ensure!(
            (got_s[i] as f64 - ref_s[i]).abs() <= tol,
            "{}: scatter sum {i}: {} vs f64 {}",
            rep.backend,
            got_s[i],
            ref_s[i]
        );
    }
    ensure!(
        rep.scatter_mismatch == 0,
        "{}: {} scatter-add sums not bit-equal to row-order f32 (n={n} d={d} k={k})",
        rep.backend,
        rep.scatter_mismatch
    );

    // EMA: decayed, replaced (alpha 1) and weight-1 cases.
    let pos0: Vec<f32> = x[..k * d].to_vec();
    for (alpha, u0) in [(0.37, 0.8f32), (1.0, 0.5), (0.9, 0.0)] {
        let usage0: Vec<f32> = (0..k).map(|j| u0 + (j % 5) as f32 * 0.125).collect();
        let mut p = dev.upload_f32(&pos0)?;
        let mut u = dev.upload_f32(&usage0)?;
        let step = EmaStep::new(alpha);
        dev.ema_update(&mut p, &mut u, &sums, &counts, k, d, step)?;
        let gp = dev.read_f32(&p, 0..k * d)?;
        let gu = dev.read_f32(&u, 0..k)?;
        let mut wp = pos0.clone();
        let mut wu = usage0.clone();
        super::cpu::ema_host(&mut wp, &mut wu, &got_s, &got_c, d, step);
        let mism = gp
            .iter()
            .zip(&wp)
            .chain(gu.iter().zip(&wu))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        rep.ema_mismatch += mism;
        let (rp, ru) = reference::ema(&pos0, &usage0, &got_s, &got_c, d, alpha);
        for (g, r) in gp.iter().zip(&rp).chain(gu.iter().zip(&ru)) {
            ensure!(
                (*g as f64 - r).abs() <= 1e-5 * (1.0 + r.abs()),
                "{}: ema {g} vs f64 {r}",
                rep.backend
            );
        }
    }
    // Split: donors with a zero row (the child-nudge path) and normal rows.
    let mut pos1 = pos0.clone();
    if k >= 4 {
        pos1[d..2 * d].fill(0.0);
    }
    let usage1: Vec<f32> = (0..k).map(|j| 1.0 + j as f32 * 0.37).collect();
    let plan: Vec<(usize, usize)> = (0..k / 2)
        .filter(|j| j % 3 != 2)
        .map(|j| (j, k - 1 - j))
        .filter(|(a, b)| a != b)
        .collect();
    let mut p = dev.upload_f32(&pos1)?;
    let mut u = dev.upload_f32(&usage1)?;
    dev.split(&mut p, &mut u, &plan, d)?;
    let (gp, gu) = (dev.read_f32(&p, 0..k * d)?, dev.read_f32(&u, 0..k)?);
    let (mut wp, mut wu) = (pos1.clone(), usage1.clone());
    super::apply_splits(&mut wp, &mut wu, d, &plan);
    rep.ema_mismatch += gp
        .iter()
        .zip(&wp)
        .chain(gu.iter().zip(&wu))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(
        rep.ema_mismatch == 0,
        "{}: {} EMA or split values not bit-equal to the CPU float order (d={d} k={k})",
        rep.backend,
        rep.ema_mismatch
    );
    Ok(())
}

fn check_norms<D: Device>(dev: &mut D, rep: &mut Report, x: &[f32], d: usize) -> Result<()> {
    let rows = (x.len() / d).min(1100);
    let mut m = x[..rows * d].to_vec();
    // A zero row and a huge row (the max-prescale path).
    m[..d].fill(0.0);
    if rows > 1 {
        for v in &mut m[d..2 * d] {
            *v *= 1e30;
        }
    }
    let mb = dev.upload_f32(&m)?;
    let mut nb = dev.zeros_f32(rows)?;
    dev.row_norms(&mb, rows, d, &mut nb)?;
    let got = dev.read_f32(&nb, 0..rows)?;
    let cpu = crate::assign::row_norms(&m, d);
    rep.norm_mismatch += got.iter().zip(&cpu).filter(|(a, b)| a != b).count();
    let want = reference::row_norms(&m, d);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if i == 1 {
            continue; // overflows f32 by design
        }
        let e = (*g as f64 - w).abs() / w.max(1e-30);
        ensure!(
            e <= NORM_EPS || *w == 0.0 && *g == 0.0,
            "{}: row norm {i}: {g} vs {w}",
            rep.backend
        );
        if *w > 0.0 {
            rep.norm_err = rep.norm_err.max(e);
        }
    }
    let mut ub = dev.zeros_f32(rows * d)?;
    dev.normalize_rows(&mb, rows, d, &mut ub)?;
    let got = dev.read_f32(&ub, 0..rows * d)?;
    let mut cpu = m.clone();
    crate::model::normalize_rows(&mut cpu, d);
    rep.norm_mismatch += got.iter().zip(&cpu).filter(|(a, b)| a != b).count();
    let want = reference::normalize(&m, d);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let e = (*g as f64 - w).abs();
        ensure!(
            e <= NORM_EPS * w.abs().max(1.0 / (d as f64).sqrt()),
            "{}: normalize {i}: {g} vs {w}",
            rep.backend
        );
        rep.normalize_err = rep.normalize_err.max(e);
    }
    ensure!(
        rep.norm_mismatch == 0,
        "{}: {} row norms or normalized values differ from the CPU float order (d={d})",
        rep.backend,
        rep.norm_mismatch
    );
    Ok(())
}
