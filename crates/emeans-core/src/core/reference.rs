//! The f64 reference of every layer-1 op: plain loops, no blocking, no
//! parallelism. Slow on purpose; [`super::contract`] compares backends to it.

use crate::assign::Metric;

/// Score of one row against one centroid in f64: inner product, or squared distance.
pub fn score(x: &[f32], c: &[f32], metric: Metric) -> f64 {
    match metric {
        Metric::InnerProduct => x.iter().zip(c).map(|(&a, &b)| a as f64 * b as f64).sum(),
        Metric::L2 => x
            .iter()
            .zip(c)
            .map(|(&a, &b)| {
                let t = a as f64 - b as f64;
                t * t
            })
            .sum(),
    }
}

fn better(metric: Metric, a: f64, b: f64) -> bool {
    match metric {
        Metric::InnerProduct => a > b,
        Metric::L2 => a < b,
    }
}

/// Best centroid (lowest id on ties) and its f64 score, per row of `x` (`m x d`).
pub fn top1(x: &[f32], d: usize, c: &[f32], metric: Metric) -> (Vec<u32>, Vec<f64>) {
    let k = c.len() / d;
    x.chunks(d)
        .map(|r| {
            let mut bj = 0;
            let mut bs = score(r, &c[..d], metric);
            for j in 1..k {
                let s = score(r, &c[j * d..(j + 1) * d], metric);
                if better(metric, s, bs) {
                    bs = s;
                    bj = j;
                }
            }
            (bj as u32, bs)
        })
        .unzip()
}

/// Best `kk` data rows per query by inner product, best first (larger score, lower id).
pub fn topk(
    queries: &[f32],
    data: &[f32],
    d: usize,
    kk: usize,
    exclude_self: bool,
) -> (Vec<u32>, Vec<f64>) {
    let n = data.len() / d;
    let mut ids = Vec::new();
    let mut sims = Vec::new();
    for (i, q) in queries.chunks(d).enumerate() {
        let mut all: Vec<(f64, u32)> = (0..n)
            .filter(|&j| !(exclude_self && j == i))
            .map(|j| {
                (
                    score(q, &data[j * d..(j + 1) * d], Metric::InnerProduct),
                    j as u32,
                )
            })
            .collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for t in 0..kk {
            match all.get(t) {
                Some(&(s, j)) => {
                    ids.push(j);
                    sims.push(s);
                }
                None => {
                    ids.push(u32::MAX);
                    sims.push(f64::NEG_INFINITY);
                }
            }
        }
    }
    (ids, sims)
}

/// Per-label sums in f64 and counts of the rows `rows` (each `d` long).
pub fn scatter_add(rows: &[&[f32]], labels: &[u32], k: usize, d: usize) -> (Vec<f64>, Vec<u32>) {
    let mut sums = vec![0f64; k * d];
    let mut counts = vec![0u32; k];
    for (r, &l) in rows.iter().zip(labels) {
        let l = l as usize;
        counts[l] += 1;
        for (s, &v) in sums[l * d..(l + 1) * d].iter_mut().zip(r.iter()) {
            *s += v as f64;
        }
    }
    (sums, counts)
}

pub fn row_norms(m: &[f32], d: usize) -> Vec<f64> {
    m.chunks(d)
        .map(|r| r.iter().map(|&v| v as f64 * v as f64).sum())
        .collect()
}

/// Rows scaled to unit norm in f64 (zero rows stay zero).
pub fn normalize(m: &[f32], d: usize) -> Vec<f64> {
    m.chunks(d)
        .flat_map(|r| {
            let n = r.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
            let s = if n > 0.0 { n } else { 1.0 };
            r.iter().map(move |&v| v as f64 / s)
        })
        .collect()
}

/// The EMA step in f64: (positions, masses).
pub fn ema(
    positions: &[f32],
    usage: &[f32],
    sums: &[f32],
    counts: &[u32],
    d: usize,
    alpha: f64,
) -> (Vec<f64>, Vec<f64>) {
    let replace = alpha == 1.0;
    let mut p: Vec<f64> = positions.iter().map(|&v| v as f64).collect();
    let mut u: Vec<f64> = usage
        .iter()
        .map(|&v| {
            if replace {
                v as f64
            } else {
                v as f64 * (1.0 - alpha)
            }
        })
        .collect();
    for (j, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let c = c as f64;
        let retained = if replace { 0.0 } else { u[j] };
        let mass = retained + alpha * c;
        let w = alpha * c / mass;
        for t in 0..d {
            let mean = sums[j * d + t] as f64 / c;
            p[j * d + t] += w * (mean - p[j * d + t]);
        }
        u[j] = mass;
    }
    (p, u)
}
