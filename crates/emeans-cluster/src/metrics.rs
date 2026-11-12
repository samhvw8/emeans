//! Clustering quality measures.
//!
//! - External (against known labels): adjusted Rand index [`ari`] and
//!   normalized mutual information [`nmi`] (arithmetic-mean normalization,
//!   the scikit-learn default).
//! - Internal (no labels), for unit vectors: [`internal_scores`] gives the
//!   cosine-distance silhouette, Calinski-Harabasz and Davies-Bouldin indices
//!   and the spherical objective from one `n x k` product with the cluster sums.
//!
//! The silhouette is exact, not sampled. With the cosine distance `1 - x.y`
//! the mean distance from row `i` to the rows of cluster `C` is
//! `1 - x_i . s_C / |C|`, where `s_C` is the sum of the rows of `C`, so all
//! `n` silhouettes cost one `n x k` product instead of `n^2` distances.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::assign::gemm_nt;

/// Joint, row and column counts of two labelings.
type Counts = (
    HashMap<(u32, u32), f64>,
    HashMap<u32, f64>,
    HashMap<u32, f64>,
);

fn comb2(v: f64) -> f64 {
    v * (v - 1.0) / 2.0
}

fn counts(a: &[u32], b: &[u32]) -> Counts {
    let mut ab = HashMap::new();
    let mut ca = HashMap::new();
    let mut cb = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *ab.entry((x, y)).or_insert(0.0) += 1.0;
        *ca.entry(x).or_insert(0.0) += 1.0;
        *cb.entry(y).or_insert(0.0) += 1.0;
    }
    (ab, ca, cb)
}

/// Adjusted Rand index of two labelings of the same rows (1 = identical partitions).
pub fn ari(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len(), "label vectors differ in length");
    let n = a.len() as f64;
    if a.len() < 2 {
        return 1.0;
    }
    let (ab, ca, cb) = counts(a, b);
    let index: f64 = ab.values().map(|&v| comb2(v)).sum();
    let sa: f64 = ca.values().map(|&v| comb2(v)).sum();
    let sb: f64 = cb.values().map(|&v| comb2(v)).sum();
    let expected = sa * sb / comb2(n);
    let max = 0.5 * (sa + sb);
    if (max - expected).abs() < 1e-12 {
        // Both partitions are trivial (all one cluster or all singletons).
        return 1.0;
    }
    (index - expected) / (max - expected)
}

/// Normalized mutual information, `I(A;B) / ((H(A) + H(B)) / 2)`.
pub fn nmi(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len(), "label vectors differ in length");
    let n = a.len() as f64;
    if a.is_empty() {
        return 1.0;
    }
    let (ab, ca, cb) = counts(a, b);
    let h = |c: &HashMap<u32, f64>| -> f64 {
        c.values()
            .map(|&v| {
                let p = v / n;
                -p * p.ln()
            })
            .sum()
    };
    let (ha, hb) = (h(&ca), h(&cb));
    if ha == 0.0 && hb == 0.0 {
        return 1.0;
    }
    let mi: f64 = ab
        .iter()
        .map(|(&(x, y), &v)| v / n * (v * n / (ca[&x] * cb[&y])).ln())
        .sum();
    (mi / (0.5 * (ha + hb))).clamp(0.0, 1.0)
}

/// Internal quality of one partition of unit vectors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Internal {
    /// Non-empty clusters.
    pub k_eff: usize,
    /// Mean silhouette with the cosine distance `1 - x.y`; 0 when `k_eff < 2`.
    /// Singleton clusters score 0 (scikit-learn convention). Higher is better.
    pub silhouette: f64,
    /// Variance-ratio criterion on the unit vectors (Euclidean). Higher is better.
    pub calinski_harabasz: f64,
    /// Davies-Bouldin index on the unit vectors (Euclidean). Lower is better.
    pub davies_bouldin: f64,
    /// Mean `2 - 2 cos` to the normalized cluster mean (the spherical e-means objective).
    pub objective: f64,
    /// Rows per label `0..k` (zeros for empty clusters).
    pub sizes: Vec<usize>,
}

/// Silhouette, Calinski-Harabasz, Davies-Bouldin and objective of `labels`
/// (values `< k`) on the unit rows of `x` (`n x d`).
pub fn internal_scores(x: &[f32], d: usize, labels: &[u32], k: usize) -> Result<Internal> {
    ensure!(d > 0 && x.len().is_multiple_of(d), "x must be n x d");
    let n = x.len() / d;
    ensure!(labels.len() == n, "one label per row");
    ensure!(
        labels.iter().all(|&l| (l as usize) < k),
        "label out of range"
    );
    let mut sizes = vec![0usize; k];
    let mut sums = vec![0f64; k * d];
    for (i, &l) in labels.iter().enumerate() {
        sizes[l as usize] += 1;
        let s = &mut sums[l as usize * d..(l as usize + 1) * d];
        for (a, &b) in s.iter_mut().zip(&x[i * d..(i + 1) * d]) {
            *a += b as f64;
        }
    }
    let live: Vec<usize> = (0..k).filter(|&j| sizes[j] > 0).collect();
    let k_eff = live.len();
    let sums32: Vec<f32> = live
        .iter()
        .flat_map(|&j| sums[j * d..(j + 1) * d].iter().map(|&v| v as f32))
        .collect();
    let sq_sum: Vec<f64> = live
        .iter()
        .map(|&j| sums[j * d..(j + 1) * d].iter().map(|v| v * v).sum())
        .collect();
    let mut slot = vec![usize::MAX; k];
    for (s, &j) in live.iter().enumerate() {
        slot[j] = s;
    }
    let xx: Vec<f64> = x
        .par_chunks(d)
        .map(|r| r.iter().map(|&v| v as f64 * v as f64).sum())
        .collect();

    // Dot products of every row with every live cluster sum, in row blocks.
    let block = ((4 << 20) / k_eff.max(1)).clamp(256, 16384);
    let mut sil_sum = 0f64;
    let mut obj_sum = 0f64;
    let mut scatter = vec![0f64; k_eff];
    let mut g = Vec::new();
    for b0 in (0..n).step_by(block) {
        let b1 = (b0 + block).min(n);
        g.resize((b1 - b0) * k_eff, 0f32);
        gemm_nt(&x[b0 * d..b1 * d], &sums32, d, &mut g)?;
        let part: Vec<(f64, f64, usize, f64)> = (b0..b1)
            .into_par_iter()
            .map(|i| {
                let row = &g[(i - b0) * k_eff..(i - b0 + 1) * k_eff];
                let own = slot[labels[i] as usize];
                let na = sizes[live[own]] as f64;
                let dot_own = row[own] as f64;
                // Euclidean distance to the own mean, for Davies-Bouldin.
                let dist = (xx[i] - 2.0 * dot_own / na + sq_sum[own] / (na * na))
                    .max(0.0)
                    .sqrt();
                let cos_c = dot_own / sq_sum[own].sqrt().max(1e-30);
                let obj = 2.0 - 2.0 * cos_c;
                let sil = if k_eff < 2 || na < 2.0 {
                    0.0
                } else {
                    let a = 1.0 - (dot_own - xx[i]) / (na - 1.0);
                    let mut b = f64::INFINITY;
                    for (s, &v) in row.iter().enumerate() {
                        if s != own {
                            b = b.min(1.0 - v as f64 / sizes[live[s]] as f64);
                        }
                    }
                    let m = a.max(b);
                    if m > 0.0 { (b - a) / m } else { 0.0 }
                };
                (sil, obj, own, dist)
            })
            .collect();
        for (sil, obj, own, dist) in part {
            sil_sum += sil;
            obj_sum += obj;
            scatter[own] += dist;
        }
    }
    let nf = n.max(1) as f64;

    // Calinski-Harabasz: between / within dispersion on the unit vectors.
    let total_xx: f64 = xx.iter().sum();
    let within: f64 = total_xx
        - live
            .iter()
            .zip(&sq_sum)
            .map(|(&j, &s)| s / sizes[j] as f64)
            .sum::<f64>();
    let mut grand = vec![0f64; d];
    for &j in &live {
        for (a, &b) in grand.iter_mut().zip(&sums[j * d..(j + 1) * d]) {
            *a += b;
        }
    }
    let total = total_xx - grand.iter().map(|v| v * v).sum::<f64>() / nf;
    let between = total - within;
    let calinski_harabasz = if k_eff < 2 || n <= k_eff || within <= 0.0 {
        0.0
    } else {
        (between / (k_eff - 1) as f64) / (within / (n - k_eff) as f64)
    };

    // Davies-Bouldin: mean over clusters of the worst (S_a + S_b) / ||m_a - m_b||.
    let means: Vec<Vec<f64>> = live
        .iter()
        .map(|&j| {
            sums[j * d..(j + 1) * d]
                .iter()
                .map(|v| v / sizes[j] as f64)
                .collect()
        })
        .collect();
    let spread: Vec<f64> = live
        .iter()
        .zip(&scatter)
        .map(|(&j, s)| s / sizes[j] as f64)
        .collect();
    let davies_bouldin = if k_eff < 2 {
        0.0
    } else {
        (0..k_eff)
            .into_par_iter()
            .map(|a| {
                (0..k_eff)
                    .filter(|&b| b != a)
                    .map(|b| {
                        let m: f64 = means[a]
                            .iter()
                            .zip(&means[b])
                            .map(|(p, q)| (p - q) * (p - q))
                            .sum::<f64>()
                            .sqrt();
                        if m > 0.0 {
                            (spread[a] + spread[b]) / m
                        } else {
                            f64::INFINITY
                        }
                    })
                    .fold(0.0, f64::max)
            })
            .sum::<f64>()
            / k_eff as f64
    };

    Ok(Internal {
        k_eff,
        silhouette: sil_sum / nf,
        calinski_harabasz,
        davies_bouldin,
        objective: obj_sum / nf,
        sizes,
    })
}

/// The `q`-quantile (0..=1, nearest rank) of `v`; NaN for an empty slice.
pub fn quantile(v: &[f32], q: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let i = ((q.clamp(0.0, 1.0) * (s.len() - 1) as f64).round()) as usize;
    s[i]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ari_nmi_basics() {
        let a = [0, 0, 1, 1, 2, 2];
        let b = [5, 5, 7, 7, 9, 9];
        assert!((ari(&a, &b) - 1.0).abs() < 1e-12);
        assert!((nmi(&a, &b) - 1.0).abs() < 1e-12);
        // scikit-learn: adjusted_rand_score([0,0,1,1],[0,0,1,2]) = 0.5714285714285715
        assert!((ari(&[0, 0, 1, 1], &[0, 0, 1, 2]) - 0.571_428_571_428_571_5).abs() < 1e-12);
        // normalized_mutual_info_score([0,0,1,1],[0,0,1,2]) = 0.8
        assert!((nmi(&[0, 0, 1, 1], &[0, 0, 1, 2]) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn silhouette_matches_brute_force() {
        let d = 5;
        let mut rng = crate::rng::RandomState::new(3);
        let n = 60;
        let mut x: Vec<f32> = (0..n * d)
            .map(|_| rng.random_sample() as f32 - 0.5)
            .collect();
        crate::model::normalize_rows(&mut x, d);
        let labels: Vec<u32> = (0..n as u32).map(|i| i % 4).collect();
        let got = internal_scores(&x, d, &labels, 5).unwrap();
        let dot = |i: usize, j: usize| -> f64 {
            (0..d)
                .map(|t| x[i * d + t] as f64 * x[j * d + t] as f64)
                .sum()
        };
        let mut s = 0.0;
        for i in 0..n {
            let mut mean = [0f64; 4];
            let mut cnt = [0f64; 4];
            for j in 0..n {
                if j != i {
                    mean[labels[j] as usize] += 1.0 - dot(i, j);
                    cnt[labels[j] as usize] += 1.0;
                }
            }
            let a = mean[labels[i] as usize] / cnt[labels[i] as usize];
            let b = (0..4)
                .filter(|&c| c != labels[i] as usize)
                .map(|c| mean[c] / cnt[c])
                .fold(f64::INFINITY, f64::min);
            s += (b - a) / a.max(b);
        }
        assert_eq!(got.k_eff, 4);
        assert!(
            (got.silhouette - s / n as f64).abs() < 1e-5,
            "{} vs {}",
            got.silhouette,
            s / n as f64
        );
    }
}
