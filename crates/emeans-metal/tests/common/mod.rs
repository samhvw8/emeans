//! Shared test helpers.
#![allow(dead_code)]

use emeans_core::rng::RandomState;

/// Gaussian mixture from MT19937 (Box-Muller); unit rows when `spherical`.
pub fn mixture(
    n: usize,
    d: usize,
    centers: usize,
    spread: f32,
    spherical: bool,
    seed: u32,
) -> Vec<f32> {
    let mut rng = RandomState::new(seed);
    let mut normal = move || {
        let u1 = rng.random_sample().max(1e-300);
        let u2 = rng.random_sample();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };
    let c: Vec<f32> = (0..centers * d).map(|_| 3.0 * normal()).collect();
    let mut x = vec![0f32; n * d];
    for i in 0..n {
        let j = i % centers;
        for t in 0..d {
            x[i * d + t] = c[j * d + t] + spread * normal();
        }
        if spherical {
            let r = &mut x[i * d..(i + 1) * d];
            let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            r.iter_mut().for_each(|v| *v /= norm);
        }
    }
    x
}

pub fn exact(x: &[f32], c: &[f32], ip: bool) -> f64 {
    if ip {
        x.iter().zip(c).map(|(a, b)| *a as f64 * *b as f64).sum()
    } else {
        x.iter()
            .zip(c)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum()
    }
}

/// Count label disagreements that are not ties (exact f64 scores differ by more than tol).
pub fn non_ties(x: &[f32], c: &[f32], d: usize, a: &[u32], b: &[u32], ip: bool, tol: f64) -> usize {
    a.iter()
        .zip(b)
        .enumerate()
        .filter(|(i, (la, lb))| {
            la != lb && {
                let r = &x[i * d..(i + 1) * d];
                let sa = exact(r, &c[**la as usize * d..(**la as usize + 1) * d], ip);
                let sb = exact(r, &c[**lb as usize * d..(**lb as usize + 1) * d], ip);
                (sa - sb).abs() > tol * sb.abs().max(1.0)
            }
        })
        .count()
}
