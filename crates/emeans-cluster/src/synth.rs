//! Synthetic test data with known structure, for validating automatic k,
//! incremental updates and near-duplicate grouping without real photos.
//!
//! [`nested`] draws unit vectors from a two-level mixture on the sphere:
//! broad groups around a shared global direction (real embeddings share a
//! common component, so unrelated photos still have a positive cosine), sub
//! groups around each broad center, points around each sub center, and some
//! noise rows that belong to no group. With the defaults the mean cosine is
//! about 0.75 inside a sub group, 0.55 inside a broad group, 0.22 across groups.

use crate::model::normalize_rows;
use crate::rng::RandomState;

/// Parameters of [`nested`].
#[derive(Clone, Debug)]
pub struct NestedSpec {
    pub n: usize,
    pub d: usize,
    pub n_broad: usize,
    /// Sub groups per broad group, drawn uniformly from this inclusive range.
    pub sub_range: (usize, usize),
    /// Fraction of rows that belong to no group.
    pub noise_frac: f64,
    /// Weight of the shared global direction in each broad center.
    pub global_weight: f32,
    /// Offset of a sub center from its broad center (norm, before normalizing).
    pub sub_spread: f32,
    /// Offset of a point from its sub center (norm, before normalizing).
    pub point_spread: f32,
    /// Lognormal sigma of broad and sub group weights (size inequality).
    pub size_sigma: f64,
    /// Smallest sub group.
    pub min_sub: usize,
    pub seed: u32,
}

impl Default for NestedSpec {
    fn default() -> Self {
        Self {
            n: 16_000,
            d: 768,
            n_broad: 8,
            sub_range: (3, 12),
            noise_frac: 0.03,
            global_weight: 0.82,
            sub_spread: 0.6,
            point_spread: 0.577,
            size_sigma: 0.7,
            min_sub: 15,
            seed: 7,
        }
    }
}

/// Rows of [`nested`] with their true labels.
#[derive(Clone, Debug)]
pub struct Nested {
    pub d: usize,
    /// `n x d` unit rows, shuffled.
    pub x: Vec<f32>,
    /// Broad group of each row; -1 for noise.
    pub broad: Vec<i32>,
    /// Global sub group id of each row; -1 for noise.
    pub sub: Vec<i32>,
    /// Sub groups in each broad group.
    pub subs_per_broad: Vec<usize>,
}

/// Normal draws from MT19937 (Box-Muller).
pub struct Normal(RandomState);

impl Normal {
    pub fn new(seed: u32) -> Self {
        Self(RandomState::new(seed))
    }

    pub fn sample(&mut self) -> f32 {
        let u1 = self.0.random_sample().max(1e-300);
        let u2 = self.0.random_sample();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    pub fn uniform(&mut self) -> f64 {
        self.0.random_sample()
    }

    /// A vector of `d` normal draws scaled to norm about 1 (N(0, I/d)).
    pub fn direction(&mut self, d: usize) -> Vec<f32> {
        let s = 1.0 / (d as f32).sqrt();
        (0..d).map(|_| self.sample() * s).collect()
    }

    pub fn permutation(&mut self, n: usize) -> Vec<u32> {
        self.0.permutation(n)
    }
}

fn unit(mut v: Vec<f32>, d: usize) -> Vec<f32> {
    normalize_rows(&mut v, d);
    v
}

fn add_scaled(a: &[f32], b: &[f32], s: f32) -> Vec<f32> {
    a.iter().zip(b).map(|(p, q)| p + s * q).collect()
}

/// Split `total` into parts proportional to `w`, each at least `min`.
fn allocate(total: usize, w: &[f64], min: usize) -> Vec<usize> {
    let s: f64 = w.iter().sum();
    let free = total.saturating_sub(min * w.len());
    let mut out: Vec<usize> = w
        .iter()
        .map(|v| min + (free as f64 * v / s).floor() as usize)
        .collect();
    let mut left = total.saturating_sub(out.iter().sum());
    let mut i = 0;
    while left > 0 {
        out[i % w.len()] += 1;
        left -= 1;
        i += 1;
    }
    out
}

/// Draw a nested mixture (see the module docs).
pub fn nested(spec: &NestedSpec) -> Nested {
    let d = spec.d;
    let mut rng = Normal::new(spec.seed);
    let g = unit(rng.direction(d), d);
    let n_noise = (spec.n as f64 * spec.noise_frac).round() as usize;
    let n_groups = spec.n - n_noise;
    let lognormal = |rng: &mut Normal, s: f64| (s * rng.sample() as f64).exp();
    let bw: Vec<f64> = (0..spec.n_broad)
        .map(|_| lognormal(&mut rng, spec.size_sigma))
        .collect();
    let mut subs_per_broad = Vec::new();
    let mut sub_w = Vec::new();
    for _ in 0..spec.n_broad {
        let (lo, hi) = spec.sub_range;
        let m = lo + (rng.uniform() * (hi - lo + 1) as f64).floor() as usize;
        let m = m.min(hi);
        subs_per_broad.push(m);
        sub_w.push(
            (0..m)
                .map(|_| lognormal(&mut rng, spec.size_sigma))
                .collect::<Vec<_>>(),
        );
    }
    // Sub-group sizes: broad weight times sub weight, each at least min_sub.
    let flat_w: Vec<f64> = bw
        .iter()
        .zip(&sub_w)
        .flat_map(|(b, ws)| {
            let s: f64 = ws.iter().sum();
            ws.iter().map(move |w| b * w / s).collect::<Vec<_>>()
        })
        .collect();
    let sizes = allocate(n_groups, &flat_w, spec.min_sub);

    let mut rows: Vec<(Vec<f32>, i32, i32)> = Vec::with_capacity(spec.n);
    let mut sid = 0usize;
    for (b, &m) in subs_per_broad.iter().enumerate() {
        let bc = unit(add_scaled(&rng.direction(d), &g, spec.global_weight), d);
        for _ in 0..m {
            let sc = unit(add_scaled(&bc, &rng.direction(d), spec.sub_spread), d);
            for _ in 0..sizes[sid] {
                let p = unit(add_scaled(&sc, &rng.direction(d), spec.point_spread), d);
                rows.push((p, b as i32, sid as i32));
            }
            sid += 1;
        }
    }
    for _ in 0..n_noise {
        // Noise: shares the global direction but belongs to no broad group.
        let c = unit(add_scaled(&rng.direction(d), &g, spec.global_weight), d);
        let p = unit(add_scaled(&c, &rng.direction(d), spec.point_spread), d);
        rows.push((p, -1, -1));
    }
    let perm = rng.permutation(rows.len());
    let mut x = Vec::with_capacity(spec.n * d);
    let mut broad = Vec::with_capacity(spec.n);
    let mut sub = Vec::with_capacity(spec.n);
    for &i in &perm {
        let (p, b, s) = &rows[i as usize];
        x.extend_from_slice(p);
        broad.push(*b);
        sub.push(*s);
    }
    Nested {
        d,
        x,
        broad,
        sub,
        subs_per_broad,
    }
}

/// Near-duplicate test data: `x` plus bursts of perturbed copies.
#[derive(Clone, Debug)]
pub struct WithBursts {
    /// Original rows first, then the copies.
    pub x: Vec<f32>,
    /// True near-duplicate set of every row: the original row id (singletons map to themselves).
    pub truth: Vec<u32>,
    /// Creation time of every row (seconds).
    pub created_unix: Vec<f64>,
}

/// Append `n_bursts` bursts: each takes a random original row and adds
/// `size_range` copies at cosine about `1 / (1 + eps^2)` from each other,
/// taken one second apart. Original rows get random times over five years.
pub fn with_bursts(
    x: &[f32],
    d: usize,
    n_bursts: usize,
    size_range: (usize, usize),
    eps: f32,
    seed: u32,
) -> WithBursts {
    let n = x.len() / d;
    let mut rng = Normal::new(seed);
    let mut out = x.to_vec();
    let mut truth: Vec<u32> = (0..n as u32).collect();
    let mut created: Vec<f64> = (0..n)
        .map(|_| rng.uniform() * 5.0 * 365.0 * 86400.0)
        .collect();
    let picks = rng.permutation(n);
    for &base in picks.iter().take(n_bursts) {
        let b = base as usize;
        let (lo, hi) = size_range;
        let m = lo + (rng.uniform() * (hi - lo + 1) as f64).floor() as usize;
        let m = m.min(hi);
        for t in 0..m {
            let p = unit(
                add_scaled(&x[b * d..(b + 1) * d], &rng.direction(d), eps),
                d,
            );
            out.extend_from_slice(&p);
            truth.push(base);
            created.push(created[b] + 1.0 + t as f64);
        }
    }
    WithBursts {
        x: out,
        truth,
        created_unix: created,
    }
}
