//! Scaling laws and the resolved training plan (faiss `base.py`).
//!
//! Equations refer to "E-Means: Scaling Laws for Mini-Batch K-Means",
//! arXiv 2611.09501. Constants are the shipped fits from faiss commit
//! 2b648f06, not the paper's Table 2.

use serde::{Deserialize, Serialize};

/// Epsilon in Equation 5: the cosine schedule's minimum step / initial step.
pub const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Logical batch updates between revival checks (Algorithm 1, line 9).
pub const CHECK_UNUSED_EVERY: usize = 5;
/// Initial EMA mass N_j (Algorithm 1, line 1). The paper uses 1e-10.
pub const INIT_USAGE: f32 = 1.0 / 65536.0;
/// Relative coordinate perturbation for donor splits (Appendix J, Eq. 8).
pub const SPLIT_EPSILON: f64 = 1e-4;
/// Revival threshold tau: revive N_j < tau * mean(N).
pub const UNUSED_THRESHOLD: f64 = 0.1;
/// `unused_threshold = None` disables revival at or below this dimension.
pub const NO_REVIVAL_MAX_D: usize = 2;
/// `batch_law = auto` picks the quality law at or below this dimension.
pub const AUTO_QUALITY_MAX_D: usize = 16;
/// Default rows per assignment chunk (faiss `assignment_chunk_size`).
pub const DEFAULT_ASSIGNMENT_CHUNK: usize = 65536;

/// Batch law B = C_B * k^b_k * n^b_n * T^b_T (Equation 7): (C_B, b_k, b_n, b_T).
const THROUGHPUT_LAW: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
const QUALITY_LAW: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
/// Step law alpha0 = C * k^a_k * B^a_B * n^a_n * T^a_T (Equation 6).
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BatchLaw {
    /// Quality law when d <= 16, throughput law above.
    #[default]
    Auto,
    Throughput,
    /// Batch about 8 times smaller than throughput.
    Quality,
}

impl BatchLaw {
    /// Resolve `Auto` from the dimension: (law, reason).
    pub fn resolve(self, d: usize) -> (BatchLaw, String) {
        match self {
            BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => {
                (BatchLaw::Quality, format!("d <= {AUTO_QUALITY_MAX_D}"))
            }
            BatchLaw::Auto => (BatchLaw::Throughput, format!("d > {AUTO_QUALITY_MAX_D}")),
            law => (law, "requested".into()),
        }
    }
}

impl std::str::FromStr for BatchLaw {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "auto" => Ok(BatchLaw::Auto),
            "throughput" => Ok(BatchLaw::Throughput),
            "quality" => Ok(BatchLaw::Quality),
            _ => anyhow::bail!("unknown batch law {s:?} (auto, throughput, quality)"),
        }
    }
}

/// Round a positive value to the nearest power of two (ties go up).
pub fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    (if value / low < high / value {
        low
    } else {
        high
    }) as usize
}

/// Initial EMA step coupled to a batch size by the fitted step law, clamped to [1e-4, 1].
pub fn step_law(n: usize, k: usize, nepochs: f64, batch_size: usize) -> f64 {
    let (c, ak, ab, an, at) = ALPHA_LAW;
    let a = c
        * (k as f64).powf(ak)
        * (batch_size as f64).powf(ab)
        * (n as f64).powf(an)
        * nepochs.powf(at);
    a.clamp(1e-4, 1.0)
}

/// Batch size B and initial step alpha0 (faiss `_resolve_recipe`).
///
/// `law` must already be resolved (not `Auto`) when `batch_size` is None.
pub fn resolve_batch_and_step(
    n: usize,
    k: usize,
    nepochs: f64,
    law: BatchLaw,
    batch_size: Option<usize>,
    alpha0: Option<f64>,
) -> (usize, f64) {
    let b = match batch_size {
        Some(b) => b.min(n),
        None => {
            let (c, bk, bn, bt) = match law {
                BatchLaw::Throughput => THROUGHPUT_LAW,
                BatchLaw::Quality => QUALITY_LAW,
                BatchLaw::Auto => panic!("resolve the batch law before the recipe"),
            };
            let raw = c * (k as f64).powf(bk) * (n as f64).powf(bn) * nepochs.powf(bt);
            let mut b = round_power_of_two(raw);
            // These intentionally differ: they reproduce the two fitted recipes.
            b = match law {
                BatchLaw::Throughput => b.min(round_power_of_two(n as f64 / 2.0)),
                _ => b.min(n / 2),
            };
            b.min(n).max(1)
        }
    };
    let a = alpha0.unwrap_or_else(|| step_law(n, k, nepochs, b));
    (b, a)
}

/// Rows per epoch: whole passes, a rounded fractional prefix, then the closing pass.
pub fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor();
    let remainder = ((nepochs - whole) * n as f64).round_ties_even() as usize;
    let whole = whole as usize;
    if whole == 0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole - 1];
    if remainder > 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// One reproducible permutation seed per epoch.
pub fn epoch_seed(seed: u32, epoch: usize) -> u32 {
    ((seed as u64 * 1_000_003 + epoch as u64) % (1u64 << 32)) as u32
}

/// Cosine-anneal an EMA step towards one tenth of its initial value (Equation 5).
pub fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Revival threshold tau, picked from `d` when not given.
pub fn resolve_unused_threshold(given: Option<f64>, d: usize) -> anyhow::Result<f64> {
    match given {
        None => Ok(if d <= NO_REVIVAL_MAX_D {
            0.0
        } else {
            UNUSED_THRESHOLD
        }),
        Some(t) if t >= 0.0 => Ok(t),
        Some(_) => anyhow::bail!("unused_threshold must be >= 0 (0 disables revival)"),
    }
}

/// The resolved training plan for one run (faiss `self.recipe`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    /// `canonical_v1`, `manual_batch` (batch given) or `manual` (batch and alpha0 given).
    pub name: String,
    /// Resolved law, or None when the batch size was given.
    pub batch_law: Option<BatchLaw>,
    pub batch_law_requested: Option<BatchLaw>,
    pub batch_law_reason: Option<String>,
    pub batch_size: usize,
    pub batch_size_requested: Option<usize>,
    pub alpha0: f64,
    /// `given` or `step_law`.
    pub alpha0_source: String,
    pub nepochs: f64,
    pub steps_per_pass: usize,
    pub total_steps: usize,
    pub epoch_rows: Vec<usize>,
    pub epoch_steps: Vec<usize>,
    pub alpha_min_factor: f64,
    pub check_unused_every: usize,
    pub unused_threshold: f64,
    pub split_epsilon: f64,
    pub init_usage: f32,
    pub seed: u32,
    pub backend: String,
    pub assignment_chunk_size: usize,
    pub spherical: bool,
    pub n: usize,
}

impl Recipe {
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        n: usize,
        k: usize,
        d: usize,
        nepochs: f64,
        batch_law: BatchLaw,
        batch_size: Option<usize>,
        alpha0: Option<f64>,
        unused_threshold: f64,
        seed: u32,
        spherical: bool,
        assignment_chunk_size: usize,
        backend: String,
    ) -> Recipe {
        let (law, reason) = batch_law.resolve(d);
        let (b, a0) = resolve_batch_and_step(n, k, nepochs, law, batch_size, alpha0);
        let epoch_rows = epoch_rows(n, nepochs);
        let epoch_steps: Vec<usize> = epoch_rows.iter().map(|r| r.div_ceil(b)).collect();
        let given = batch_size.is_some();
        Recipe {
            name: match (given, alpha0.is_some()) {
                (true, true) => "manual",
                (true, false) => "manual_batch",
                _ => "canonical_v1",
            }
            .into(),
            batch_law: (!given).then_some(law),
            batch_law_requested: (!given).then_some(batch_law),
            batch_law_reason: (!given).then_some(reason),
            batch_size: b,
            batch_size_requested: batch_size,
            alpha0: a0,
            alpha0_source: if alpha0.is_some() {
                "given"
            } else {
                "step_law"
            }
            .into(),
            nepochs,
            steps_per_pass: n.div_ceil(b),
            total_steps: epoch_steps.iter().sum(),
            epoch_rows,
            epoch_steps,
            alpha_min_factor: ALPHA_MIN_FACTOR,
            check_unused_every: CHECK_UNUSED_EVERY,
            unused_threshold,
            split_epsilon: SPLIT_EPSILON,
            init_usage: INIT_USAGE,
            seed,
            backend,
            assignment_chunk_size,
            spherical,
            n,
        }
    }
}
