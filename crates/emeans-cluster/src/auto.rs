//! Automatic k in two levels: broad groups, then leaves inside each group.
//!
//! e-means is fast, so the sweeps train many models and score each with
//! internal criteria from [`crate::metrics`] (one `n x k` product per run).
//! Each k is the best of `restarts` runs, each seeded by k-means++ with the
//! cosine distance. [`auto_cluster`] works in three stages:
//!
//! 1. **Leaves, top-down.** A level-1 sweep over `k1_range` takes the argmax
//!    of `criterion1` (silhouette), then a level-2 sweep runs inside each group
//!    of at least `min_split_size` rows, over `2..=min(k2_max, size /
//!    min_cluster_size)`. A group is split only when the best silhouette of its
//!    sweep reaches `split_min_silhouette`. Among the k whose `criterion2`
//!    (Davies-Bouldin) is within `level2_tolerance` of the best, it takes the
//!    one that puts the most rows into clusters of browsable size, then the
//!    smaller k.
//! 2. **Re-split.** Leaves above `max_cluster_size` get a cheap k = 2..4
//!    screen; when it shows structure, the leaf gets its own level-2 sweep.
//!    Every row then moves to its nearest leaf mean.
//! 3. **Level 1, bottom-up** (default, [`Level1Method::BottomUp`]). Average
//!    linkage on the leaves (exact mean row cosine), cut where `criterion1` of
//!    the row partition is best. "No grouping" is a candidate cut, so data
//!    without broad structure stays flat. [`Level1Method::TopDown`] keeps the
//!    groups of stage 1 instead: on nested data its sweep often lands on sub
//!    groups (k-means at the broad k splits big broad groups and merges small
//!    ones) or merges broad groups.
//!
//! `.okf/design/autok-incremental-dedup.md` has the comparisons behind these defaults.

use emeans_core::exec::Instant;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::assign::SharedAssigner;
use crate::dedup::DedupConfig;
use crate::incremental::{IncrementalConfig, State};
use crate::metrics::{Internal, internal_scores, quantile};
use crate::model::{Assignment, Config, Emeans};
use crate::{Backend, make_assigner};

/// Internal criterion that picks k.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Criterion {
    Silhouette,
    CalinskiHarabasz,
    DaviesBouldin,
}

impl Criterion {
    /// The criterion as "higher is better".
    pub fn value(self, s: &KScore) -> f64 {
        match self {
            Criterion::Silhouette => s.silhouette,
            Criterion::CalinskiHarabasz => s.calinski_harabasz,
            Criterion::DaviesBouldin => -s.davies_bouldin,
        }
    }
}

/// How each e-means run of a sweep picks its initial centroids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Init {
    /// k-means++ with the cosine distance (D^2 sampling). One seeding per
    /// restart serves every k of the sweep (its first k rows).
    KMeansPlusPlus,
    /// k random rows (e-means / faiss default).
    RandomRows,
}

/// How level 1 is formed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level1Method {
    /// Group the final leaves by average linkage (exact mean row cosine) and
    /// cut where `criterion1` of the row partition is best; "no grouping" is a
    /// candidate cut, so data without broad structure stays flat.
    #[default]
    BottomUp,
    /// Keep the groups of the top-down level-1 sweep.
    TopDown,
}

fn default_resplit_passes() -> usize {
    2
}

fn default_level1_max_share() -> f64 {
    0.5
}

/// CPU: the sweeps train hundreds of small models, where per-batch Metal
/// overhead dominates (4x slower than Accelerate at n = 4000, measured).
fn default_backend() -> Backend {
    Backend::Cpu(crate::CpuKernel::fastest())
}

/// Parameters of [`auto_cluster`]. Defaults are starting points to calibrate on real data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoConfig {
    /// Level-1 k sweep, inclusive. Default (2, 40).
    pub k1_range: (usize, usize),
    /// Largest level-2 k. Default 20.
    pub k2_max: usize,
    /// Criterion that picks the level-1 k. Default silhouette.
    pub criterion1: Criterion,
    /// Criterion that picks each level-2 k. Default Davies-Bouldin.
    pub criterion2: Criterion,
    /// Level 1 takes the smallest k within this relative slack of the best criterion. Default 0 (argmax).
    pub level1_tolerance: f64,
    /// Level 2 considers every k within this relative slack, then prefers browsable sizes. Default 0.05.
    pub level2_tolerance: f64,
    /// Smallest cluster size a person wants to browse; bounds the level-2 k. Default 20.
    pub min_cluster_size: usize,
    /// Largest browsable cluster; used by the level-2 size preference. Default 200.
    pub max_cluster_size: usize,
    /// Level-1 groups below this size are not split. Default 60.
    pub min_split_size: usize,
    /// Split a level-1 group only when its best silhouette reaches this. Default 0.05.
    pub split_min_silhouette: f64,
    /// e-means runs per k (different seeds); the lowest objective wins. Default 4.
    pub restarts: usize,
    /// e-means epochs per run. Default 5.
    pub nepochs: f64,
    /// Initial centroids of each run. Default k-means++.
    pub init: Init,
    /// How level 1 is formed from the leaves. Default bottom-up.
    #[serde(default)]
    pub level1_method: Level1Method,
    /// Passes that split again leaves above `max_cluster_size` with structure. Default 2 (0 = off).
    #[serde(default = "default_resplit_passes")]
    pub resplit_passes: usize,
    /// Bottom-up level 1: a cut whose largest group holds more than this share of
    /// the rows is not a candidate (average linkage chains into one giant group on
    /// data without broad structure). Default 0.5; 1.0 turns the guard off.
    #[serde(default = "default_level1_max_share")]
    pub level1_max_share: f64,
    pub seed: u32,
    /// Policy for [`Hierarchy::add`].
    pub incremental: IncrementalConfig,
    /// Print sweep progress to stderr.
    pub verbose: bool,
    /// Assignment backend; not saved (pass it to `Hierarchy::load`).
    #[serde(skip, default = "default_backend")]
    pub backend: Backend,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            k1_range: (2, 40),
            k2_max: 20,
            criterion1: Criterion::Silhouette,
            criterion2: Criterion::DaviesBouldin,
            level1_tolerance: 0.0,
            level2_tolerance: 0.05,
            min_cluster_size: 20,
            max_cluster_size: 200,
            min_split_size: 60,
            split_min_silhouette: 0.05,
            restarts: 4,
            nepochs: 5.0,
            init: Init::KMeansPlusPlus,
            level1_method: Level1Method::BottomUp,
            resplit_passes: 2,
            level1_max_share: default_level1_max_share(),
            seed: 1234,
            incremental: IncrementalConfig::default(),
            verbose: false,
            backend: default_backend(),
        }
    }
}

/// Scores of one k in a sweep.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KScore {
    pub k: usize,
    pub k_eff: usize,
    pub silhouette: f64,
    pub calinski_harabasz: f64,
    pub davies_bouldin: f64,
    pub objective: f64,
    /// Fraction of rows in clusters of `min_cluster_size..=max_cluster_size` rows.
    pub browsable: f64,
    /// Seconds for the restarts and scoring of this k.
    pub seconds: f64,
}

/// Why a level-1 group has the level-2 k it has.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupDecision {
    /// No rows at training time; the group has no level-2 model.
    Empty,
    /// Fewer than `min_split_size` rows, or too few for two clusters of `min_cluster_size`.
    TooSmall,
    /// The best silhouette of the sweep stayed below `split_min_silhouette`.
    NoStructure {
        best_silhouette: f64,
    },
    Split {
        k: usize,
    },
}

/// Sweep results.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Level-1 candidates: dendrogram cuts (bottom-up; the last entry is "no
    /// grouping") or the top-down k sweep.
    pub level1: Vec<KScore>,
    /// Level-2 sweep of each top-down group (stage A order); empty when not swept.
    pub level2: Vec<Vec<KScore>>,
    pub seconds: f64,
    /// The top-down level-1 k sweep that produced the first leaves.
    #[serde(default)]
    pub stage_a_level1: Vec<KScore>,
    /// Leaves split again because they were above `max_cluster_size` with structure.
    #[serde(default)]
    pub resplits: usize,
    /// Final number of leaves.
    #[serde(default)]
    pub leaves: usize,
    /// Number of groups at the largest drop in merge similarity (diagnostic only).
    #[serde(default)]
    pub gap_k: usize,
    /// The chosen cut is "no grouping": every leaf is its own level-1 group.
    #[serde(default)]
    pub flat: bool,
    /// Bottom-up cuts that were not candidates: largest group above `level1_max_share`.
    #[serde(default)]
    pub level1_rejected: Vec<usize>,
}

/// The k chosen at each level.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChosenK {
    pub k1: usize,
    /// Level-2 k per level-1 group (1 = not split; 0 = empty group).
    pub k2: Vec<usize>,
}

/// One level-1 group and its level-2 model.
pub struct Group {
    /// Level-2 e-means (k = 1 when the group is not split); None for an empty group.
    pub model: Option<Emeans>,
    /// Stable level-2 id of each level-2 centroid.
    pub ids: Vec<u32>,
    /// Outlier floor (cosine) of each level-2 centroid.
    pub floors: Vec<f32>,
    pub decision: GroupDecision,
    /// Next unused stable level-2 id.
    pub next_id: u32,
}

/// Labels of rows under a [`Hierarchy`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    /// Stable level-1 id.
    pub l1: Vec<u32>,
    /// Stable level-2 id inside the level-1 group.
    pub l2: Vec<u32>,
    /// Cosine to the level-1 centroid.
    pub sim1: Vec<f32>,
    /// Cosine to the leaf (level-2) centroid.
    pub sim: Vec<f32>,
    /// `sim` is below the leaf's outlier floor (far below what is typical for
    /// the leaf): show the row in a "new / unsorted" bucket.
    pub outlier: Vec<bool>,
    /// Level-1 centroid index (model slot).
    pub slot1: Vec<u32>,
    /// Level-2 centroid index inside the group's model.
    pub slot2: Vec<u32>,
}

impl Labels {
    pub fn len(&self) -> usize {
        self.l1.len()
    }

    pub fn is_empty(&self) -> bool {
        self.l1.is_empty()
    }

    /// Stable leaf id of row `i`.
    pub fn leaf(&self, i: usize) -> (u32, u32) {
        (self.l1[i], self.l2[i])
    }

    /// Leaf ids as one `u32` per row (`l1 * 65536 + l2`), for ARI/NMI.
    pub fn leaf_codes(&self) -> Vec<u32> {
        self.l1
            .iter()
            .zip(&self.l2)
            .map(|(a, b)| (a << 16) | (b & 0xffff))
            .collect()
    }

    /// The labels of rows `r`.
    pub fn slice(&self, r: std::ops::Range<usize>) -> Labels {
        Labels {
            l1: self.l1[r.clone()].to_vec(),
            l2: self.l2[r.clone()].to_vec(),
            sim1: self.sim1[r.clone()].to_vec(),
            sim: self.sim[r.clone()].to_vec(),
            outlier: self.outlier[r.clone()].to_vec(),
            slot1: self.slot1[r.clone()].to_vec(),
            slot2: self.slot2[r].to_vec(),
        }
    }

    /// Mean `2 - 2 cos` to the leaf centroids.
    pub fn objective(&self) -> f64 {
        self.sim.iter().map(|&s| 2.0 - 2.0 * s as f64).sum::<f64>() / self.sim.len().max(1) as f64
    }
}

/// A two-level clustering that can label new rows and update itself.
pub struct Hierarchy {
    pub d: usize,
    pub config: AutoConfig,
    /// Dedup parameters used upstream, kept with the model so later runs reuse them.
    pub dedup: Option<DedupConfig>,
    pub level1: Emeans,
    /// Stable level-1 id of each level-1 centroid.
    pub ids1: Vec<u32>,
    pub groups: Vec<Group>,
    pub diagnostics: Diagnostics,
    pub(crate) state: State,
    pub(crate) shared: SharedAssigner,
}

pub(crate) fn gather(x: &[f32], d: usize, rows: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * d);
    for &r in rows {
        out.extend_from_slice(&x[r as usize * d..(r as usize + 1) * d]);
    }
    out
}

fn seed_for(cfg: &AutoConfig, salt: u32, k: usize, restart: usize) -> u32 {
    cfg.seed
        .wrapping_add(salt.wrapping_mul(7919))
        .wrapping_add((k as u32).wrapping_mul(131))
        .wrapping_add((restart as u32).wrapping_mul(1_000_003))
}

/// Train one spherical e-means model and assign its rows.
pub(crate) fn fit(
    x: &[f32],
    d: usize,
    k: usize,
    nepochs: f64,
    seed: u32,
    init_rows: Option<&[u32]>,
    shared: &SharedAssigner,
) -> Result<(Emeans, Assignment)> {
    let mut c = Config::new(d, k);
    c.spherical = true;
    c.nepochs = nepochs;
    c.seed = seed;
    c.capture_assignments = false;
    let mut m = Emeans::with_assigner(c, Box::new(shared.clone()))?;
    match init_rows {
        Some(r) => {
            ensure!(r.len() >= k, "need k init rows");
            m.train_from(x, Some(&gather(x, d, &r[..k])))?
        }
        None => m.train(x)?,
    };
    let a = m.assign(x)?;
    Ok((m, a))
}

fn browsable(sizes: &[usize], cfg: &AutoConfig) -> f64 {
    let total: usize = sizes.iter().sum();
    let ok: usize = sizes
        .iter()
        .filter(|&&s| s >= cfg.min_cluster_size && s <= cfg.max_cluster_size)
        .sum();
    ok as f64 / total.max(1) as f64
}

fn score(k: usize, i: &Internal, cfg: &AutoConfig, seconds: f64) -> KScore {
    KScore {
        k,
        k_eff: i.k_eff,
        silhouette: i.silhouette,
        calinski_harabasz: i.calinski_harabasz,
        davies_bouldin: i.davies_bouldin,
        objective: i.objective,
        browsable: browsable(&i.sizes, cfg),
        seconds,
    }
}

/// k-means++ seeding with the cosine distance `1 - x.y`: `kmax` row ids, in
/// pick order, so the first k of them seed a run with k clusters.
pub fn kmeans_pp(x: &[f32], d: usize, kmax: usize, seed: u32) -> Vec<u32> {
    use rayon::prelude::*;
    let n = x.len() / d;
    let kmax = kmax.min(n);
    let mut rng = crate::rng::RandomState::new(seed);
    let mut picks = Vec::with_capacity(kmax);
    if kmax == 0 {
        return picks;
    }
    let first = rng.random_interval(n as u64 - 1) as usize;
    picks.push(first as u32);
    let mut best = vec![f32::NEG_INFINITY; n];
    let mut last = first;
    while picks.len() < kmax {
        let c = &x[last * d..(last + 1) * d];
        best.par_iter_mut().zip(x.par_chunks(d)).for_each(|(b, r)| {
            let s: f32 = r.iter().zip(c).map(|(p, q)| p * q).sum();
            if s > *b {
                *b = s;
            }
        });
        let w: Vec<f64> = best
            .iter()
            .map(|&s| {
                let dist = (1.0 - s as f64).max(0.0);
                dist * dist
            })
            .collect();
        let total: f64 = w.iter().sum();
        let next = if total > 0.0 {
            let mut u = rng.random_sample() * total;
            let mut j = n - 1;
            for (i, &v) in w.iter().enumerate() {
                if u < v {
                    j = i;
                    break;
                }
                u -= v;
            }
            // Rounding can land on a zero-weight row; take the farthest instead.
            if w[j] == 0.0 {
                (0..n).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap()
            } else {
                j
            }
        } else {
            // All rows coincide with a pick: any unpicked row.
            (0..n).find(|i| !picks.contains(&(*i as u32))).unwrap_or(0)
        };
        picks.push(next as u32);
        last = next;
    }
    picks
}

/// Run every k in `ks` (best of `restarts`) and score it.
pub(crate) fn sweep(
    x: &[f32],
    d: usize,
    ks: &[usize],
    cfg: &AutoConfig,
    salt: u32,
    shared: &SharedAssigner,
) -> Result<Vec<(KScore, Emeans, Assignment)>> {
    let mut out = Vec::with_capacity(ks.len());
    // One seeding per restart, up to the largest k: its first k rows seed k.
    let kmax = ks.iter().copied().max().unwrap_or(1);
    let inits: Vec<Option<Vec<u32>>> = (0..cfg.restarts.max(1))
        .map(|r| match cfg.init {
            Init::KMeansPlusPlus => Some(kmeans_pp(x, d, kmax, seed_for(cfg, salt, 0, r))),
            Init::RandomRows => None,
        })
        .collect();
    for &k in ks {
        let t = Instant::now();
        let mut best: Option<(Emeans, Assignment, f64)> = None;
        for (r, seeds) in inits.iter().enumerate() {
            let (m, a) = fit(
                x,
                d,
                k,
                cfg.nepochs,
                seed_for(cfg, salt, k, r),
                seeds.as_deref(),
                shared,
            )?;
            let obj = a.mean_objective(true);
            if best.as_ref().is_none_or(|b| obj < b.2) {
                best = Some((m, a, obj));
            }
        }
        let (m, a, _) = best.unwrap();
        let i = internal_scores(x, d, &a.labels, k)?;
        let s = score(k, &i, cfg, t.elapsed().as_secs_f64());
        if cfg.verbose {
            eprintln!(
                "  k={k:3} k_eff={:3} sil={:.4} ch={:.1} db={:.3} obj={:.4} browsable={:.2} ({:.2}s)",
                s.k_eff,
                s.silhouette,
                s.calinski_harabasz,
                s.davies_bouldin,
                s.objective,
                s.browsable,
                s.seconds
            );
        }
        out.push((s, m, a));
    }
    Ok(out)
}

/// Index of the chosen k in `scores`.
pub fn select(scores: &[KScore], cfg: &AutoConfig, prefer_browsable: bool) -> usize {
    let crit = if prefer_browsable {
        cfg.criterion2
    } else {
        cfg.criterion1
    };
    let v: Vec<f64> = scores.iter().map(|s| crit.value(s)).collect();
    let best = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tol = if prefer_browsable {
        cfg.level2_tolerance
    } else {
        cfg.level1_tolerance
    };
    let floor = best - tol * best.abs();
    let cands: Vec<usize> = (0..scores.len()).filter(|&i| v[i] >= floor).collect();
    if prefer_browsable {
        *cands
            .iter()
            .max_by(|&&a, &&b| {
                scores[a]
                    .browsable
                    .total_cmp(&scores[b].browsable)
                    .then(scores[b].k.cmp(&scores[a].k))
            })
            .unwrap()
    } else {
        *cands.iter().min_by_key(|&&i| scores[i].k).unwrap()
    }
}

/// Fit the level-2 model of one level-1 group (rows `xg`).
pub(crate) fn fit_group(
    xg: &[f32],
    d: usize,
    cfg: &AutoConfig,
    salt: u32,
    shared: &SharedAssigner,
) -> Result<(Option<Emeans>, GroupDecision, Vec<KScore>)> {
    let n = xg.len() / d;
    if n == 0 {
        return Ok((None, GroupDecision::Empty, Vec::new()));
    }
    let one = |shared: &SharedAssigner| -> Result<Emeans> {
        Ok(fit(
            xg,
            d,
            1,
            cfg.nepochs,
            seed_for(cfg, salt, 1, 0),
            None,
            shared,
        )?
        .0)
    };
    let kmax = cfg.k2_max.min(n / cfg.min_cluster_size.max(1));
    if n < cfg.min_split_size || kmax < 2 {
        return Ok((Some(one(shared)?), GroupDecision::TooSmall, Vec::new()));
    }
    let ks: Vec<usize> = (2..=kmax).collect();
    let runs = sweep(xg, d, &ks, cfg, salt, shared)?;
    let scores: Vec<KScore> = runs.iter().map(|r| r.0.clone()).collect();
    let best_sil = scores
        .iter()
        .map(|s| s.silhouette)
        .fold(f64::NEG_INFINITY, f64::max);
    if best_sil < cfg.split_min_silhouette {
        return Ok((
            Some(one(shared)?),
            GroupDecision::NoStructure {
                best_silhouette: best_sil,
            },
            scores,
        ));
    }
    let pick = select(&scores, cfg, true);
    let k = scores[pick].k;
    let m = runs.into_iter().nth(pick).unwrap().1;
    Ok((Some(m), GroupDecision::Split { k }, scores))
}

/// Leaves of the hierarchy under construction: rows per leaf and the level-1
/// group each leaf came from in the top-down stage.
struct Leaves {
    rows: Vec<Vec<u32>>,
    parent: Vec<usize>,
    /// Best silhouette of the last split sweep run inside the leaf, if any.
    sweep: Vec<Option<f64>>,
}

/// Stage A, top-down: level-1 sweep, then a level-2 sweep inside each group.
fn stage_a(
    x: &[f32],
    d: usize,
    cfg: &AutoConfig,
    shared: &SharedAssigner,
    diag: &mut Diagnostics,
) -> Result<Leaves> {
    let n = x.len() / d;
    let k1_hi = cfg.k1_range.1.min(n);
    let k1_lo = cfg.k1_range.0.min(k1_hi);
    let ks: Vec<usize> = (k1_lo..=k1_hi).collect();
    if cfg.verbose {
        eprintln!("stage A level 1: n={n}, k in {k1_lo}..={k1_hi}");
    }
    let runs = sweep(x, d, &ks, cfg, 0, shared)?;
    let scores: Vec<KScore> = runs.iter().map(|r| r.0.clone()).collect();
    let pick = select(&scores, cfg, false);
    let (_, level1, a1) = runs.into_iter().nth(pick).unwrap();
    let k1 = level1.config().k;
    diag.stage_a_level1 = scores;
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); k1];
    for (i, &l) in a1.labels.iter().enumerate() {
        rows[l as usize].push(i as u32);
    }
    let mut leaves = Leaves {
        rows: Vec::new(),
        parent: Vec::new(),
        sweep: Vec::new(),
    };
    for (g, r) in rows.iter().enumerate() {
        if r.is_empty() {
            diag.level2.push(Vec::new());
            continue;
        }
        let xg = gather(x, d, r);
        let (model, decision, sc) = fit_group(&xg, d, cfg, 1 + g as u32, shared)?;
        let best = sc
            .iter()
            .map(|s| s.silhouette)
            .fold(f64::NEG_INFINITY, f64::max);
        diag.level2.push(sc);
        let model = model.expect("non-empty group has a model");
        let k2 = model.config().k;
        let mut sub: Vec<Vec<u32>> = vec![Vec::new(); k2];
        for (&i, &j) in r.iter().zip(&model.assign(&xg)?.labels) {
            sub[j as usize].push(i);
        }
        for s in sub.into_iter().filter(|s| !s.is_empty()) {
            leaves.rows.push(s);
            leaves.parent.push(g);
            leaves.sweep.push(match decision {
                GroupDecision::Split { .. } => None,
                _ => best.is_finite().then_some(best),
            });
        }
    }
    Ok(leaves)
}

/// Largest k of the cheap structure screen in [`resplit`].
const SCREEN_K: usize = 4;

/// Split again every leaf above `max_cluster_size` whose own sweep shows structure.
fn resplit(
    x: &[f32],
    d: usize,
    cfg: &AutoConfig,
    shared: &SharedAssigner,
    leaves: Leaves,
) -> Result<(Leaves, usize)> {
    let mut cur = leaves;
    let mut splits = 0;
    for pass in 0..cfg.resplit_passes {
        let mut next = Leaves {
            rows: Vec::new(),
            parent: Vec::new(),
            sweep: Vec::new(),
        };
        let mut changed = false;
        for (li, ((r, p), sw)) in cur
            .rows
            .into_iter()
            .zip(cur.parent)
            .zip(cur.sweep)
            .enumerate()
        {
            // Small leaves, and leaves whose own sweep already found no
            // structure, stay as they are.
            let swept_flat = sw.is_some_and(|b| b < cfg.split_min_silhouette);
            if r.len() <= cfg.max_cluster_size || r.len() < cfg.min_split_size || swept_flat {
                next.rows.push(r);
                next.parent.push(p);
                next.sweep.push(sw);
                continue;
            }
            let xg = gather(x, d, &r);
            let salt = 50_000 + 1000 * pass as u32 + li as u32;
            // Screen with k = 2..=4 first: most large leaves are one true
            // group, and the full sweep costs about 6x more.
            let kmax = cfg.k2_max.min(r.len() / cfg.min_cluster_size.max(1));
            let screen: Vec<usize> = (2..=kmax.min(SCREEN_K)).collect();
            let screen_best = sweep(&xg, d, &screen, cfg, salt, shared)?
                .iter()
                .map(|s| s.0.silhouette)
                .fold(f64::NEG_INFINITY, f64::max);
            if screen.is_empty() || screen_best < cfg.split_min_silhouette {
                next.rows.push(r);
                next.parent.push(p);
                next.sweep
                    .push(screen_best.is_finite().then_some(screen_best));
                continue;
            }
            let (model, decision, sc) = fit_group(&xg, d, cfg, salt, shared)?;
            let best = sc
                .iter()
                .map(|s| s.silhouette)
                .fold(f64::NEG_INFINITY, f64::max);
            match (decision, model) {
                (GroupDecision::Split { k }, Some(m)) => {
                    let mut sub: Vec<Vec<u32>> = vec![Vec::new(); k];
                    for (&i, &j) in r.iter().zip(&m.assign(&xg)?.labels) {
                        sub[j as usize].push(i);
                    }
                    for s in sub.into_iter().filter(|s| !s.is_empty()) {
                        next.rows.push(s);
                        next.parent.push(p);
                        next.sweep.push(None);
                    }
                    splits += 1;
                    changed = true;
                }
                _ => {
                    next.rows.push(r);
                    next.parent.push(p);
                    next.sweep.push(best.is_finite().then_some(best));
                }
            }
        }
        cur = next;
        if !changed {
            break;
        }
    }
    Ok((cur, splits))
}

/// Normalized means of the rows of each cluster (`k x d`).
pub(crate) fn unit_means(x: &[f32], d: usize, rows: &[Vec<u32>]) -> Vec<f32> {
    let mut c = vec![0f32; rows.len() * d];
    for (dst, r) in c.chunks_mut(d).zip(rows) {
        let mut acc = vec![0f64; d];
        for &i in r {
            for (a, &v) in acc.iter_mut().zip(&x[i as usize * d..(i as usize + 1) * d]) {
                *a += v as f64;
            }
        }
        for (o, a) in dst.iter_mut().zip(&acc) {
            *o = *a as f32;
        }
    }
    crate::model::normalize_rows(&mut c, d);
    c
}

/// Nearest leaf of every row (cosine), with one more pass on the means.
fn refine(x: &[f32], d: usize, leaves: Leaves) -> Result<Leaves> {
    let n = x.len() / d;
    let mut cur = leaves;
    for _ in 0..2 {
        let c = unit_means(x, d, &cur.rows);
        let mut a = crate::assign::CpuAssigner::default();
        let (lab, _) = crate::assign::assign_all(
            &mut a,
            crate::Rows::Dense(x),
            d,
            &c,
            crate::Metric::InnerProduct,
            1 << 16,
        )?;
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); cur.rows.len()];
        for (i, &l) in lab.iter().enumerate().take(n) {
            rows[l as usize].push(i as u32);
        }
        let keep: Vec<usize> = (0..rows.len()).filter(|&j| !rows[j].is_empty()).collect();
        cur = Leaves {
            rows: keep.iter().map(|&j| std::mem::take(&mut rows[j])).collect(),
            parent: keep.iter().map(|&j| cur.parent[j]).collect(),
            sweep: keep.iter().map(|&j| cur.sweep[j]).collect(),
        };
    }
    Ok(cur)
}

/// Average linkage over the leaves with the exact mean row cosine
/// `s_A . s_B / (|A| |B|)`; returns the merges as (a, b, similarity), where the
/// merged cluster keeps id `a`.
fn average_linkage(x: &[f32], d: usize, rows: &[Vec<u32>]) -> Vec<(usize, usize, f64)> {
    let l = rows.len();
    let mut sums: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let mut s = vec![0f64; d];
            for &i in r {
                for (a, &v) in s.iter_mut().zip(&x[i as usize * d..(i as usize + 1) * d]) {
                    *a += v as f64;
                }
            }
            s
        })
        .collect();
    let mut size: Vec<f64> = rows.iter().map(|r| r.len() as f64).collect();
    // dots[a][b] = s_a . s_b; bilinear, so a merge adds rows.
    let mut dots = vec![vec![0f64; l]; l];
    for a in 0..l {
        for b in a..l {
            let v: f64 = sums[a].iter().zip(&sums[b]).map(|(p, q)| p * q).sum();
            dots[a][b] = v;
            dots[b][a] = v;
        }
    }
    let mut alive: Vec<bool> = vec![true; l];
    let mut merges = Vec::with_capacity(l.saturating_sub(1));
    for _ in 1..l {
        let mut best = (f64::NEG_INFINITY, 0, 0);
        for a in 0..l {
            if !alive[a] {
                continue;
            }
            for b in a + 1..l {
                if alive[b] {
                    let s = dots[a][b] / (size[a] * size[b]);
                    if s > best.0 {
                        best = (s, a, b);
                    }
                }
            }
        }
        let (s, a, b) = best;
        alive[b] = false;
        size[a] += size[b];
        let sb = std::mem::take(&mut sums[b]);
        for (p, q) in sums[a].iter_mut().zip(&sb) {
            *p += q;
        }
        // dots is symmetric: column a becomes row a + row b.
        let merged: Vec<f64> = dots[a].iter().zip(&dots[b]).map(|(p, q)| p + q).collect();
        for (row, &v) in dots.iter_mut().zip(&merged) {
            row[a] = v;
        }
        dots[a] = merged;
        dots[a][a] = sums[a].iter().map(|v| v * v).sum();
        merges.push((a, b, s));
    }
    merges
}

/// Group of each leaf after the first `l - k` merges, numbered 0..k by lowest leaf.
fn cut(l: usize, merges: &[(usize, usize, f64)], k: usize) -> Vec<usize> {
    let mut owner: Vec<usize> = (0..l).collect();
    for &(a, b, _) in &merges[..l - k] {
        for o in owner.iter_mut() {
            if *o == b {
                *o = a;
            }
        }
    }
    let mut ids = std::collections::BTreeMap::new();
    for &o in &owner {
        let next = ids.len();
        ids.entry(o).or_insert(next);
    }
    owner.iter().map(|o| ids[o]).collect()
}

/// Stage C, bottom-up: group the leaves by average linkage and pick the cut
/// whose row partition scores best on `criterion1`. Cutting at k = number of
/// leaves (no grouping) is a candidate, so flat data stays flat.
fn group_bottom_up(
    x: &[f32],
    d: usize,
    cfg: &AutoConfig,
    leaves: &Leaves,
    diag: &mut Diagnostics,
) -> Result<Vec<usize>> {
    let l = leaves.rows.len();
    if l <= 2 {
        return Ok((0..l).collect());
    }
    let merges = average_linkage(x, d, &leaves.rows);
    let n = x.len() / d;
    let mut leaf_of = vec![0usize; n];
    for (j, r) in leaves.rows.iter().enumerate() {
        for &i in r {
            leaf_of[i as usize] = j;
        }
    }
    let hi = cfg.k1_range.1.min(l - 1);
    let lo = cfg.k1_range.0.max(2).min(hi);
    let mut ks: Vec<usize> = (lo..=hi).collect();
    ks.push(l);
    let mut scores = Vec::with_capacity(ks.len());
    let mut eligible = Vec::with_capacity(ks.len());
    for &k in &ks {
        let t = Instant::now();
        let g = cut(l, &merges, k);
        let labels: Vec<u32> = leaf_of.iter().map(|&j| g[j] as u32).collect();
        let mut sizes = vec![0usize; k];
        for &c in &labels {
            sizes[c as usize] += 1;
        }
        let largest = sizes.iter().copied().max().unwrap_or(0) as f64 / n.max(1) as f64;
        // "No grouping" (k = l) is always a candidate.
        eligible.push(k == l || largest <= cfg.level1_max_share);
        let i = internal_scores(x, d, &labels, k)?;
        scores.push(score(k, &i, cfg, t.elapsed().as_secs_f64()));
    }
    // Largest drop in merge similarity, as a diagnostic.
    diag.gap_k = (0..merges.len().saturating_sub(1))
        .max_by(|&a, &b| {
            let ga = merges[a].2 - merges[a + 1].2;
            let gb = merges[b].2 - merges[b + 1].2;
            ga.total_cmp(&gb)
        })
        .map_or(1, |t| l - 1 - t);
    let v: Vec<f64> = scores
        .iter()
        .zip(&eligible)
        .map(|(s, &ok)| {
            if ok {
                cfg.criterion1.value(s)
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let best = (0..ks.len())
        .max_by(|&a, &b| v[a].total_cmp(&v[b]).then(ks[b].cmp(&ks[a])))
        .unwrap();
    diag.flat = ks[best] == l;
    diag.level1_rejected = ks
        .iter()
        .zip(&eligible)
        .filter(|(_, ok)| !**ok)
        .map(|(&k, _)| k)
        .collect();
    diag.level1 = scores;
    Ok(cut(l, &merges, ks[best]))
}

/// An e-means model whose centroids are `centroids` (unit rows), with EMA
/// masses at the batch scale of a run on `n` rows, and revival off so updates
/// keep cluster slots.
pub(crate) fn model_from(
    centroids: &[f32],
    counts: &[usize],
    d: usize,
    cfg: &AutoConfig,
    shared: &SharedAssigner,
) -> Result<Emeans> {
    let k = counts.len();
    let n: usize = counts.iter().sum::<usize>().max(1);
    let mut c = Config::new(d, k);
    c.spherical = true;
    c.nepochs = cfg.nepochs;
    c.seed = cfg.seed;
    c.capture_assignments = false;
    c.unused_threshold = Some(0.0);
    let (law, _) = c.batch_law.resolve(d);
    let (b, _) = crate::recipe::resolve_batch_and_step(n, k, c.nepochs, law, None, None);
    let usage: Vec<f32> = counts
        .iter()
        .map(|&m| (m as f64 * b as f64 / n as f64).max(1e-3) as f32)
        .collect();
    Emeans::from_snapshot(
        crate::io::Snapshot {
            config: c,
            recipe: None,
            n_splits: 0,
            iteration_stats: Vec::new(),
            centroids: Some(centroids.to_vec()),
            positions: centroids.to_vec(),
            usage,
        },
        Box::new(shared.clone()),
    )
}

/// Two-level clustering of the unit rows of `x` (`n x d`) with automatic k.
///
/// Returns the hierarchy and the labels of the rows of `x`. Pass the dedup
/// representatives as `x`, not every photo, so bursts do not form clusters.
///
/// 1. Leaves: a top-down pass (level-1 sweep, level-2 sweep per group), then
///    leaves above `max_cluster_size` are split again when they have structure,
///    and every row moves to its nearest leaf.
/// 2. Level 1: `Level1Method::BottomUp` (default) groups the leaves by average
///    linkage and picks the cut by `criterion1`; `TopDown` keeps the groups of
///    the top-down pass.
pub fn auto_cluster(x: &[f32], d: usize, cfg: &AutoConfig) -> Result<(Hierarchy, Labels)> {
    ensure!(d > 0 && x.len().is_multiple_of(d), "x must be n x d");
    let n = x.len() / d;
    ensure!(n >= 2, "need at least 2 rows");
    ensure!(
        cfg.k1_range.0 >= 1 && cfg.k1_range.0 <= cfg.k1_range.1,
        "bad k1_range"
    );
    let t = Instant::now();
    let shared = SharedAssigner::new(make_assigner(cfg.backend)?);
    let mut diag = Diagnostics::default();
    let lap = |what: &str| {
        if cfg.verbose {
            eprintln!("[{:.1}s] {what}", t.elapsed().as_secs_f64());
        }
    };
    let leaves = stage_a(x, d, cfg, &shared, &mut diag)?;
    lap("stage A");
    let (leaves, resplits) = resplit(x, d, cfg, &shared, leaves)?;
    lap("resplit");
    let leaves = refine(x, d, leaves)?;
    lap("refine");
    diag.resplits = resplits;
    diag.leaves = leaves.rows.len();
    let group_of_leaf: Vec<usize> = match cfg.level1_method {
        Level1Method::BottomUp => group_bottom_up(x, d, cfg, &leaves, &mut diag)?,
        Level1Method::TopDown => {
            let mut ids = std::collections::BTreeMap::new();
            for &p in &leaves.parent {
                let next = ids.len();
                ids.entry(p).or_insert(next);
            }
            diag.level1 = diag.stage_a_level1.clone();
            leaves.parent.iter().map(|p| ids[p]).collect()
        }
    };
    lap("level 1");
    let k1 = group_of_leaf.iter().max().map_or(0, |m| m + 1);
    if cfg.verbose {
        eprintln!(
            "leaves {} (resplits {resplits}), level-1 groups {k1}{}",
            leaves.rows.len(),
            if diag.flat { " (flat)" } else { "" }
        );
    }

    // Assemble the models from the leaves.
    let mut group_rows: Vec<Vec<u32>> = vec![Vec::new(); k1];
    let mut groups = Vec::with_capacity(k1);
    for (g, grp_rows) in group_rows.iter_mut().enumerate() {
        let members: Vec<usize> = (0..leaves.rows.len())
            .filter(|&j| group_of_leaf[j] == g)
            .collect();
        let rows: Vec<Vec<u32>> = members.iter().map(|&j| leaves.rows[j].clone()).collect();
        let counts: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        let model = model_from(&unit_means(x, d, &rows), &counts, d, cfg, &shared)?;
        let size: usize = counts.iter().sum();
        let decision = if members.len() >= 2 {
            GroupDecision::Split { k: members.len() }
        } else {
            match leaves.sweep[members[0]] {
                Some(b) if b < cfg.split_min_silhouette => {
                    GroupDecision::NoStructure { best_silhouette: b }
                }
                _ if size < cfg.min_split_size || size <= cfg.max_cluster_size => {
                    GroupDecision::TooSmall
                }
                Some(b) => GroupDecision::NoStructure { best_silhouette: b },
                None => GroupDecision::TooSmall,
            }
        };
        for r in &rows {
            grp_rows.extend_from_slice(r);
        }
        let k2 = members.len();
        groups.push(Group {
            model: Some(model),
            ids: (0..k2 as u32).collect(),
            floors: vec![f32::NAN; k2],
            decision,
            next_id: k2 as u32,
        });
    }
    let counts1: Vec<usize> = group_rows.iter().map(|r| r.len()).collect();
    let level1 = model_from(&unit_means(x, d, &group_rows), &counts1, d, cfg, &shared)?;
    let mut h = Hierarchy {
        d,
        config: cfg.clone(),
        dedup: None,
        level1,
        ids1: (0..k1 as u32).collect(),
        groups,
        diagnostics: diag,
        state: State::new(n, k1 as u32),
        shared,
    };
    let labels = h.assign(x)?;
    h.set_floors(&labels);
    let labels = h.assign(x)?;
    h.diagnostics.seconds = t.elapsed().as_secs_f64();
    Ok((h, labels))
}

impl Hierarchy {
    pub fn chosen_k(&self) -> ChosenK {
        ChosenK {
            k1: self.level1.config().k,
            k2: self
                .groups
                .iter()
                .map(|g| g.model.as_ref().map_or(0, |m| m.config().k))
                .collect(),
        }
    }

    /// Level-1 centroids, `k1 x d` (slot order): the normalized mean of each group's rows.
    pub fn level1_centroids(&self) -> &[f32] {
        self.level1.centroids().expect("trained level 1")
    }

    /// Leaf centroids of level-1 slot `g` (`k2 x d`); the level-1 centroid when the group is empty.
    pub fn leaf_centroids(&self, g: usize) -> Vec<f32> {
        match &self.groups[g].model {
            Some(m) => m.centroids().expect("trained level 2").to_vec(),
            None => self.level1_centroids()[g * self.d..(g + 1) * self.d].to_vec(),
        }
    }

    /// Rows the model has seen (trained on or added).
    pub fn n_seen(&self) -> usize {
        self.state.n_seen
    }

    /// Label rows: nearest leaf over all leaves (cosine); level 1 is that leaf's group.
    pub fn assign(&self, x: &[f32]) -> Result<Labels> {
        let d = self.d;
        ensure!(x.len().is_multiple_of(d), "x must be n x d");
        let n = x.len() / d;
        let mut cents = Vec::new();
        let mut slot = Vec::new();
        for (g, grp) in self.groups.iter().enumerate() {
            if let Some(m) = &grp.model {
                let c = m.centroids().context("level-2 centroids")?;
                cents.extend_from_slice(c);
                slot.extend((0..m.config().k).map(|j| (g as u32, j as u32)));
            }
        }
        ensure!(!slot.is_empty(), "hierarchy has no leaves");
        let mut a = self.shared.clone();
        let (leaf, sims) = crate::assign::assign_all(
            &mut a,
            crate::Rows::Dense(x),
            d,
            &cents,
            crate::Metric::InnerProduct,
            1 << 16,
        )?;
        let c1 = self.level1_centroids();
        let mut out = Labels {
            l1: vec![0; n],
            l2: vec![0; n],
            sim1: vec![0.0; n],
            sim: sims,
            outlier: vec![false; n],
            slot1: vec![0; n],
            slot2: vec![0; n],
        };
        for i in 0..n {
            let (g, j) = slot[leaf[i] as usize];
            let grp = &self.groups[g as usize];
            out.slot1[i] = g;
            out.slot2[i] = j;
            out.l1[i] = self.ids1[g as usize];
            out.l2[i] = grp.ids.get(j as usize).copied().unwrap_or(0);
            let r = &x[i * d..(i + 1) * d];
            out.sim1[i] = r
                .iter()
                .zip(&c1[g as usize * d..(g as usize + 1) * d])
                .map(|(p, q)| p * q)
                .sum();
            let f = grp.floors.get(j as usize).copied().unwrap_or(f32::NAN);
            out.outlier[i] = f.is_finite() && out.sim[i] < f;
        }
        Ok(out)
    }

    /// Cosines to the leaf centroid of the rows of every leaf, with the
    /// `(group slot, leaf slot)` of each entry, in group then leaf order.
    pub(crate) fn leaf_sims(&self, labels: &Labels) -> (Vec<(usize, usize)>, Vec<Vec<f32>>) {
        let mut at = Vec::new();
        let mut first = Vec::with_capacity(self.groups.len());
        for (g, grp) in self.groups.iter().enumerate() {
            first.push(at.len());
            at.extend((0..grp.ids.len()).map(|j| (g, j)));
        }
        let mut sims = vec![Vec::new(); at.len()];
        for i in 0..labels.len() {
            let (g, j) = (labels.slot1[i] as usize, labels.slot2[i] as usize);
            if j < self.groups[g].ids.len() {
                sims[first[g] + j].push(labels.sim[i]);
            }
        }
        (at, sims)
    }

    /// Recompute every leaf's outlier floor from the labels of the rows it was fit on.
    pub(crate) fn set_floors(&mut self, labels: &Labels) {
        let (at, sims) = self.leaf_sims(labels);
        let floors = leaf_floors(&sims, &self.config.incremental);
        for grp in self.groups.iter_mut() {
            grp.floors = vec![f32::NAN; grp.ids.len()];
        }
        for (&(g, j), f) in at.iter().zip(floors) {
            self.groups[g].floors[j] = f;
        }
    }
}

/// Rows a leaf needs before its own spread sets its floor; a smaller leaf
/// uses the spread pooled over all leaves.
pub(crate) const OWN_SPREAD_MIN: usize = 20;

/// Normal-consistency factor of the median absolute deviation.
const MAD_SIGMA: f32 = 1.4826;

/// Robust location and spread of the member cosines of every leaf.
pub(crate) struct Spread {
    /// Median member cosine per leaf; NaN for an empty leaf.
    pub median: Vec<f32>,
    /// Median absolute deviation from the leaf median, per leaf.
    pub mad: Vec<f32>,
    /// Median absolute deviation from the own leaf's median over all rows.
    pub pooled_mad: f32,
    /// Median cosine over all rows.
    pub global_median: f32,
}

pub(crate) fn spread(sims: &[Vec<f32>]) -> Spread {
    let median: Vec<f32> = sims.iter().map(|v| quantile(v, 0.5)).collect();
    let mut all = Vec::new();
    let mut residuals = Vec::new();
    let mut mad = Vec::with_capacity(sims.len());
    for (v, &m) in sims.iter().zip(&median) {
        let r: Vec<f32> = v.iter().map(|s| (s - m).abs()).collect();
        mad.push(quantile(&r, 0.5));
        residuals.extend(r);
        all.extend_from_slice(v);
    }
    Spread {
        median,
        mad,
        pooled_mad: quantile(&residuals, 0.5),
        global_median: quantile(&all, 0.5),
    }
}

/// Outlier floor of every leaf from its members' cosines at a fit.
///
/// `median - outlier_mad_k * 1.4826 * MAD`: a row is an outlier when it is far
/// below what is typical for its leaf, measured by the leaf's own robust
/// spread. Unlike a quantile of the members, this does not flag a fixed share
/// of every leaf: a tight leaf with no stragglers flags no row, and a leaf of
/// any size can flag one. Leaves under [`OWN_SPREAD_MIN`] rows (or with zero
/// spread) use the spread pooled over all leaves; an empty leaf uses the
/// global median. `outlier_min_sim` replaces the rule with one absolute floor.
pub(crate) fn leaf_floors(sims: &[Vec<f32>], inc: &IncrementalConfig) -> Vec<f32> {
    if let Some(t) = inc.outlier_min_sim {
        return vec![t; sims.len()];
    }
    let sp = spread(sims);
    let k = inc.outlier_mad_k as f32;
    sims.iter()
        .enumerate()
        .map(|(j, v)| {
            if v.is_empty() {
                return sp.global_median - k * MAD_SIGMA * sp.pooled_mad;
            }
            let own = sp.mad[j];
            let s = if v.len() >= OWN_SPREAD_MIN && own > 0.0 {
                own
            } else {
                sp.pooled_mad
            };
            if s > 0.0 {
                sp.median[j] - k * MAD_SIGMA * s
            } else {
                // No spread anywhere: flag nothing that was fit.
                v.iter().copied().fold(f32::INFINITY, f32::min)
            }
        })
        .collect()
}

/// The floor of a leaf whose median member cosine is `median`, with the pooled spread.
pub(crate) fn pooled_floor(median: f32, sp: &Spread, inc: &IncrementalConfig) -> f32 {
    match inc.outlier_min_sim {
        Some(t) => t,
        None => median - inc.outlier_mad_k as f32 * MAD_SIGMA * sp.pooled_mad,
    }
}

/// Fraction of rows whose stable leaf id is the same in `a` and `b`.
pub fn label_stability(a: &Labels, b: &Labels) -> f64 {
    let n = a.len().min(b.len());
    let same = (0..n).filter(|&i| a.leaf(i) == b.leaf(i)).count();
    same as f64 / n.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floors_flag_stragglers_not_a_fixed_share() {
        let inc = IncrementalConfig::default();
        // A large clean leaf, a small leaf with one straggler, an empty leaf.
        let clean: Vec<f32> = (0..200).map(|i| 0.88 + 0.0002 * i as f32).collect();
        let mut small: Vec<f32> = (0..9).map(|i| 0.90 + 0.002 * i as f32).collect();
        small.push(0.55);
        let sims = vec![clean.clone(), small.clone(), Vec::new()];
        let f = leaf_floors(&sims, &inc);
        let flagged = |v: &[f32], f: f32| v.iter().filter(|&&s| s < f).count();
        assert_eq!(flagged(&clean, f[0]), 0, "a clean leaf flags no row");
        assert_eq!(flagged(&small, f[1]), 1, "a small leaf flags its straggler");
        assert!(f[1] > 0.55 && f[1] < 0.90);
        assert!(f[2].is_finite(), "an empty leaf still has a floor");
        // One absolute floor replaces the rule.
        let abs = IncrementalConfig {
            outlier_min_sim: Some(0.5),
            ..Default::default()
        };
        assert_eq!(leaf_floors(&sims, &abs), vec![0.5; 3]);
        // Identical rows: nothing that was fit is flagged.
        let same = vec![vec![0.9f32; 30]];
        assert_eq!(flagged(&same[0], leaf_floors(&same, &inc)[0]), 0);
    }
}
