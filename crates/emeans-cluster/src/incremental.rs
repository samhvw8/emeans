//! Incremental updates of a [`Hierarchy`]: label new rows without retraining,
//! and update or retrain only when the new rows ask for it.
//!
//! `add(x_all, n_old)` takes the full matrix (old rows first, new rows
//! appended, the `embeddings.f32` order) and:
//!
//! 1. Labels the new rows (nearest leaf). A row whose cosine to its leaf
//!    centroid is below the leaf's outlier floor goes into the pending bucket.
//!    The floor is the leaf's median member cosine at the last fit minus
//!    `outlier_mad_k` robust standard deviations (or the absolute
//!    `outlier_min_sim`); see `auto::leaf_floors`.
//! 2. Picks an action:
//!    - **FullRetrain** when the rows added since the last full training reach
//!      `retrain_added_fraction` of the rows it saw, or when the pending bucket
//!      reaches `update_min_outliers` and outliers are at least
//!      `retrain_outlier_fraction` of the rows added since the full training
//!      (the clusters no longer describe the new photos).
//!    - **EmaUpdate** when the pending bucket reaches `update_min_outliers`, or
//!      the rows added since the last update reach `update_added_fraction`:
//!      `update_epochs` of e-means continue from the saved EMA state of each
//!      group's leaf model (`train_resume` on the rows nearest to its leaves);
//!      level-1 centroids then become their groups' normalized row means.
//!    - **AssignOnly** otherwise.
//! 3. Keeps cluster ids stable: after an update or retrain, new centroids are
//!    matched to old ones by the Hungarian method on cosine similarity; a pair
//!    below `match_min_sim` is not a match, and an unmatched cluster gets a
//!    fresh id. Ids are never reused.
//!
//! [`Hierarchy::save`] writes a directory: `hierarchy.json` (config, dedup
//! parameters, ids, floors, decisions, counters) and one model file per level-1
//! and level-2 model, each with its EMA state.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::Backend;
use crate::assign::{SharedAssigner, gemm_nt};
use crate::auto::{
    AutoConfig, Diagnostics, Group, GroupDecision, Hierarchy, Labels, auto_cluster, gather,
    model_from, pooled_floor, spread, unit_means,
};
use crate::dedup::{DedupConfig, DedupInput, DedupResult, dedup};
use crate::make_assigner;
use crate::metrics::quantile;
use crate::model::Emeans;

/// Policy of [`Hierarchy::add`]. Defaults are starting points to calibrate on real data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Leaf outlier floor = the leaf's median member cosine minus this many
    /// robust standard deviations (1.4826 x the median absolute deviation).
    /// Default 3.0. Replaces the earlier `outlier_quantile`, which flagged a
    /// fixed share of every leaf's own training rows.
    #[serde(default = "default_outlier_mad_k")]
    pub outlier_mad_k: f64,
    /// Absolute outlier floor (cosine) instead of the robust rule. Default None.
    pub outlier_min_sim: Option<f32>,
    /// Smallest coherent group of pending outliers that an update turns into
    /// a new leaf. Default 20; 0 turns the grow step off.
    #[serde(default = "default_grow_min_size")]
    pub grow_min_size: usize,
    /// Pending outliers that trigger an update. Default 50.
    pub update_min_outliers: usize,
    /// Rows added since the last update, as a fraction of the rows then, that trigger an update. Default 0.10.
    pub update_added_fraction: f64,
    /// e-means epochs of an update. Default 1.0.
    pub update_epochs: f64,
    /// Rows added since the last full training, as a fraction of its rows, that trigger a retrain. Default 0.5.
    pub retrain_added_fraction: f64,
    /// Outlier share among rows added since the last full training that triggers a retrain. Default 0.25.
    pub retrain_outlier_fraction: f64,
    /// Cosine between an old and a new centroid needed to keep the old id. Default 0.8.
    pub match_min_sim: f32,
}

fn default_outlier_mad_k() -> f64 {
    3.0
}

fn default_grow_min_size() -> usize {
    20
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            outlier_mad_k: default_outlier_mad_k(),
            outlier_min_sim: None,
            grow_min_size: default_grow_min_size(),
            update_min_outliers: 50,
            update_added_fraction: 0.10,
            update_epochs: 1.0,
            retrain_added_fraction: 0.5,
            retrain_outlier_fraction: 0.25,
            match_min_sim: 0.8,
        }
    }
}

/// Counters behind the update policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct State {
    pub n_seen: usize,
    /// Rows at the last full training.
    pub n_full: usize,
    /// Rows at the last update (or full training).
    pub n_update: usize,
    pub added_since_full: usize,
    pub outliers_since_full: usize,
    pub added_since_update: usize,
    /// Rows labelled as outliers since the last update.
    pub pending: Vec<u32>,
    pub next_id1: u32,
}

impl State {
    pub fn new(n: usize, next_id1: u32) -> Self {
        Self {
            n_seen: n,
            n_full: n,
            n_update: n,
            added_since_full: 0,
            outliers_since_full: 0,
            added_since_update: 0,
            pending: Vec::new(),
            next_id1,
        }
    }
}

/// What `add` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    AssignOnly,
    EmaUpdate,
    FullRetrain,
}

/// Result of [`Hierarchy::add`].
#[derive(Clone, Debug)]
pub struct AddReport {
    pub action: Action,
    /// Labels of all rows after the action (old rows may move after an update or retrain).
    pub labels: Labels,
    /// New rows below their leaf's outlier floor when first labelled.
    pub new_outliers: usize,
    /// Rows in the pending bucket after the action.
    pub pending: usize,
    /// Outlier share among rows added since the last full training, before the action.
    pub outlier_fraction: f64,
    /// Stable ids `(l1, l2)` of the leaves the update formed from pending outliers.
    pub new_leaves: Vec<(u32, u32)>,
}

/// Result of [`Hierarchy::add_photos`].
#[derive(Clone, Debug)]
pub struct PhotoAddReport {
    /// The representative-level report (labels of every representative).
    pub reps: AddReport,
    /// Stable leaf id `(l1, l2)` of every photo (its representative's).
    pub leaf: Vec<(u32, u32)>,
    /// Outlier flag of every photo (its representative's).
    pub outlier: Vec<bool>,
    /// New photos that joined an existing dedup group.
    pub joined_existing: usize,
    /// Dedup groups formed by the remaining new photos (one new representative each).
    pub new_groups: usize,
}

/// Hungarian assignment minimizing `cost` (`rows x cols`, rows <= cols).
/// Returns the column of every row.
fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    assert!(n <= m);
    let inf = f64::INFINITY;
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; m + 1]);
    let (mut p, mut way) = (vec![0usize; m + 1], vec![0usize; m + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![inf; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (inf, 0);
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut ans = vec![0usize; n];
    for j in 1..=m {
        if p[j] != 0 {
            ans[p[j] - 1] = j - 1;
        }
    }
    ans
}

/// For each new centroid (unit rows), the old centroid it continues, if any:
/// a maximum-similarity one-to-one matching, pairs below `min_sim` dropped.
pub fn match_centroids(old: &[f32], new: &[f32], d: usize, min_sim: f32) -> Vec<Option<usize>> {
    let (ko, kn) = (old.len() / d, new.len() / d);
    let sim = |a: usize, b: usize| -> f64 {
        old[a * d..(a + 1) * d]
            .iter()
            .zip(&new[b * d..(b + 1) * d])
            .map(|(p, q)| *p as f64 * *q as f64)
            .sum()
    };
    let mut out = vec![None; kn];
    if ko == 0 || kn == 0 {
        return out;
    }
    if kn <= ko {
        let cost: Vec<Vec<f64>> = (0..kn)
            .map(|b| (0..ko).map(|a| -sim(a, b)).collect())
            .collect();
        for (b, a) in hungarian(&cost).into_iter().enumerate() {
            if sim(a, b) >= min_sim as f64 {
                out[b] = Some(a);
            }
        }
    } else {
        let cost: Vec<Vec<f64>> = (0..ko)
            .map(|a| (0..kn).map(|b| -sim(a, b)).collect())
            .collect();
        for (a, b) in hungarian(&cost).into_iter().enumerate() {
            if sim(a, b) >= min_sim as f64 {
                out[b] = Some(a);
            }
        }
    }
    out
}

/// Stable ids for new slots: matched slots keep the old id, others take fresh ones.
fn carry_ids(matched: &[Option<usize>], old_ids: &[u32], next: &mut u32) -> Vec<u32> {
    matched
        .iter()
        .map(|m| match m {
            Some(o) => old_ids[*o],
            None => {
                let id = *next;
                *next += 1;
                id
            }
        })
        .collect()
}

fn resume(m: &mut Emeans, x: &[f32], epochs: f64) -> Result<()> {
    let keep = m.config().nepochs;
    m.config_mut()?.nepochs = epochs;
    let r = m.train_resume(x).map(|_| ());
    // A failed model refuses more work, so restoring the config cannot hide the error.
    if let Ok(c) = m.config_mut() {
        c.nepochs = keep;
    }
    r
}

impl Hierarchy {
    /// Label rows `n_old..` of `x_all` and update the model when the policy says so.
    ///
    /// `n_old` must equal [`Hierarchy::n_seen`]: rows are append-only.
    pub fn add(&mut self, x_all: &[f32], n_old: usize) -> Result<AddReport> {
        let d = self.d;
        ensure!(x_all.len().is_multiple_of(d), "x must be n x d");
        let n = x_all.len() / d;
        ensure!(
            n_old == self.state.n_seen,
            "the model has seen {} rows but n_old is {n_old}",
            self.state.n_seen
        );
        ensure!(n >= n_old, "x_all has fewer rows than n_old");
        let fresh = self.assign(&x_all[n_old * d..])?;
        let new_outliers = fresh.outlier.iter().filter(|&&o| o).count();
        let inc = self.config.incremental.clone();
        let st = &mut self.state;
        st.pending.extend(
            fresh
                .outlier
                .iter()
                .enumerate()
                .filter(|(_, o)| **o)
                .map(|(i, _)| (n_old + i) as u32),
        );
        let m = n - n_old;
        st.n_seen = n;
        st.added_since_full += m;
        st.added_since_update += m;
        st.outliers_since_full += new_outliers;
        let outlier_fraction = st.outliers_since_full as f64 / st.added_since_full.max(1) as f64;
        let added_full = st.added_since_full as f64 / st.n_full.max(1) as f64;
        let added_update = st.added_since_update as f64 / st.n_update.max(1) as f64;
        let bucket_full = st.pending.len() >= inc.update_min_outliers;
        let action = if added_full >= inc.retrain_added_fraction
            || (bucket_full && outlier_fraction >= inc.retrain_outlier_fraction)
        {
            Action::FullRetrain
        } else if bucket_full || added_update >= inc.update_added_fraction {
            Action::EmaUpdate
        } else {
            Action::AssignOnly
        };
        let mut new_leaves = Vec::new();
        match action {
            Action::AssignOnly => {}
            Action::EmaUpdate => new_leaves = self.ema_update(x_all)?,
            Action::FullRetrain => self.full_retrain(x_all)?,
        }
        Ok(AddReport {
            action,
            new_leaves,
            labels: self.assign(x_all)?,
            new_outliers,
            pending: self.state.pending.len(),
            outlier_fraction,
        })
    }

    /// Form new leaves from the pending outliers that are coherent groups.
    ///
    /// Candidates are the pending rows that are still outliers. The row with
    /// the most candidates near it seeds a group; the group is then the
    /// candidates within a typical leaf floor of the group's mean, and nearer
    /// to it than to their current leaf (4 rounds of mean and members). It
    /// becomes a leaf when it has `grow_min_size` rows and its median member
    /// cosine reaches `cohesion`, the 10% quantile of the existing leaves'
    /// median member cosines: as tight as the leaves already there.
    ///
    /// The leaf joins the level-1 group whose centroid it is as close to as
    /// that group's own leaves are (groups of two or more leaves only);
    /// otherwise it becomes a new level-1 group. Slots are appended, so every
    /// existing slot keeps its id. Returns the stable ids of the new leaves.
    fn grow(&mut self, x_all: &[f32]) -> Result<Vec<(u32, u32)>> {
        let d = self.d;
        let inc = self.config.incremental.clone();
        let min_size = inc.grow_min_size;
        let mut made = Vec::new();
        if min_size == 0 || self.state.pending.len() < min_size {
            return Ok(made);
        }
        let labels = self.assign(x_all)?;
        let pend: Vec<u32> = self
            .state
            .pending
            .iter()
            .copied()
            .filter(|&i| labels.outlier[i as usize])
            .collect();
        let p = pend.len();
        if p < min_size {
            return Ok(made);
        }
        let (_, sims) = self.leaf_sims(&labels);
        let sp = spread(&sims);
        let medians: Vec<f32> = sp
            .median
            .iter()
            .copied()
            .filter(|m| m.is_finite())
            .collect();
        let cohesion = quantile(&medians, 0.10);
        let floor = pooled_floor(cohesion, &sp, &inc);
        ensure!(floor.is_finite(), "no leaf statistics to grow from");
        // Two rows at `floor` from a shared center have about this cosine.
        let pair = floor.max(0.0) * floor.max(0.0);
        let xp = gather(x_all, d, &pend);
        let cur: Vec<f32> = pend.iter().map(|&i| labels.sim[i as usize]).collect();
        let mut degree = vec![0u32; p];
        let chunk = 1024usize;
        let mut dots = vec![0f32; chunk.min(p) * p];
        for b0 in (0..p).step_by(chunk) {
            let b1 = (b0 + chunk).min(p);
            let out = &mut dots[..(b1 - b0) * p];
            gemm_nt(&xp[b0 * d..b1 * d], &xp, d, out)?;
            for (t, row) in out.chunks(p).enumerate() {
                degree[b0 + t] = row.iter().filter(|&&s| s >= pair).count() as u32 - 1;
            }
        }
        let to = |c: &[f32]| -> Result<Vec<f32>> {
            let mut out = vec![0f32; p];
            gemm_nt(&xp, c, d, &mut out)?;
            Ok(out)
        };
        let mean_of = |rows: &[usize]| -> Vec<f32> {
            let idx: Vec<u32> = rows.iter().map(|&t| t as u32).collect();
            unit_means(&xp, d, &[idx])
        };
        let mut free = vec![true; p];
        let mut tried = vec![false; p];
        let mut fails = 0;
        while fails < 3 && free.iter().filter(|&&f| f).count() >= min_size {
            let Some(seed) = (0..p)
                .filter(|&t| free[t] && !tried[t])
                .max_by_key(|&t| (degree[t], std::cmp::Reverse(t)))
            else {
                break;
            };
            tried[seed] = true;
            let s0 = to(&xp[seed * d..(seed + 1) * d])?;
            let mut mem: Vec<usize> = (0..p).filter(|&t| free[t] && s0[t] >= pair).collect();
            for _ in 0..4 {
                if mem.len() < min_size {
                    break;
                }
                let s = to(&mean_of(&mem))?;
                mem = (0..p)
                    .filter(|&t| free[t] && s[t] >= floor && s[t] > cur[t])
                    .collect();
            }
            if mem.len() < min_size {
                fails += 1;
                continue;
            }
            let center = mean_of(&mem);
            let s = to(&center)?;
            let member_sims: Vec<f32> = mem.iter().map(|&t| s[t]).collect();
            if quantile(&member_sims, 0.5) < cohesion {
                fails += 1;
                continue;
            }
            for &t in &mem {
                free[t] = false;
            }
            made.push(self.add_leaf(&center, mem.len(), &labels)?);
        }
        Ok(made)
    }

    /// Append a leaf with centroid `center` (see [`Hierarchy::grow`] for where).
    /// `labels` are the current labels of all rows, for the EMA masses.
    fn add_leaf(&mut self, center: &[f32], rows: usize, labels: &Labels) -> Result<(u32, u32)> {
        let d = self.d;
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(p, q)| p * q).sum() };
        let c1 = self.level1_centroids().to_vec();
        let mut home: Option<(f32, usize)> = None;
        for (g, grp) in self.groups.iter().enumerate() {
            let Some(m) = grp.model.as_ref().filter(|m| m.config().k >= 2) else {
                continue;
            };
            let cg = &c1[g * d..(g + 1) * d];
            let reach = m
                .centroids()
                .context("level-2 centroids")?
                .chunks(d)
                .map(|leaf| dot(leaf, cg))
                .fold(f32::INFINITY, f32::min);
            let s = dot(center, cg);
            if s >= reach && home.is_none_or(|h| s > h.0) {
                home = Some((s, g));
            }
        }
        let mut count1 = vec![0usize; self.groups.len()];
        for &g in &labels.slot1 {
            count1[g as usize] += 1;
        }
        if let Some((_, g)) = home {
            let grp = &mut self.groups[g];
            let model = grp.model.as_ref().expect("checked above");
            let k = model.config().k;
            let mut cents = model.centroids().context("level-2 centroids")?.to_vec();
            cents.extend_from_slice(center);
            let mut counts = vec![0usize; k];
            for (&g1, &j) in labels.slot1.iter().zip(&labels.slot2) {
                if g1 as usize == g {
                    counts[j as usize] += 1;
                }
            }
            counts.iter_mut().for_each(|c| *c = (*c).max(1));
            counts.push(rows);
            grp.model = Some(model_from(&cents, &counts, d, &self.config, &self.shared)?);
            let id = grp.next_id;
            grp.next_id += 1;
            grp.ids.push(id);
            grp.floors.push(f32::NAN);
            grp.decision = GroupDecision::Split { k: k + 1 };
            return Ok((self.ids1[g], id));
        }
        let model = model_from(center, &[rows], d, &self.config, &self.shared)?;
        self.groups.push(Group {
            model: Some(model),
            ids: vec![0],
            floors: vec![f32::NAN],
            decision: GroupDecision::TooSmall,
            next_id: 1,
        });
        let id1 = self.state.next_id1;
        self.state.next_id1 += 1;
        self.ids1.push(id1);
        let mut cents1 = c1;
        cents1.extend_from_slice(center);
        count1.iter_mut().for_each(|c| *c = (*c).max(1));
        count1.push(rows);
        self.level1 = model_from(&cents1, &count1, d, &self.config, &self.shared)?;
        Ok((id1, 0))
    }

    /// Grow new leaves from coherent pending outliers ([`Hierarchy::grow`]),
    /// then continue e-means from the saved EMA state: each group's leaf model
    /// on the rows whose nearest leaf is in that group, then the level-1
    /// centroids become the normalized means of their groups' rows. Groups
    /// keep their slots and ids; leaves keep theirs unless matching moves
    /// them. Returns the stable ids of the new leaves.
    pub fn ema_update(&mut self, x_all: &[f32]) -> Result<Vec<(u32, u32)>> {
        let d = self.d;
        let n = x_all.len() / d;
        let epochs = self.config.incremental.update_epochs;
        let min_sim = self.config.incremental.match_min_sim;
        let new_leaves = self.grow(x_all)?;
        let labels = self.assign(x_all)?;
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); self.groups.len()];
        for (i, &g) in labels.slot1.iter().enumerate() {
            rows[g as usize].push(i as u32);
        }
        for (g, grp) in self.groups.iter_mut().enumerate() {
            let Some(model) = grp.model.as_mut() else {
                continue;
            };
            if rows[g].len() < model.config().k {
                continue;
            }
            let xg = gather(x_all, d, &rows[g]);
            let old2 = model.centroids().context("level-2 centroids")?.to_vec();
            resume(model, &xg, epochs).context("level-2 update")?;
            let new2 = model.centroids().context("level-2 centroids")?.to_vec();
            let m2 = match_centroids(&old2, &new2, d, min_sim);
            grp.ids = carry_ids(&m2, &grp.ids, &mut grp.next_id);
        }
        // Level 1 follows its groups: normalized mean of each group's rows
        // (after the leaf update); an empty group keeps its old centroid.
        let labels = self.assign(x_all)?;
        let k1 = self.groups.len();
        let mut grp_rows: Vec<Vec<u32>> = vec![Vec::new(); k1];
        for (i, &g) in labels.slot1.iter().enumerate() {
            grp_rows[g as usize].push(i as u32);
        }
        let mut c1 = unit_means(x_all, d, &grp_rows);
        let old1 = self.level1_centroids();
        for (g, r) in grp_rows.iter().enumerate() {
            if r.is_empty() {
                c1[g * d..(g + 1) * d].copy_from_slice(&old1[g * d..(g + 1) * d]);
            }
        }
        let counts: Vec<usize> = grp_rows.iter().map(|r| r.len().max(1)).collect();
        self.level1 = model_from(&c1, &counts, d, &self.config, &self.shared)?;
        let labels = self.assign(x_all)?;
        self.set_floors(&labels);
        let st = &mut self.state;
        st.pending.clear();
        st.added_since_update = 0;
        st.n_update = n;
        Ok(new_leaves)
    }

    /// Run `auto_cluster` on all rows and carry the old ids over to matching clusters.
    pub fn full_retrain(&mut self, x_all: &[f32]) -> Result<()> {
        let d = self.d;
        let n = x_all.len() / d;
        let min_sim = self.config.incremental.match_min_sim;
        let mut cfg = self.config.clone();
        cfg.seed = cfg.seed.wrapping_add(self.state.n_seen as u32);
        let (mut h, _) = auto_cluster(x_all, d, &cfg)?;
        let matched = match_centroids(self.level1_centroids(), h.level1_centroids(), d, min_sim);
        let mut next1 = self.state.next_id1;
        h.ids1 = carry_ids(&matched, &self.ids1, &mut next1);
        for (slot, m) in matched.iter().enumerate() {
            if let Some(o) = m {
                let old = self.leaf_centroids(*o);
                let new = h.leaf_centroids(slot);
                let m2 = match_centroids(&old, &new, d, min_sim);
                let og = &self.groups[*o];
                let mut next = og.next_id.max(1);
                let ids = if og.model.is_some() {
                    carry_ids(&m2, &og.ids, &mut next)
                } else {
                    carry_ids(&vec![None; m2.len()], &[], &mut next)
                };
                let g = &mut h.groups[slot];
                if g.model.is_some() {
                    g.ids = ids;
                }
                g.next_id = next;
            }
        }
        h.state = State::new(n, next1);
        h.config = self.config.clone();
        h.dedup = self.dedup.clone();
        *self = h;
        Ok(())
    }

    /// Add new photos when the hierarchy clusters dedup representatives.
    ///
    /// `photos` holds every photo (old rows first, new rows appended) and `dd`
    /// the dedup groups of the old photos; `self` must have been trained on (or
    /// fed) exactly `dd`'s representatives in group order. Each new photo:
    ///
    /// 1. joins an existing group when it shares a burst id with one of the
    ///    group's photos, or when one of its `k` neighbours among all photos is
    ///    an old photo with `cos >= sim_threshold` (and the time rule) and the
    ///    photo is within `rep_threshold` of that group's **anchor** (and the
    ///    time rule to it): the star rule of [`crate::dedup`], tested against
    ///    the same row as there, so an add and a fresh dedup agree. The
    ///    closest such anchor wins. New photos of one burst are one unit: it
    ///    joins only when every one of its photos fits the anchor;
    /// 2. otherwise is grouped with the other remaining new photos by
    ///    [`dedup`]; each such group appends one representative.
    ///
    /// Existing groups, anchors and representatives never change, so the
    /// hierarchy rows stay append-only; the new representatives go through
    /// [`Hierarchy::add`]. A fresh dedup of all photos can still differ where
    /// a new photo would have become the anchor of old photos or linked two
    /// old groups: that needs old groups to change.
    /// Dedup parameters come from `self.dedup` (the default when None).
    pub fn add_photos(
        &mut self,
        photos: &[f32],
        n_old: usize,
        dd: &mut DedupResult,
        input: &DedupInput<'_>,
        backend: Backend,
    ) -> Result<PhotoAddReport> {
        let d = self.d;
        ensure!(photos.len().is_multiple_of(d), "photos must be n x d");
        let n = photos.len() / d;
        ensure!(n >= n_old, "photos has fewer rows than n_old");
        ensure!(
            dd.group_of.len() == n_old,
            "dedup result covers {} photos but n_old is {n_old}",
            dd.group_of.len()
        );
        let n_reps_old = dd.representatives.len();
        ensure!(
            n_reps_old == self.state.n_seen,
            "the hierarchy has seen {} rows but the dedup result has {n_reps_old} groups",
            self.state.n_seen
        );
        if let Some(t) = input.created_unix {
            ensure!(t.len() == n, "one timestamp per photo");
        }
        if let Some(b) = input.burst {
            ensure!(b.len() == n, "one burst id per photo");
        }
        if let Some(p) = input.priority {
            ensure!(p.len() == n, "one priority per photo");
        }
        let cfg = self.dedup.clone().unwrap_or_default();
        let rep_t = cfg.rep_threshold.unwrap_or(cfg.sim_threshold);
        let time_ok = |i: usize, j: usize| -> bool {
            match (cfg.max_gap_secs, input.created_unix) {
                (Some(gap), Some(t)) if t[i].is_finite() && t[j].is_finite() => {
                    (t[i] - t[j]).abs() <= gap
                }
                _ => true,
            }
        };
        let row = |i: usize| &photos[i * d..(i + 1) * d];
        let dot =
            |a: usize, b: usize| -> f32 { row(a).iter().zip(row(b)).map(|(p, q)| p * q).sum() };
        let m = n - n_old;

        // Neighbours of the new photos among all photos (self excluded).
        let k = cfg.k.min(n.saturating_sub(1)).max(1);
        let nb = if m > 0 && n > 1 {
            Some(crate::knn::knn(
                &photos[n_old * d..],
                photos,
                d,
                k + 1,
                backend,
            )?)
        } else {
            None
        };
        // Results written before anchors were stored tested against the representative.
        if dd.anchors.len() != n_reps_old {
            dd.anchors = dd.representatives.clone();
        }
        let mut burst_group = std::collections::HashMap::new();
        if let Some(b) = input.burst {
            for (i, id) in b.iter().enumerate().take(n_old) {
                if let Some(id) = id {
                    burst_group.entry(*id).or_insert(dd.group_of[i]);
                }
            }
        }
        // Units of new photos: a burst is one unit and always stays together.
        let mut units: Vec<Vec<usize>> = Vec::new();
        let mut unit_of_burst = std::collections::HashMap::new();
        for t in 0..m {
            match input.burst.and_then(|b| b[n_old + t]) {
                Some(id) => {
                    let u = *unit_of_burst.entry(id).or_insert_with(|| {
                        units.push(Vec::new());
                        units.len() - 1
                    });
                    units[u].push(t);
                }
                None => units.push(vec![t]),
            }
        }
        // The star rule of `dedup`: within the threshold of the group's anchor.
        let fits = |i: usize, g: u32| -> Option<f32> {
            let a = dd.anchor(g as usize) as usize;
            let s = dot(i, a);
            (s >= rep_t && time_ok(i, a)).then_some(s)
        };
        let mut joined = vec![None; m];
        let mut nn_new = vec![f32::NEG_INFINITY; m];
        for unit in &units {
            // Groups that a row of the unit is linked to.
            let mut linked = std::collections::BTreeSet::new();
            let mut forced = None;
            for &t in unit {
                let i = n_old + t;
                if forced.is_none()
                    && let Some(Some(id)) = input.burst.map(|b| b[i])
                {
                    forced = burst_group.get(&id).copied();
                }
                let Some(nb) = &nb else { continue };
                let (ids, sims) = nb.row(t);
                for (&j, &s) in ids.iter().zip(sims) {
                    if j == crate::knn::NONE || j as usize == i {
                        continue;
                    }
                    if s > nn_new[t] {
                        nn_new[t] = s;
                    }
                    let j = j as usize;
                    if j < n_old && s >= cfg.sim_threshold && time_ok(i, j) {
                        linked.insert(dd.group_of[j]);
                    }
                }
            }
            // Every row of the unit must fit the anchor; the closest anchor wins.
            let mut best: Option<(f32, u32)> = None;
            for &g in &linked {
                let worst = unit
                    .iter()
                    .map(|&t| fits(n_old + t, g))
                    .try_fold(f32::INFINITY, |acc, s| s.map(|s| acc.min(s)));
                if let Some(w) = worst
                    && best.is_none_or(|b| w > b.0)
                {
                    best = Some((w, g));
                }
            }
            let group = forced.or(best.map(|b| b.1));
            for &t in unit {
                joined[t] = group;
            }
        }

        // Group the remaining new photos among themselves.
        let rest: Vec<usize> = (0..m).filter(|&t| joined[t].is_none()).collect();
        let mut new_groups = 0usize;
        let mut sub_split = 0usize;
        let mut sub_pairs = 0usize;
        let mut group_of_new: Vec<u32> = joined.iter().map(|g| g.unwrap_or(u32::MAX)).collect();
        if !rest.is_empty() {
            let idx: Vec<u32> = rest.iter().map(|&t| (n_old + t) as u32).collect();
            let xs = gather(photos, d, &idx);
            let sub_t: Option<Vec<f64>> = input
                .created_unix
                .map(|v| idx.iter().map(|&i| v[i as usize]).collect());
            let sub_b: Option<Vec<Option<u32>>> = input
                .burst
                .map(|v| idx.iter().map(|&i| v[i as usize]).collect());
            let sub_p: Option<Vec<f32>> = input
                .priority
                .map(|v| idx.iter().map(|&i| v[i as usize]).collect());
            let sub_in = DedupInput {
                created_unix: sub_t.as_deref(),
                burst: sub_b.as_deref(),
                priority: sub_p.as_deref(),
            };
            let r = dedup(&xs, d, &sub_in, &cfg, backend)?;
            let base = dd.representatives.len() as u32;
            for (&t, &g) in rest.iter().zip(&r.group_of) {
                group_of_new[t] = base + g;
            }
            for (g, (&rep, &size)) in r.representatives.iter().zip(&r.sizes).enumerate() {
                dd.representatives.push(idx[rep as usize]);
                dd.anchors.push(idx[r.anchor(g) as usize]);
                dd.sizes.push(size);
            }
            new_groups = r.n_groups();
            sub_split = r.split_components;
            sub_pairs = r.linked_pairs;
        }
        for &g in group_of_new.iter().filter(|&&g| (g as usize) < n_reps_old) {
            dd.sizes[g as usize] += 1;
        }
        dd.group_of.extend_from_slice(&group_of_new);
        dd.nn_sim.extend_from_slice(&nn_new);
        dd.split_components += sub_split;
        dd.linked_pairs += sub_pairs;

        let reps = dd.gather(photos, d);
        let report = self.add(&reps, n_reps_old)?;
        let leaf = dd
            .group_of
            .iter()
            .map(|&g| report.labels.leaf(g as usize))
            .collect();
        let outlier = dd
            .group_of
            .iter()
            .map(|&g| report.labels.outlier[g as usize])
            .collect();
        Ok(PhotoAddReport {
            joined_existing: joined.iter().filter(|g| g.is_some()).count(),
            new_groups,
            leaf,
            outlier,
            reps: report,
        })
    }

    // ------------------------------------------------------------ save/load

    /// Write the hierarchy to directory `dir` (replaced as a whole).
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        let tmp = dir.with_extension("tmp");
        if tmp.exists() {
            fs::remove_dir_all(&tmp)?;
        }
        fs::create_dir_all(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        self.level1.save(tmp.join("level1.emeans"))?;
        for (g, grp) in self.groups.iter().enumerate() {
            if let Some(m) = &grp.model {
                m.save(tmp.join(format!("group-{g}.emeans")))?;
            }
        }
        fs::write(
            tmp.join("hierarchy.json"),
            serde_json::to_vec_pretty(&self.manifest())?,
        )?;
        let old = dir.with_extension("old");
        if old.exists() {
            fs::remove_dir_all(&old)?;
        }
        if dir.exists() {
            fs::rename(dir, &old)?;
        }
        fs::rename(&tmp, dir)?;
        if old.exists() {
            fs::remove_dir_all(&old)?;
        }
        Ok(())
    }

    /// The hierarchy as one byte blob, for a store without a file system:
    /// magic `EMHIER\0\x01`, u64 manifest length, the manifest JSON (as in
    /// `hierarchy.json`), then each model as u64 length + model file bytes
    /// (level 1, then every group with a model, in group order). Little-endian.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(&self.manifest())?;
        let mut buf = Vec::new();
        buf.extend_from_slice(BLOB_MAGIC);
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(&json);
        let models = std::iter::once(&self.level1)
            .chain(self.groups.iter().filter_map(|g| g.model.as_ref()));
        for m in models {
            let b = m.to_bytes()?;
            buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
            buf.extend_from_slice(&b);
        }
        Ok(buf)
    }

    /// Read a hierarchy from the bytes of [`Hierarchy::to_bytes`].
    pub fn from_bytes(b: &[u8], backend: Backend) -> Result<Hierarchy> {
        ensure!(
            b.len() >= 8 && &b[..8] == BLOB_MAGIC,
            "not an emeans hierarchy"
        );
        let mut off = 8;
        let mut take = |what: &str| -> Result<&[u8]> {
            ensure!(b.len() - off >= 8, "truncated {what}");
            let n = usize::try_from(u64::from_le_bytes(b[off..off + 8].try_into().unwrap()))
                .ok()
                .filter(|&n| n <= b.len() - off - 8)
                .with_context(|| format!("truncated {what}"))?;
            let part = &b[off + 8..off + 8 + n];
            off += 8 + n;
            Ok(part)
        };
        let mf: Manifest = serde_json::from_slice(take("manifest")?)?;
        let h = Self::assemble(mf, backend, |g, assigner| {
            let what = match g {
                None => "level-1 model".to_string(),
                Some(g) => format!("group {g} model"),
            };
            Emeans::from_bytes_with(take(&what)?, assigner).with_context(|| what)
        })?;
        ensure!(
            off == b.len(),
            "{} bytes after the last model",
            b.len() - off
        );
        Ok(h)
    }

    fn manifest(&self) -> Manifest {
        Manifest {
            version: MANIFEST_VERSION,
            d: self.d,
            config: self.config.clone(),
            dedup: self.dedup.clone(),
            ids1: self.ids1.clone(),
            groups: self
                .groups
                .iter()
                .map(|g| GroupFile {
                    has_model: g.model.is_some(),
                    ids: g.ids.clone(),
                    floors: g
                        .floors
                        .iter()
                        .map(|&f| f.is_finite().then_some(f))
                        .collect(),
                    decision: g.decision.clone(),
                    next_id: g.next_id,
                })
                .collect(),
            diagnostics: self.diagnostics.clone(),
            state: self.state.clone(),
        }
    }

    /// Read a hierarchy written by [`Hierarchy::save`].
    pub fn load(dir: impl AsRef<Path>, backend: Backend) -> Result<Hierarchy> {
        let dir = dir.as_ref();
        let b = fs::read(dir.join("hierarchy.json"))
            .with_context(|| format!("read {}/hierarchy.json", dir.display()))?;
        let mf: Manifest = serde_json::from_slice(&b)?;
        Self::assemble(mf, backend, |g, assigner| match g {
            None => Emeans::load_with(dir.join("level1.emeans"), assigner),
            Some(g) => Emeans::load_with(dir.join(format!("group-{g}.emeans")), assigner),
        })
    }

    /// Build a hierarchy from its manifest; `model(None)` reads the level-1
    /// model, `model(Some(g))` group `g`'s (called in that order).
    fn assemble(
        mf: Manifest,
        backend: Backend,
        mut model: impl FnMut(Option<usize>, Box<dyn crate::Assigner>) -> Result<Emeans>,
    ) -> Result<Hierarchy> {
        if mf.version != MANIFEST_VERSION {
            bail!("unsupported hierarchy version {}", mf.version);
        }
        let shared = SharedAssigner::new(make_assigner(backend)?);
        let level1 = model(None, Box::new(shared.clone()))?;
        ensure!(
            level1.config().d == mf.d,
            "level-1 model d differs from the manifest"
        );
        ensure!(
            level1.config().k == mf.ids1.len() && mf.groups.len() == mf.ids1.len(),
            "level-1 k, ids and groups disagree"
        );
        let mut groups = Vec::with_capacity(mf.groups.len());
        for (g, gf) in mf.groups.into_iter().enumerate() {
            let model = if gf.has_model {
                let m = model(Some(g), Box::new(shared.clone()))?;
                ensure!(
                    m.config().k == gf.ids.len() && m.config().d == mf.d,
                    "group {g}: model and ids disagree"
                );
                Some(m)
            } else {
                None
            };
            groups.push(Group {
                model,
                ids: gf.ids,
                floors: gf
                    .floors
                    .into_iter()
                    .map(|f| f.unwrap_or(f32::NAN))
                    .collect(),
                decision: gf.decision,
                next_id: gf.next_id,
            });
        }
        let mut config = mf.config;
        config.backend = backend;
        Ok(Hierarchy {
            d: mf.d,
            config,
            dedup: mf.dedup,
            level1,
            ids1: mf.ids1,
            groups,
            diagnostics: mf.diagnostics,
            state: mf.state,
            shared,
        })
    }
}

const MANIFEST_VERSION: u32 = 1;
const BLOB_MAGIC: &[u8; 8] = b"EMHIER\0\x01";

#[derive(Serialize, Deserialize)]
struct GroupFile {
    has_model: bool,
    ids: Vec<u32>,
    /// None for a leaf with no members at the last fit (JSON has no NaN).
    floors: Vec<Option<f32>>,
    decision: GroupDecision,
    next_id: u32,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    d: usize,
    config: AutoConfig,
    dedup: Option<DedupConfig>,
    ids1: Vec<u32>,
    groups: Vec<GroupFile>,
    diagnostics: Diagnostics,
    state: State,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_finds_the_optimum() {
        // Brute force over all permutations of 5.
        let c = [
            [7.0, 3.0, 9.0, 1.0, 4.0],
            [2.0, 8.0, 6.0, 5.0, 3.0],
            [4.0, 4.0, 2.0, 7.0, 9.0],
            [6.0, 1.0, 3.0, 8.0, 2.0],
            [5.0, 7.0, 4.0, 2.0, 6.0],
        ];
        let cost: Vec<Vec<f64>> = c.iter().map(|r| r.to_vec()).collect();
        let a = hungarian(&cost);
        let total: f64 = a.iter().enumerate().map(|(i, &j)| c[i][j]).sum();
        let mut best = f64::INFINITY;
        let mut p = [0, 1, 2, 3, 4];
        permute(&mut p, 0, &mut |p| {
            best = best.min(p.iter().enumerate().map(|(i, &j)| c[i][j]).sum());
        });
        assert_eq!(total, best);
    }

    fn permute(p: &mut [usize; 5], k: usize, f: &mut dyn FnMut(&[usize; 5])) {
        if k == p.len() {
            f(p);
            return;
        }
        for i in k..p.len() {
            p.swap(k, i);
            permute(p, k + 1, f);
            p.swap(k, i);
        }
    }

    #[test]
    fn matching_keeps_ids_under_permutation() {
        let d = 4;
        let old = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0f32,
        ];
        // New: old 2, old 0, and a new direction.
        let new = [
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0f32,
        ];
        let m = match_centroids(&old, &new, d, 0.8);
        assert_eq!(m, vec![Some(2), Some(0), None]);
        let mut next = 3;
        assert_eq!(carry_ids(&m, &[10, 11, 12], &mut next), vec![12, 10, 3]);
        assert_eq!(next, 4);
    }
}
