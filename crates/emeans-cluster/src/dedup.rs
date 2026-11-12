//! Near-duplicate groups: bursts and near-identical shots.
//!
//! Only one representative per group goes into clustering; every member
//! inherits its representative's label ([`DedupResult::expand`]).
//!
//! 1. Exact kNN (`k` neighbours per row, [`crate::knn`]).
//! 2. Link `i - j` when `cos >= sim_threshold` and, when both rows have a
//!    timestamp and `max_gap_secs` is set, `|t_i - t_j| <= max_gap_secs`.
//!    Rows that share a burst id are always linked.
//! 3. Connected components of the links (union-find).
//! 4. Anti-chaining: single linkage drifts (A ~ B ~ C ... where the ends are
//!    unrelated, for example a slow pan). Each component is therefore cut into
//!    stars: the unassigned row with the most links becomes the group's
//!    **anchor**, and every unassigned row with `cos >= rep_threshold` to the
//!    anchor joins it. A burst joins only when every one of its rows passes.
//!    Every member outside the anchor's own burst is then within the threshold
//!    of the anchor, so no two such members are further apart than twice the
//!    threshold angle. The rows of the anchor's burst are exempt: PhotoKit says
//!    they belong together, whatever their similarity. Full complete linkage
//!    would bound the diameter more tightly but is order-dependent and needs
//!    all pairwise similarities of a component; the star cover needs one
//!    similarity per member and anchor, and components are small.
//! 5. Representative: the member with the highest caller priority (for example
//!    1 for favorites), then the highest mean similarity to the other members
//!    (the medoid), then the lowest row id. The representative is the row that
//!    goes into clustering; it is not the anchor. Membership is always tested
//!    against the anchor ([`DedupResult::anchors`]), here and in
//!    [`crate::Hierarchy::add_photos`], so a later add and a fresh dedup apply
//!    the same rule.

use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Backend;
use crate::knn::{NONE, Neighbors, knn_self};

/// Dedup parameters. The defaults are starting points; calibrate them on real data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DedupConfig {
    /// Neighbours searched per row. Bounds how many links a row can have; the
    /// star cover still checks every member of a component against its center.
    pub k: usize,
    /// Cosine at or above which two rows are near-duplicates. Default 0.95.
    pub sim_threshold: f32,
    /// Cosine to the star center needed to stay in a group; None = `sim_threshold`.
    pub rep_threshold: Option<f32>,
    /// Largest time gap (seconds) for a link when both rows have a timestamp.
    /// None (default): similarity alone decides.
    pub max_gap_secs: Option<f64>,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            k: 10,
            sim_threshold: 0.95,
            rep_threshold: None,
            max_gap_secs: None,
        }
    }
}

/// Optional per-row side information.
#[derive(Clone, Copy, Debug, Default)]
pub struct DedupInput<'a> {
    /// Creation time (Unix seconds) per row; NaN = unknown (no time rule for that row).
    pub created_unix: Option<&'a [f64]>,
    /// Burst id per row (the caller interns the PhotoKit burst identifier).
    /// Rows with the same id are always in one group.
    pub burst: Option<&'a [Option<u32>]>,
    /// Representative preference per row; higher wins (for example 1 for favorites).
    pub priority: Option<&'a [f32]>,
}

/// Near-duplicate groups of `n` rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DedupResult {
    /// Group of every row. Groups are numbered by their lowest row id.
    pub group_of: Vec<u32>,
    /// Representative row of every group: the row that stands for the group in clustering.
    pub representatives: Vec<u32>,
    /// Anchor row of every group: the star center that membership is tested
    /// against. Empty in results written before anchors were stored;
    /// [`DedupResult::anchor`] then falls back to the representative.
    #[serde(default)]
    pub anchors: Vec<u32>,
    /// Members per group.
    pub sizes: Vec<u32>,
    /// Similarity of every row to its nearest other row (for calibrating the threshold).
    pub nn_sim: Vec<f32>,
    /// Links that passed the threshold and the time rule (each pair counted once).
    pub linked_pairs: usize,
    /// Components that the star cover cut into more than one group.
    pub split_components: usize,
}

impl DedupResult {
    pub fn n_groups(&self) -> usize {
        self.representatives.len()
    }

    /// The anchor row of group `g` (the representative when no anchors are stored).
    pub fn anchor(&self, g: usize) -> u32 {
        if self.anchors.len() == self.representatives.len() {
            self.anchors[g]
        } else {
            self.representatives[g]
        }
    }

    /// Give every row the label of its group's representative.
    /// `rep_labels[g]` is the label of `representatives[g]`.
    pub fn expand<T: Copy>(&self, rep_labels: &[T]) -> Vec<T> {
        assert_eq!(rep_labels.len(), self.representatives.len());
        self.group_of
            .iter()
            .map(|&g| rep_labels[g as usize])
            .collect()
    }

    /// The representative rows of `x` (`n x d`), in group order.
    pub fn gather(&self, x: &[f32], d: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.representatives.len() * d);
        for &r in &self.representatives {
            out.extend_from_slice(&x[r as usize * d..(r as usize + 1) * d]);
        }
        out
    }
}

struct UnionFind(Vec<u32>);

impl UnionFind {
    fn new(n: usize) -> Self {
        Self((0..n as u32).collect())
    }

    fn find(&mut self, mut i: u32) -> u32 {
        while self.0[i as usize] != i {
            let p = self.0[i as usize];
            self.0[i as usize] = self.0[p as usize];
            i = p;
        }
        i
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Lower root wins, so results do not depend on edge order.
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.0[hi as usize] = lo;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// Near-duplicate groups of the unit rows of `x` (`n x d`).
pub fn dedup(
    x: &[f32],
    d: usize,
    input: &DedupInput<'_>,
    cfg: &DedupConfig,
    backend: Backend,
) -> Result<DedupResult> {
    ensure!(d > 0 && x.len().is_multiple_of(d), "x must be n x d");
    let n = x.len() / d;
    let k = cfg.k.min(n.saturating_sub(1)).max(1);
    let nb = if n > 1 {
        knn_self(x, d, k, backend)?
    } else {
        Neighbors {
            k: 1,
            ids: vec![NONE; n],
            sims: vec![f32::NEG_INFINITY; n],
        }
    };
    dedup_with_neighbors(x, d, &nb, input, cfg)
}

/// [`dedup`] with neighbours computed by the caller (`knn_self` on `x`).
pub fn dedup_with_neighbors(
    x: &[f32],
    d: usize,
    nb: &Neighbors,
    input: &DedupInput<'_>,
    cfg: &DedupConfig,
) -> Result<DedupResult> {
    ensure!(d > 0 && x.len().is_multiple_of(d), "x must be n x d");
    let n = x.len() / d;
    ensure!(nb.len() == n, "neighbours must have one row per row of x");
    ensure!(
        (-1.0..=1.0).contains(&cfg.sim_threshold),
        "sim_threshold must be in [-1, 1]"
    );
    if let Some(t) = input.created_unix {
        ensure!(t.len() == n, "one timestamp per row");
    }
    if let Some(b) = input.burst {
        ensure!(b.len() == n, "one burst id per row");
    }
    if let Some(p) = input.priority {
        ensure!(p.len() == n, "one priority per row");
        ensure!(p.iter().all(|v| !v.is_nan()), "priority must not be NaN");
    }
    let rep_t = cfg.rep_threshold.unwrap_or(cfg.sim_threshold);
    let time_ok = |i: usize, j: usize| -> bool {
        match (cfg.max_gap_secs, input.created_unix) {
            (Some(gap), Some(t)) if t[i].is_finite() && t[j].is_finite() => {
                (t[i] - t[j]).abs() <= gap
            }
            _ => true,
        }
    };
    let row = |i: usize| &x[i * d..(i + 1) * d];

    // Burst units (always together) and link components.
    let mut unit = UnionFind::new(n);
    if let Some(b) = input.burst {
        let mut first = std::collections::HashMap::new();
        for (i, id) in b.iter().enumerate() {
            if let Some(id) = id {
                let f = *first.entry(*id).or_insert(i as u32);
                unit.union(f, i as u32);
            }
        }
    }
    let mut comp = UnionFind::new(n);
    for i in 0..n as u32 {
        let u = unit.find(i);
        comp.union(u, i);
    }
    let mut degree = vec![0u32; n];
    let mut pairs = std::collections::HashSet::new();
    for i in 0..n {
        let (ids, sims) = nb.row(i);
        for (&j, &s) in ids.iter().zip(sims) {
            if j == NONE || s < cfg.sim_threshold || !time_ok(i, j as usize) {
                continue;
            }
            comp.union(i as u32, j);
            let key = (i.min(j as usize) as u32, i.max(j as usize) as u32);
            if pairs.insert(key) {
                degree[i] += 1;
                degree[j as usize] += 1;
            }
        }
    }

    // Members of each component, in row order.
    let mut members: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for i in 0..n as u32 {
        members.entry(comp.find(i)).or_default().push(i);
    }
    let unit_of: Vec<u32> = (0..n as u32).map(|i| unit.find(i)).collect();

    // Star cover of each component.
    // Each group with its anchor (the star center).
    let mut groups: Vec<(Vec<u32>, u32)> = Vec::new();
    let mut split_components = 0usize;
    for rows in members.values() {
        if rows.len() == 1 {
            groups.push((rows.clone(), rows[0]));
            continue;
        }
        let mut units: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
        for &r in rows {
            units.entry(unit_of[r as usize]).or_default().push(r);
        }
        let mut order = rows.clone();
        order.sort_by_key(|&r| (std::cmp::Reverse(degree[r as usize]), r));
        let mut taken: std::collections::HashSet<u32> = Default::default();
        let before = groups.len();
        for &c in &order {
            let cu = unit_of[c as usize];
            if taken.contains(&cu) {
                continue;
            }
            taken.insert(cu);
            let mut g = units[&cu].clone();
            let rest: Vec<u32> = units
                .keys()
                .copied()
                .filter(|u| !taken.contains(u))
                .collect();
            let joins: Vec<bool> = rest
                .par_iter()
                .map(|u| {
                    // Every row of a burst must fit, or the burst would pull
                    // rows far from the anchor into the group.
                    units[u].iter().all(|&r| {
                        time_ok(r as usize, c as usize)
                            && dot(row(r as usize), row(c as usize)) >= rep_t
                    })
                })
                .collect();
            for (u, j) in rest.iter().zip(joins) {
                if j {
                    taken.insert(*u);
                    g.extend_from_slice(&units[u]);
                }
            }
            g.sort_unstable();
            groups.push((g, c));
        }
        if groups.len() - before > 1 {
            split_components += 1;
        }
    }
    groups.sort_by_key(|g| g.0[0]);
    let anchors: Vec<u32> = groups.iter().map(|g| g.1).collect();
    let groups: Vec<Vec<u32>> = groups.into_iter().map(|g| g.0).collect();

    // Representatives.
    let representatives: Vec<u32> = groups
        .par_iter()
        .map(|g| {
            if g.len() == 1 {
                return g[0];
            }
            let mut best = (f32::NEG_INFINITY, f64::NEG_INFINITY, u32::MAX);
            for &a in g {
                let mean: f64 = g
                    .iter()
                    .filter(|&&b| b != a)
                    .map(|&b| dot(row(a as usize), row(b as usize)) as f64)
                    .sum::<f64>()
                    / (g.len() - 1) as f64;
                let p = input.priority.map_or(0.0, |p| p[a as usize]);
                let better = p > best.0 || (p == best.0 && mean > best.1);
                if better {
                    best = (p, mean, a);
                }
            }
            best.2
        })
        .collect();
    let mut group_of = vec![0u32; n];
    for (gi, g) in groups.iter().enumerate() {
        for &r in g {
            group_of[r as usize] = gi as u32;
        }
    }
    let nn_sim = (0..n).map(|i| nb.row(i).1[0]).collect();
    Ok(DedupResult {
        group_of,
        representatives,
        anchors,
        sizes: groups.iter().map(|g| g.len() as u32).collect(),
        nn_sim,
        linked_pairs: pairs.len(),
        split_components,
    })
}
