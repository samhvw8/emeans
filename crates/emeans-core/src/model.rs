//! The e-means model: faiss `NumpyEmeans` (Algorithm 1 of arXiv 2611.09501) in Rust.
//!
//! Float rules follow the reference: centroids, masses, batch sums and counts
//! are f32; epoch counts and objectives are f64; the EMA step is computed in
//! f64 and cast to f32 where numpy casts it (NEP 50 weak Python scalars).

use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::assign::{Assigner, CpuAssigner, Metric, Rows};
use crate::core::cpu::{add_rows, ema_host};
use crate::core::{EmaStep, apply_splits, closing_means, plan_revival};
use crate::recipe::{
    self, BatchLaw, CHECK_UNUSED_EVERY, DEFAULT_ASSIGNMENT_CHUNK, INIT_USAGE, Recipe,
};
use crate::rng::RandomState;

/// Constructor options (faiss `NumpyEmeans.__init__`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub d: usize,
    pub k: usize,
    /// Epoch budget T; fractional values run a partial pass before the closing pass.
    pub nepochs: f64,
    /// Statistical batch size B; None uses the batch law.
    pub batch_size: Option<usize>,
    /// Initial EMA step; None uses the step law at the resolved B.
    pub alpha0: Option<f64>,
    pub batch_law: BatchLaw,
    pub seed: u32,
    /// Cluster unit vectors by inner product with centroids kept on the unit sphere.
    pub spherical: bool,
    /// Revival threshold tau; None is 0.1, or 0 (off) when d <= 2.
    pub unused_threshold: Option<f64>,
    /// Train on at most `max_points_per_centroid * k` random rows.
    pub max_points_per_centroid: Option<usize>,
    /// Rows per assignment call; None is 65536.
    pub assignment_chunk_size: Option<usize>,
    /// Return last-pass labels from `train`.
    pub capture_assignments: bool,
    /// Print the recipe and per-epoch lines to stderr.
    pub verbose: bool,
}

impl Config {
    /// faiss defaults: 5 epochs, `batch_law = auto`, seed 1234, Euclidean.
    pub fn new(d: usize, k: usize) -> Self {
        Self {
            d,
            k,
            nepochs: 5.0,
            batch_size: None,
            alpha0: None,
            batch_law: BatchLaw::Auto,
            seed: 1234,
            spherical: false,
            unused_threshold: None,
            max_points_per_centroid: None,
            assignment_chunk_size: None,
            capture_assignments: true,
            verbose: false,
        }
    }

    pub fn metric(&self) -> Metric {
        if self.spherical {
            Metric::InnerProduct
        } else {
            Metric::L2
        }
    }
}

/// Statistics of one epoch (faiss `iteration_stats`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpochStats {
    /// Moving-pass objective: sum of squared distances, or sum of 2 - 2 cos (spherical).
    pub obj: f64,
    /// Seconds since `begin`.
    pub time: f64,
    /// k * sum(n_j^2) / rows^2 over the epoch's assignments.
    pub imbalance_factor: f64,
    /// Clusters revived by splitting during the epoch.
    pub nsplit: usize,
}

/// Result of one `accumulate` call.
#[derive(Clone, Debug)]
pub struct Accumulated {
    /// Rows accumulated so far in the current logical batch.
    pub n: usize,
    /// Rows the current logical batch needs before `update`.
    pub target: usize,
    pub complete: bool,
    /// Batch objective so far (f64).
    pub objective: f64,
    /// Nearest centroid of each supplied row against the batch snapshot.
    pub labels: Vec<u32>,
    /// Score of each supplied row: squared distance, or inner product (spherical).
    pub scores: Vec<f32>,
}

/// Output of `train`.
#[derive(Clone, Debug)]
pub struct Trained {
    /// Final centroids, `k x d`.
    pub centroids: Vec<f32>,
    /// Last-pass label of every input row; -1 for rows outside the training subset.
    /// None when `capture_assignments` is off.
    pub labels: Option<Vec<i32>>,
}

/// Output of `assign`.
#[derive(Clone, Debug)]
pub struct Assignment {
    pub labels: Vec<u32>,
    /// Squared L2 distance, or cosine similarity (spherical) to the assigned centroid.
    pub scores: Vec<f32>,
}

impl Assignment {
    /// Mean squared distance, or mean of 2 - 2 cos (spherical): the final objective.
    pub fn mean_objective(&self, spherical: bool) -> f64 {
        let n = self.scores.len().max(1) as f64;
        let s: f64 = self
            .scores
            .iter()
            .map(|&v| {
                if spherical {
                    2.0 - 2.0 * v as f64
                } else {
                    v as f64
                }
            })
            .sum();
        s / n
    }
}

struct Batch {
    sums: Vec<f32>,
    counts: Vec<u32>,
    objective: f64,
    n: usize,
}

struct Run {
    step: usize,
    epoch: usize,
    epoch_steps: usize,
    epoch_rows: usize,
    splits_before: usize,
    pass_counts: Vec<f64>,
    pass_sums: Option<Vec<f32>>,
    objective: f64,
    split_rng: RandomState,
    start: Instant,
    batch: Option<Batch>,
}

/// An e-means run and, after `finalize`, a trained model.
pub struct Emeans {
    config: Config,
    unused_threshold: f64,
    assigner: Mutex<Box<dyn Assigner>>,
    recipe: Option<Recipe>,
    /// EMA positions c_j = S_j / N_j (raw, not normalized), `k x d`.
    positions: Vec<f32>,
    /// EMA masses N_j.
    usage: Vec<f32>,
    /// Spherical readout cache: unit directions of `positions`.
    unit: Vec<f32>,
    unit_valid: bool,
    run: Option<Run>,
    centroids: Option<Vec<f32>>,
    iteration_stats: Vec<EpochStats>,
    n_splits: usize,
    failed: bool,
}

impl Emeans {
    /// A new model with the fastest CPU assigner.
    pub fn new(config: Config) -> Result<Self> {
        Self::with_assigner(config, Box::new(CpuAssigner::default()))
    }

    pub fn with_assigner(config: Config, assigner: Box<dyn Assigner>) -> Result<Self> {
        ensure!(config.d > 0 && config.k > 0, "d and k must be positive");
        ensure!(
            config.nepochs.is_finite() && config.nepochs > 0.0,
            "nepochs must be positive"
        );
        if let Some(m) = config.max_points_per_centroid {
            ensure!(m >= 1, "max_points_per_centroid must be >= 1 or None");
        }
        let unused_threshold = recipe::resolve_unused_threshold(config.unused_threshold, config.d)?;
        Ok(Self {
            config,
            unused_threshold,
            assigner: Mutex::new(assigner),
            recipe: None,
            positions: Vec::new(),
            usage: Vec::new(),
            unit: Vec::new(),
            unit_valid: false,
            run: None,
            centroids: None,
            iteration_stats: Vec::new(),
            n_splits: 0,
            failed: false,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Change options for the next run (for example nepochs before `resume`).
    pub fn config_mut(&mut self) -> Result<&mut Config> {
        ensure!(self.run.is_none(), "cannot change the config during a run");
        Ok(&mut self.config)
    }

    pub fn set_assigner(&mut self, assigner: Box<dyn Assigner>) {
        self.assigner = Mutex::new(assigner);
    }

    pub fn assigner_name(&self) -> String {
        self.assigner.lock().map(|a| a.name()).unwrap_or_default()
    }

    pub fn recipe(&self) -> Option<&Recipe> {
        self.recipe.as_ref()
    }

    pub fn iteration_stats(&self) -> &[EpochStats] {
        &self.iteration_stats
    }

    /// Splits applied in the current or last run.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Final centroids after `finalize`, `k x d`.
    pub fn centroids(&self) -> Option<&[f32]> {
        self.centroids.as_deref()
    }

    /// EMA positions (raw) and masses, kept after `finalize` so training can continue.
    pub fn ema_state(&self) -> Option<(&[f32], &[f32])> {
        (!self.usage.is_empty()).then_some((&self.positions[..], &self.usage[..]))
    }

    /// Per-epoch objectives (faiss `obj`).
    pub fn obj(&self) -> Vec<f64> {
        self.iteration_stats.iter().map(|s| s.obj).collect()
    }

    pub fn is_running(&self) -> bool {
        self.run.is_some()
    }

    /// Centroids the next assignment uses: unit directions in spherical mode.
    pub fn current_centroids(&mut self) -> Result<&[f32]> {
        self.require_ok()?;
        if self.run.is_none() {
            return self
                .centroids
                .as_deref()
                .context("no run started and no trained centroids");
        }
        self.refresh_readout();
        Ok(if self.config.spherical {
            &self.unit
        } else {
            &self.positions
        })
    }

    fn require_ok(&self) -> Result<()> {
        ensure!(!self.failed, "create a new object after execution failure");
        Ok(())
    }

    fn refresh_readout(&mut self) {
        if self.config.spherical && !self.unit_valid {
            self.unit.resize(self.positions.len(), 0.0);
            self.unit.copy_from_slice(&self.positions);
            normalize_rows(&mut self.unit, self.config.d);
            self.unit_valid = true;
        }
    }

    // ------------------------------------------------------------------ begin

    /// Start a caller-driven run on `global_n` rows from `init_centroids` (`k x d`).
    ///
    /// Each epoch supplies `recipe().epoch_rows[e]` rows in logical batches of
    /// `batch_size` (the last one may be shorter): `accumulate` until complete,
    /// then `update`; `end_epoch` after each epoch; `finalize` at the end.
    pub fn begin(&mut self, global_n: usize, init_centroids: &[f32]) -> Result<()> {
        let (k, d) = (self.config.k, self.config.d);
        ensure!(
            init_centroids.len() == k * d,
            "init centroids must be k x d"
        );
        let mut positions = init_centroids.to_vec();
        if self.config.spherical {
            normalize_rows(&mut positions, d);
        }
        self.start(global_n, positions, vec![INIT_USAGE; k])
    }

    /// Start a run from saved EMA state (positions and masses) instead of fresh centroids.
    ///
    /// The step schedule, revival and closing means restart for `global_n` rows.
    /// The masses carry over, so old clusters resist moving in proportion to them.
    pub fn begin_with_state(
        &mut self,
        global_n: usize,
        positions: &[f32],
        usage: &[f32],
    ) -> Result<()> {
        let (k, d) = (self.config.k, self.config.d);
        ensure!(positions.len() == k * d, "positions must be k x d");
        ensure!(usage.len() == k, "usage must have k entries");
        ensure!(
            usage.iter().all(|u| u.is_finite() && *u >= 0.0),
            "usage must be finite and >= 0"
        );
        self.start(global_n, positions.to_vec(), usage.to_vec())
    }

    /// Continue from this model's own EMA state (after `finalize` or `load`).
    pub fn resume(&mut self, global_n: usize) -> Result<()> {
        let (p, u) = self.ema_state().context("no EMA state to resume from")?;
        let (p, u) = (p.to_vec(), u.to_vec());
        self.start(global_n, p, u)
    }

    fn start(&mut self, n: usize, positions: Vec<f32>, usage: Vec<f32>) -> Result<()> {
        self.require_ok()?;
        ensure!(self.run.is_none(), "a run is already in progress");
        ensure!(n > 0, "global_n must be positive");
        ensure!(
            positions.iter().all(|v| v.is_finite()),
            "initial centroids must be finite"
        );
        let backend = self.assigner_name();
        let recipe = Recipe::resolve(
            n,
            self.config.k,
            self.config.d,
            self.config.nepochs,
            self.config.batch_law,
            self.config.batch_size,
            self.config.alpha0,
            self.unused_threshold,
            self.config.seed,
            self.config.spherical,
            self.config
                .assignment_chunk_size
                .unwrap_or(DEFAULT_ASSIGNMENT_CHUNK),
            backend,
        );
        ensure!(
            recipe.epoch_rows.iter().all(|&r| r > 0),
            "nepochs gives an empty epoch for n = {n}"
        );
        self.positions = positions;
        self.usage = usage;
        self.unit_valid = false;
        self.centroids = None;
        self.iteration_stats.clear();
        self.n_splits = 0;
        if self.config.verbose {
            eprintln!(
                "Clustering {} points in {}D to {} clusters, {} epochs (batch {}, alpha0 {:.3}, {:?} law{})",
                n,
                self.config.d,
                self.config.k,
                recipe.nepochs,
                recipe.batch_size,
                recipe.alpha0,
                recipe.batch_law,
                if self.config.spherical {
                    ", spherical"
                } else {
                    ""
                }
            );
        }
        self.recipe = Some(recipe);
        self.run = Some(Run {
            step: 0,
            epoch: 0,
            epoch_steps: 0,
            epoch_rows: 0,
            splits_before: 0,
            pass_counts: Vec::new(),
            pass_sums: None,
            objective: 0.0,
            split_rng: RandomState::new(self.config.seed),
            start: Instant::now(),
            batch: None,
        });
        self.start_epoch();
        Ok(())
    }

    fn start_epoch(&mut self) {
        let k = self.config.k;
        let kd = k * self.config.d;
        let last = self.recipe.as_ref().unwrap().epoch_rows.len() - 1;
        let n_splits = self.n_splits;
        let run = self.run.as_mut().unwrap();
        run.epoch_steps = 0;
        run.epoch_rows = 0;
        run.splits_before = n_splits;
        run.pass_counts = vec![0.0; k];
        run.objective = 0.0;
        run.pass_sums = (run.epoch == last).then(|| vec![0.0; kd]);
    }

    /// Rows the current logical batch needs.
    pub fn local_batch_size(&self) -> Result<usize> {
        let run = self.run.as_ref().context("no run in progress")?;
        let r = self.recipe.as_ref().unwrap();
        ensure!(run.epoch < r.epoch_rows.len(), "all epochs are done");
        Ok(r.batch_size.min(r.epoch_rows[run.epoch] - run.epoch_rows))
    }

    /// Whether this is the closing epoch, whose labels give the closing means.
    pub fn in_last_epoch(&self) -> bool {
        match (&self.run, &self.recipe) {
            (Some(run), Some(r)) => run.epoch + 1 == r.epoch_rows.len(),
            _ => false,
        }
    }

    // ------------------------------------------------------------- accumulate

    /// Assign rows (`m x d`) against the batch's centroid snapshot and add them
    /// to the batch counts and sums (Algorithm 1, lines 6-7).
    pub fn accumulate(&mut self, x: &[f32]) -> Result<Accumulated> {
        ensure!(x.len().is_multiple_of(self.config.d), "rows must be m x d");
        self.accumulate_rows(Rows::Dense(x))
    }

    /// `accumulate` for rows given by id into a larger `n x d` matrix.
    pub fn accumulate_rows(&mut self, rows: Rows<'_>) -> Result<Accumulated> {
        self.require_ok()?;
        let r = self.accumulate_inner(rows);
        if r.is_err() {
            self.failed = true;
        }
        r
    }

    fn accumulate_inner(&mut self, rows: Rows<'_>) -> Result<Accumulated> {
        let (k, d) = (self.config.k, self.config.d);
        let target = self.local_batch_size()?;
        let m = rows.len(d);
        self.refresh_readout();
        let metric = self.config.metric();
        let spherical = self.config.spherical;
        let chunk = self.recipe.as_ref().unwrap().assignment_chunk_size;
        let run = self.run.as_mut().unwrap();
        let batch = run.batch.get_or_insert_with(|| Batch {
            sums: vec![0.0; k * d],
            counts: vec![0; k],
            objective: 0.0,
            n: 0,
        });
        ensure!(
            batch.n + m <= target,
            "batch overflow: {} + {} rows > {} for this logical batch",
            batch.n,
            m,
            target
        );
        let centroids: &[f32] = if spherical {
            &self.unit
        } else {
            &self.positions
        };
        let mut labels = vec![0u32; m];
        let mut scores = vec![0f32; m];
        let mut assigner = self
            .assigner
            .lock()
            .map_err(|_| anyhow::anyhow!("assigner lock poisoned"))?;
        for b in (0..m).step_by(chunk.max(1)) {
            let e = (b + chunk).min(m);
            let part = rows.slice(b, e, d);
            assigner.assign(
                part,
                d,
                centroids,
                metric,
                &mut labels[b..e],
                &mut scores[b..e],
            )?;
            let mut obj = 0f64;
            if spherical {
                obj += 2.0 * (e - b) as f64;
                for &s in &scores[b..e] {
                    obj += (-2.0f32 * s) as f64;
                }
            } else {
                for &s in &scores[b..e] {
                    obj += s as f64;
                }
            }
            batch.objective += obj;
            add_rows(
                &mut batch.sums,
                &mut batch.counts,
                part,
                &labels[b..e],
                k,
                d,
            )?;
        }
        batch.n += m;
        Ok(Accumulated {
            n: batch.n,
            target,
            complete: batch.n == target,
            objective: batch.objective,
            labels,
            scores,
        })
    }

    // ----------------------------------------------------------------- update

    /// Apply the complete batch (Algorithm 1, line 8), then revive dead clusters
    /// every 5 updates (line 9).
    pub fn update(&mut self) -> Result<()> {
        self.require_ok()?;
        let r = self.update_inner();
        if r.is_err() {
            self.failed = true;
        }
        r
    }

    fn update_inner(&mut self) -> Result<()> {
        let target = self.local_batch_size()?;
        let d = self.config.d;
        let recipe = self.recipe.as_ref().unwrap();
        let (alpha0, total_steps) = (recipe.alpha0, recipe.total_steps);
        let run = self.run.as_mut().unwrap();
        let batch = run
            .batch
            .take()
            .context("update called before accumulate")?;
        ensure!(
            batch.n == target,
            "incomplete batch: {} of {} rows",
            batch.n,
            target
        );
        let alpha = recipe::scheduled_alpha(alpha0, run.step, total_steps);
        // Keep the batch in the epoch totals before the EMA (line 12 needs them).
        run.objective += batch.objective;
        for (p, &c) in run.pass_counts.iter_mut().zip(&batch.counts) {
            *p += c as f64;
        }
        if let Some(ps) = run.pass_sums.as_mut() {
            ps.par_iter_mut()
                .zip(&batch.sums)
                .for_each(|(p, &s)| *p += s);
        }
        ema_host(
            &mut self.positions,
            &mut self.usage,
            &batch.sums,
            &batch.counts,
            d,
            EmaStep::new(alpha),
        );
        self.unit_valid = false;
        // _finish_step
        let step = run.step;
        if (step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
            let mut rng = run.split_rng.clone();
            let n = self.revive_unused(&mut rng, d);
            let run = self.run.as_mut().unwrap();
            run.split_rng = rng;
            self.n_splits += n;
        }
        let run = self.run.as_mut().unwrap();
        run.step += 1;
        run.epoch_steps += 1;
        run.epoch_rows += target;
        Ok(())
    }

    /// Revive clusters below tau * mean mass by splitting donors sampled in
    /// proportion to their masses (Algorithm 1, line 9; Appendix J).
    fn revive_unused(&mut self, rng: &mut RandomState, d: usize) -> usize {
        let plan = plan_revival(&self.usage, self.unused_threshold, rng);
        if plan.is_empty() {
            return 0;
        }
        apply_splits(&mut self.positions, &mut self.usage, d, &plan);
        self.unit_valid = false;
        plan.len()
    }

    // -------------------------------------------------------------- end_epoch

    /// Finish one epoch and record its statistics.
    pub fn end_epoch(&mut self) -> Result<()> {
        self.require_ok()?;
        let k = self.config.k;
        let recipe = self.recipe.as_ref().context("no run in progress")?;
        let n_epochs = recipe.epoch_rows.len();
        let run = self.run.as_mut().context("no run in progress")?;
        ensure!(run.epoch < n_epochs, "all epochs are done");
        let rows = recipe.epoch_rows[run.epoch];
        ensure!(
            run.batch.is_none() && run.epoch_rows == rows,
            "epoch {} ended after {} of {} rows",
            run.epoch,
            run.epoch_rows,
            rows
        );
        let count_square: f64 = run.pass_counts.iter().map(|c| c * c).sum();
        let stat = EpochStats {
            obj: run.objective,
            time: run.start.elapsed().as_secs_f64(),
            imbalance_factor: k as f64 * count_square / (rows as f64 * rows as f64),
            nsplit: self.n_splits - run.splits_before,
        };
        if self.config.verbose {
            eprintln!(
                "  Epoch {} / {} ({} rows), iteration {} ({:.2} s): objective={}",
                run.epoch + 1,
                n_epochs,
                rows,
                run.step,
                stat.time,
                stat.obj
            );
        }
        self.iteration_stats.push(stat);
        run.epoch += 1;
        if run.epoch < n_epochs {
            self.start_epoch();
        }
        Ok(())
    }

    // --------------------------------------------------------------- finalize

    /// Closing means from the last epoch's assignments (Algorithm 1, line 12).
    ///
    /// Clusters with no last-epoch rows keep their EMA position. Spherical
    /// closing means are projected onto the unit sphere.
    pub fn finalize(&mut self) -> Result<&[f32]> {
        self.require_ok()?;
        if self.run.is_none() {
            return self.centroids.as_deref().context("no run to finalize");
        }
        let d = self.config.d;
        let n_epochs = self.recipe.as_ref().unwrap().epoch_rows.len();
        let run = self.run.as_ref().unwrap();
        ensure!(
            run.epoch == n_epochs,
            "finalize after epoch {} of {}",
            run.epoch,
            n_epochs
        );
        let sums = run.pass_sums.as_ref().context("closing sums missing")?;
        let out = closing_means(
            &self.positions,
            sums,
            &run.pass_counts,
            d,
            self.config.spherical,
        );
        self.centroids = Some(out);
        self.run = None;
        Ok(self.centroids.as_deref().unwrap())
    }

    // ------------------------------------------------------------------ train

    /// Train on `x` (`n x d`): sample k initial centroids, shuffle each epoch,
    /// and close with exact means. With `max_points_per_centroid`, train on a
    /// random subset and label the other rows -1.
    pub fn train(&mut self, x: &[f32]) -> Result<Trained> {
        self.train_from(x, None)
    }

    /// `train` with given initial centroids (`k x d`).
    pub fn train_from(&mut self, x: &[f32], init: Option<&[f32]>) -> Result<Trained> {
        let d = self.config.d;
        ensure!(x.len().is_multiple_of(d), "x must be n x d");
        let n = x.len() / d;
        let r = match self.training_subset(n) {
            None => self.train_rows(x, init, None),
            Some(keep) => {
                let mut sub = vec![0f32; keep.len() * d];
                sub.par_chunks_mut(d)
                    .zip(keep.par_iter())
                    .for_each(|(dst, &i)| {
                        dst.copy_from_slice(&x[i as usize * d..(i as usize + 1) * d])
                    });
                let t = self.train_rows(&sub, init, None)?;
                let labels = t.labels.map(|sampled| {
                    let mut all = vec![-1i32; n];
                    for (&i, &l) in keep.iter().zip(&sampled) {
                        all[i as usize] = l;
                    }
                    all
                });
                Ok(Trained {
                    centroids: t.centroids,
                    labels,
                })
            }
        };
        if r.is_err() {
            self.failed = true;
        }
        r
    }

    /// Continue training from the saved EMA state on `x` (new or grown data).
    pub fn train_resume(&mut self, x: &[f32]) -> Result<Trained> {
        let d = self.config.d;
        ensure!(x.len().is_multiple_of(d), "x must be n x d");
        let (p, u) = self.ema_state().context("no EMA state to resume from")?;
        let state = (p.to_vec(), u.to_vec());
        let r = self.train_rows(x, None, Some(state));
        if r.is_err() {
            self.failed = true;
        }
        r
    }

    /// Sorted row ids kept by `max_points_per_centroid`, or None for all rows.
    ///
    /// Deviation: faiss draws them with numpy `default_rng(seed).choice`; this
    /// port uses a legacy MT19937 permutation (seed ^ 0x9E3779B9) instead.
    pub fn training_subset(&self, n: usize) -> Option<Vec<u32>> {
        let limit = self.config.max_points_per_centroid? * self.config.k;
        if n <= limit {
            return None;
        }
        if self.config.verbose {
            eprintln!(
                "WARNING clustering {n} points to {} centroids: sampling {limit} points",
                self.config.k
            );
        }
        let mut keep = RandomState::new(self.config.seed ^ 0x9E37_79B9).permutation(n);
        keep.truncate(limit);
        keep.sort_unstable();
        Some(keep)
    }

    fn train_rows(
        &mut self,
        x: &[f32],
        init: Option<&[f32]>,
        state: Option<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Trained> {
        let (k, d) = (self.config.k, self.config.d);
        let n = x.len() / d;
        ensure!(n >= k, "need at least k = {k} rows, got {n}");
        // The device loop's first epoch order, drawn while the init runs (both
        // are full MT19937 permutations of n). Unused without an engine.
        let seed0 = recipe::epoch_seed(self.config.seed, 0);
        let first = std::thread::spawn(move || RandomState::new(seed0).permutation(n));
        match (state, init) {
            (Some((p, u)), _) => self.begin_with_state(n, &p, &u)?,
            (None, Some(c)) => self.begin(n, c)?,
            (None, None) if n == k => self.begin(n, x)?,
            (None, None) => {
                let perm = RandomState::new(self.config.seed).permutation(n);
                let mut c = vec![0f32; k * d];
                for (dst, &i) in c.chunks_mut(d).zip(&perm[..k]) {
                    dst.copy_from_slice(&x[i as usize * d..(i as usize + 1) * d]);
                }
                self.begin(n, &c)?
            }
        }
        if let Some(t) = self.train_device(x, first)? {
            return Ok(t);
        }
        // No engine (unfused legacy assigners): the streaming API, host EMA.
        self.assigner
            .lock()
            .map_err(|_| anyhow::anyhow!("assigner lock poisoned"))?
            .bind_data(x, d)?;
        let r = self.train_epochs(x, n);
        if let Ok(mut a) = self.assigner.lock() {
            a.unbind_data();
        }
        r
    }

    /// Run the started run on the assigner's engine (the device loop,
    /// `core::train`). None when the assigner has no engine.
    fn train_device(
        &mut self,
        x: &[f32],
        first: std::thread::JoinHandle<Vec<u32>>,
    ) -> Result<Option<Trained>> {
        let mut first = Some(first);
        let recipe = self.recipe.clone().context("no run started")?;
        let start = self.run.as_ref().context("no run started")?.start;
        let cfg = &self.config;
        let (positions, usage, tau) = (&self.positions, &self.usage, self.unused_threshold);
        let mut out = None;
        let handled = self
            .assigner
            .lock()
            .map_err(|_| anyhow::anyhow!("assigner lock poisoned"))?
            .with_engine(&mut |engine| {
                let job = crate::core::train::Job {
                    x,
                    d: cfg.d,
                    k: cfg.k,
                    recipe: &recipe,
                    spherical: cfg.spherical,
                    tau,
                    seed: cfg.seed,
                    positions: positions.clone(),
                    usage: usage.clone(),
                    capture: cfg.capture_assignments,
                    verbose: cfg.verbose,
                    start,
                    first_order: first.take(),
                };
                out = Some(engine.run_train(job)?);
                Ok(())
            });
        match handled {
            None => Ok(None),
            Some(r) => {
                r?;
                let o = out.context("engine did not run")?;
                self.positions = o.positions;
                self.usage = o.usage;
                self.unit_valid = false;
                self.iteration_stats = o.stats;
                self.n_splits = o.n_splits;
                self.centroids = Some(o.centroids.clone());
                self.run = None;
                Ok(Some(Trained {
                    centroids: o.centroids,
                    labels: o.labels,
                }))
            }
        }
    }

    fn train_epochs(&mut self, x: &[f32], n: usize) -> Result<Trained> {
        let recipe = self.recipe.clone().unwrap();
        let mut labels = self.config.capture_assignments.then(|| vec![-1i32; n]);
        let last = recipe.epoch_rows.len() - 1;
        for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
            let order =
                RandomState::new(recipe::epoch_seed(self.config.seed, epoch)).permutation(n);
            for b in (0..rows).step_by(recipe.batch_size) {
                let ids = &order[b..(b + recipe.batch_size).min(rows)];
                let acc = self.accumulate_rows(Rows::Indexed { data: x, ids })?;
                if epoch == last
                    && let Some(all) = labels.as_mut()
                {
                    for (&i, &l) in ids.iter().zip(&acc.labels) {
                        all[i as usize] = l as i32;
                    }
                }
                self.update()?;
            }
            self.end_epoch()?;
        }
        let centroids = self.finalize()?.to_vec();
        Ok(Trained { centroids, labels })
    }

    // ----------------------------------------------------------------- assign

    /// Nearest final centroid of every row: squared distances, or cosine
    /// similarities in spherical mode (rows must already be unit vectors).
    pub fn assign(&self, x: &[f32]) -> Result<Assignment> {
        let d = self.config.d;
        ensure!(x.len().is_multiple_of(d), "x must be n x d");
        self.assign_rows(Rows::Dense(x))
    }

    pub fn assign_rows(&self, rows: Rows<'_>) -> Result<Assignment> {
        self.require_ok()?;
        let d = self.config.d;
        let centroids = self.centroids.as_deref().context("model is not trained")?;
        let chunk = self
            .config
            .assignment_chunk_size
            .unwrap_or(DEFAULT_ASSIGNMENT_CHUNK);
        let mut a = self
            .assigner
            .lock()
            .map_err(|_| anyhow::anyhow!("assigner lock poisoned"))?;
        let (labels, scores) =
            crate::assign::assign_all(a.as_mut(), rows, d, centroids, self.config.metric(), chunk)?;
        Ok(Assignment { labels, scores })
    }

    // -------------------------------------------------------------- save/load

    #[doc(hidden)]
    pub fn snapshot(&self) -> Result<crate::io::Snapshot> {
        if self.run.is_some() {
            bail!("finish the run (finalize) before saving");
        }
        Ok(crate::io::Snapshot {
            config: self.config.clone(),
            recipe: self.recipe.clone(),
            n_splits: self.n_splits,
            iteration_stats: self.iteration_stats.clone(),
            centroids: self.centroids.clone(),
            positions: self.positions.clone(),
            usage: self.usage.clone(),
        })
    }

    #[doc(hidden)]
    pub fn from_snapshot(s: crate::io::Snapshot, assigner: Box<dyn Assigner>) -> Result<Self> {
        let mut m = Self::with_assigner(s.config, assigner)?;
        m.recipe = s.recipe;
        m.n_splits = s.n_splits;
        m.iteration_stats = s.iteration_stats;
        m.centroids = s.centroids;
        m.positions = s.positions;
        m.usage = s.usage;
        Ok(m)
    }
}

/// faiss `normalize_into`: prescale by the max magnitude, then divide by the norm.
/// Zero rows stay zero.
pub fn normalize_rows(m: &mut [f32], d: usize) {
    m.par_chunks_mut(d).for_each(|r| {
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
    });
}

/// numpy's float32 pairwise summation (`FLOAT_pairwise_sum`).
pub fn numpy_sum_f32(a: &[f32]) -> f32 {
    fn pw(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            let mut res = 0f32;
            for &v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [0f32; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - n % 8 {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pw(&a[..n2]) + pw(&a[n2..])
        }
    }
    pw(a)
}
