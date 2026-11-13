//! Layer 2: the e-means training loop, written once against [`Device`].
//!
//! Per batch, all on the device and queued without a host wait: spherical
//! readout (normalize) or centroid norms (L2), fused top-1 assignment of the
//! batch's rows, scatter-add into batch sums and counts (and the closing
//! sums in the last epoch), the EMA update. The host waits only for:
//!
//! - revival checks, every `CHECK_UNUSED_EVERY` updates: it reads the masses
//!   (`k` floats), plans the splits (MT19937 draws stay on the host) and sends
//!   back only the (donor, dead) pairs for the device `split`;
//! - epoch ends: it reads the epoch's labels and scores for the statistics
//!   (objective, imbalance, last-pass labels);
//! - the end: positions, masses and closing sums.
//!
//! The row order is the MT19937 permutation of each epoch, computed on the host
//! by a helper thread (one epoch ahead; inline on targets without threads, see
//! [`crate::exec`]) and uploaded once per epoch. Every
//! number that drives a decision is the same f32/f64 arithmetic as the
//! streaming API, so the CPU backend reproduces it bit for bit.

use std::sync::mpsc;

use anyhow::{Context, Result, ensure};

use super::{Device, EmaStep, Mat, Select, closing_means, plan_revival};
use crate::assign::Metric;
use crate::exec::{self, Instant};
use crate::model::EpochStats;
use crate::recipe::{self, CHECK_UNUSED_EVERY, Recipe};
use crate::rng::RandomState;

/// Inputs of one training run. Positions are already normalized (spherical).
pub struct Job<'a> {
    pub x: &'a [f32],
    pub d: usize,
    pub k: usize,
    pub recipe: &'a Recipe,
    pub spherical: bool,
    /// Revival threshold tau.
    pub tau: f64,
    pub seed: u32,
    pub positions: Vec<f32>,
    pub usage: Vec<f32>,
    pub capture: bool,
    pub verbose: bool,
    pub start: Instant,
    /// Epoch 0's permutation, already being drawn (see `model::train_rows`).
    pub first_order: Option<exec::Task<Vec<u32>>>,
}

pub struct Outcome {
    pub centroids: Vec<f32>,
    pub positions: Vec<f32>,
    pub usage: Vec<f32>,
    pub stats: Vec<EpochStats>,
    pub n_splits: usize,
    pub labels: Option<Vec<i32>>,
}

pub fn run<D: Device>(dev: &mut D, job: Job<'_>) -> Result<Outcome> {
    let Job {
        x,
        d,
        k,
        recipe,
        spherical,
        tau,
        seed,
        positions,
        usage,
        capture,
        verbose,
        start,
        first_order,
    } = job;
    ensure!(d > 0 && x.len().is_multiple_of(d), "x must be n x d");
    let n = x.len() / d;
    ensure!(n == recipe.n, "recipe is for {} rows, x has {n}", recipe.n);
    ensure!(
        positions.len() == k * d && usage.len() == k,
        "state must be k x d and k"
    );
    let metric = if spherical {
        Metric::InnerProduct
    } else {
        Metric::L2
    };
    let bsz = recipe.batch_size;
    let chunk = recipe.assignment_chunk_size.max(1);
    let n_epochs = recipe.epoch_rows.len();

    let data = dev.bind(x)?;
    let mut order_buf = dev.zeros_u32(n)?;
    let mut labels_ep = dev.zeros_u32(n)?;
    let mut scores_ep = dev.zeros_f32(n)?;
    let mut pos = dev.upload_f32(&positions)?;
    let mut mass = dev.upload_f32(&usage)?;
    let mut readout = dev.zeros_f32(if spherical { k * d } else { k })?;
    let mut sums = dev.zeros_f32(k * d)?;
    let mut counts = dev.zeros_u32(k)?;
    let mut pass = dev.zeros_f32(k * d)?;
    let mut split_rng = RandomState::new(seed);
    let mut n_splits = 0usize;
    let mut stats = Vec::with_capacity(n_epochs);
    let mut all_labels = capture.then(|| vec![-1i32; n]);
    let mut pass_counts = vec![0f64; k];
    let mut step = 0usize;

    let mut fin = None;
    std::thread::scope(|scope| -> Result<()> {
        let mut pending: Option<Pending<'_>> = None;
        let mut first_order = first_order;
        let skip_first = first_order.is_some() as usize;
        // One epoch's permutation ahead of the device.
        // Without threads, each epoch draws its own permutation when it starts.
        let (tx, rx) = mpsc::sync_channel::<Vec<u32>>(1);
        if exec::THREADS {
            scope.spawn(move || {
                for e in skip_first..n_epochs {
                    let p = RandomState::new(recipe::epoch_seed(seed, e)).permutation(n);
                    if tx.send(p).is_err() {
                        return;
                    }
                }
            });
        } else {
            drop(tx);
        }
        for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
            let order = match (epoch, first_order.take()) {
                (0, Some(h)) => h
                    .join()
                    .map_err(|_| anyhow::anyhow!("permutation thread panicked"))?,
                _ if !exec::THREADS => {
                    RandomState::new(recipe::epoch_seed(seed, epoch)).permutation(n)
                }
                _ => rx.recv().context("permutation thread stopped")?,
            };
            let last = epoch + 1 == n_epochs;
            dev.write_u32(&mut order_buf, 0, &order[..rows])?;
            let splits_before = n_splits;
            for b in (0..rows).step_by(bsz) {
                let e = (b + bsz).min(rows);
                let mat = Mat {
                    host: x,
                    dev: &data,
                    d,
                };
                // Readout: unit directions (spherical) or squared norms (L2).
                if spherical {
                    dev.normalize_rows(&pos, k, d, &mut readout)?;
                } else {
                    dev.row_norms(&pos, k, d, &mut readout)?;
                }
                let (cents, cnorm) = if spherical {
                    (&readout, None)
                } else {
                    (&pos, Some(&readout))
                };
                for c0 in (b..e).step_by(chunk) {
                    let c1 = (c0 + chunk).min(e);
                    dev.top1(
                        mat,
                        Select {
                            ids: Some(&order_buf),
                            rows: c0..c1,
                        },
                        cents,
                        cnorm,
                        k,
                        metric,
                        &mut labels_ep,
                        &mut scores_ep,
                    )?;
                }
                dev.scatter_add(
                    mat,
                    Select {
                        ids: Some(&order_buf),
                        rows: b..e,
                    },
                    &labels_ep,
                    k,
                    &mut sums,
                    &mut counts,
                    if last { Some(&mut pass) } else { None },
                )?;
                let alpha = recipe::scheduled_alpha(recipe.alpha0, step, recipe.total_steps);
                dev.ema_update(
                    &mut pos,
                    &mut mass,
                    &sums,
                    &counts,
                    k,
                    d,
                    EmaStep::new(alpha),
                )?;
                if (step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
                    let u = dev.read_f32(&mass, 0..k)?;
                    let plan = plan_revival(&u, tau, &mut split_rng);
                    if !plan.is_empty() {
                        dev.split(&mut pos, &mut mass, &plan, d)?;
                        n_splits += plan.len();
                    }
                } else {
                    dev.flush()?;
                }
                step += 1;
            }
            // Epoch end: the epoch's labels and scores come down; a helper
            // thread adds up the statistics while the next epoch's work is
            // queued (the device does not wait for them).
            let lab = dev.read_u32(&labels_ep, 0..rows)?;
            let sc = dev.read_f32(&scores_ep, 0..rows)?;
            let time = start.elapsed().as_secs_f64();
            let nsplit = n_splits - splits_before;
            let done = pending.take().map(|h: Pending<'_>| h.join());
            if let Some(r) = done {
                collect(r, &mut stats, &mut pass_counts, &mut all_labels)?;
            }
            let order = last.then_some(order);
            // The last pass writes the captured labels itself.
            let mut all = if last { all_labels.take() } else { None };
            pending = Some(exec::spawn_scoped(scope, move || {
                let mut obj_epoch = 0f64;
                for b in (0..rows).step_by(bsz) {
                    let e = (b + bsz).min(rows);
                    let mut obj_batch = 0f64;
                    for c0 in (b..e).step_by(chunk) {
                        let c1 = (c0 + chunk).min(e);
                        let mut obj = 0f64;
                        if spherical {
                            obj += 2.0 * (c1 - c0) as f64;
                            for &s in &sc[c0..c1] {
                                obj += (-2.0f32 * s) as f64;
                            }
                        } else {
                            for &s in &sc[c0..c1] {
                                obj += s as f64;
                            }
                        }
                        obj_batch += obj;
                    }
                    obj_epoch += obj_batch;
                }
                let mut hist = vec![0u64; k];
                for &l in &lab {
                    ensure!((l as usize) < k, "device returned label {l} >= k = {k}");
                    hist[l as usize] += 1;
                }
                let count_square: f64 = hist.iter().map(|&c| (c as f64) * (c as f64)).sum();
                let stat = EpochStats {
                    obj: obj_epoch,
                    time,
                    imbalance_factor: k as f64 * count_square / (rows as f64 * rows as f64),
                    nsplit,
                };
                if verbose {
                    eprintln!(
                        "  Epoch {} / {} ({} rows), iteration {} ({:.2} s): objective={}",
                        epoch + 1,
                        n_epochs,
                        rows,
                        step,
                        stat.time,
                        stat.obj
                    );
                }
                // The last pass: its counts (closing means) and, captured, its labels.
                let last_pass = order.map(|order| {
                    if let Some(all) = all.as_mut() {
                        for (&i, &l) in order[..rows].iter().zip(&lab) {
                            all[i as usize] = l as i32;
                        }
                    }
                    (hist, all)
                });
                Ok((stat, last_pass))
            }));
        }
        // The run's results come down while the last statistics finish.
        fin = Some((
            dev.read_f32(&pos, 0..k * d)?,
            dev.read_f32(&mass, 0..k)?,
            dev.read_f32(&pass, 0..k * d)?,
        ));
        if let Some(h) = pending.take() {
            collect(h.join(), &mut stats, &mut pass_counts, &mut all_labels)?;
        }
        Ok(())
    })?;

    let (positions, usage, closing) = fin.context("run produced no results")?;
    let centroids = closing_means(&positions, &closing, &pass_counts, d, spherical);
    Ok(Outcome {
        centroids,
        positions,
        usage,
        stats,
        n_splits,
        labels: all_labels,
    })
}

/// The last pass's label counts and, when captured, every row's label.
type LastPass = (Vec<u64>, Option<Vec<i32>>);
type EpochResult = Result<(EpochStats, Option<LastPass>)>;
type Pending<'s> = exec::ScopedTask<'s, EpochResult>;

/// Take one epoch's statistics from its helper thread, in epoch order.
fn collect(
    r: std::thread::Result<EpochResult>,
    stats: &mut Vec<EpochStats>,
    pass_counts: &mut [f64],
    all_labels: &mut Option<Vec<i32>>,
) -> Result<()> {
    let (stat, last) = r.map_err(|_| anyhow::anyhow!("epoch statistics thread panicked"))??;
    stats.push(stat);
    if let Some((hist, labels)) = last {
        for (p, &c) in pass_counts.iter_mut().zip(&hist) {
            *p = c as f64;
        }
        if labels.is_some() {
            *all_labels = labels;
        }
    }
    Ok(())
}
