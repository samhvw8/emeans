//! The cross-language fixture: synthetic data, the configs, and the results
//! of the Rust facade (`emeans`), called directly (not through this crate).
//!
//! Each binding's test reads the directory, runs the same calls through its
//! language, and checks the outputs equal these files byte for byte.
//! Write it with
//! `cargo run --release -p emeans-bindings-common --example write_fixture -- <dir>`.
//!
//! Files (raw little-endian): `x.f32` (n x d), `created.f64`, `priority.f32`,
//! `train.centroids.f32`, `train.labels.i32`, `assign.labels.u32`,
//! `assign.scores.f32`, `auto.{l1,l2}.u32`, `auto.{sim1,sim}.f32`,
//! `auto.outlier.u8`, `add.*` (same names, all rows), `dedup.group_of.u32`,
//! `dedup.representatives.u32`, `dedup.sizes.u32`, `dedup.nn_sim.f32`, and
//! `fixture.json` (shapes, configs, scalar results).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use emeans::synth::{NestedSpec, nested, with_bursts};
use emeans::{AutoConfig, Backend, Config, DedupConfig, DedupInput, Emeans, make_assigner};
use serde_json::json;

fn write<T: Copy>(dir: &Path, name: &str, v: &[T], to_le: impl Fn(T) -> Vec<u8>) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|&x| to_le(x)).collect();
    fs::write(dir.join(name), bytes).with_context(|| name.to_string())
}

fn f32s(dir: &Path, name: &str, v: &[f32]) -> Result<()> {
    write(dir, name, v, |x| x.to_le_bytes().to_vec())
}
fn u32s(dir: &Path, name: &str, v: &[u32]) -> Result<()> {
    write(dir, name, v, |x| x.to_le_bytes().to_vec())
}

fn labels(dir: &Path, prefix: &str, l: &emeans::Labels) -> Result<()> {
    u32s(dir, &format!("{prefix}.l1.u32"), &l.l1)?;
    u32s(dir, &format!("{prefix}.l2.u32"), &l.l2)?;
    f32s(dir, &format!("{prefix}.sim1.f32"), &l.sim1)?;
    f32s(dir, &format!("{prefix}.sim.f32"), &l.sim)?;
    let o: Vec<u8> = l.outlier.iter().map(|&b| b as u8).collect();
    fs::write(dir.join(format!("{prefix}.outlier.u8")), o)?;
    Ok(())
}

/// Write the fixture into `dir` (created when missing).
pub fn write_fixture(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)?;

    // Data: a nested mixture of unit rows plus bursts of near-copies.
    let d = 16;
    let base = nested(&NestedSpec {
        n: 1000,
        d,
        n_broad: 4,
        sub_range: (2, 4),
        seed: 7,
        ..Default::default()
    });
    let data = with_bursts(&base.x, d, 40, (2, 5), 0.05, 11);
    let x = data.x;
    let n = x.len() / d;
    let n0 = 1000;
    let priority: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
    f32s(dir, "x.f32", &x)?;
    write(dir, "created.f64", &data.created_unix, |v| {
        v.to_le_bytes().to_vec()
    })?;
    f32s(dir, "priority.f32", &priority)?;

    // train + assign. The bindings get the same values as JSON overrides.
    let k = 8;
    let train_over = json!({"seed": 7, "spherical": true, "nepochs": 4.0});
    let mut cfg = Config::new(d, k);
    cfg.seed = 7;
    cfg.spherical = true;
    cfg.nepochs = 4.0;
    let mut model = Emeans::with_assigner(cfg, make_assigner(Backend::cpu())?)?;
    let trained = model.train(&x)?;
    let a = model.assign(&x)?;
    // The two objectives the bindings report: the last epoch's moving mean
    // (free) and the exact one over the final centroids (opt-in).
    let last = model.iteration_stats().last().context("no epoch")?;
    let rows = *model
        .recipe()
        .context("recipe")?
        .epoch_rows
        .last()
        .context("rows")?;
    let objective = last.obj / rows as f64;
    let exact_objective = a.mean_objective(true);
    f32s(dir, "train.centroids.f32", &trained.centroids)?;
    write(
        dir,
        "train.labels.i32",
        trained.labels.as_deref().context("labels")?,
        |v| v.to_le_bytes().to_vec(),
    )?;
    u32s(dir, "assign.labels.u32", &a.labels)?;
    f32s(dir, "assign.scores.f32", &a.scores)?;

    // auto_cluster on the first n0 rows, then add the rest.
    let auto_over = json!({"seed": 7, "k1_range": [2, 8], "k2_max": 6, "restarts": 2,
                           "min_cluster_size": 10, "min_split_size": 30});
    let acfg = AutoConfig {
        seed: 7,
        k1_range: (2, 8),
        k2_max: 6,
        restarts: 2,
        min_cluster_size: 10,
        min_split_size: 30,
        ..Default::default()
    };
    let (mut h, l) = emeans::auto_cluster(&x[..n0 * d], d, &acfg)?;
    labels(dir, "auto", &l)?;
    let chosen = h.chosen_k();
    let report = h.add(&x, n0)?;
    labels(dir, "add", &report.labels)?;

    // dedup with time gaps and priorities.
    let dedup_over = json!({"sim_threshold": 0.97, "max_gap_secs": 60.0});
    let dcfg = DedupConfig {
        sim_threshold: 0.97,
        max_gap_secs: Some(60.0),
        ..Default::default()
    };
    let input = DedupInput {
        created_unix: Some(&data.created_unix),
        burst: None,
        priority: Some(&priority),
    };
    let g = emeans::dedup(&x, d, &input, &dcfg, Backend::cpu())?;
    u32s(dir, "dedup.group_of.u32", &g.group_of)?;
    u32s(dir, "dedup.representatives.u32", &g.representatives)?;
    u32s(dir, "dedup.sizes.u32", &g.sizes)?;
    f32s(dir, "dedup.nn_sim.f32", &g.nn_sim)?;

    let meta = json!({
        "n": n, "d": d, "k": k, "n0": n0,
        "train": {"config": train_over, "objective": objective,
                  "exact_objective": exact_objective,
                  "obj": model.obj()},
        "assign": {"metric": "inner_product"},
        "auto": {"config": auto_over, "k1": chosen.k1, "k2": chosen.k2},
        "add": {"action": report.action, "new_outliers": report.new_outliers,
                "pending": report.pending},
        "dedup": {"config": dedup_over, "n_groups": g.n_groups(),
                  "linked_pairs": g.linked_pairs},
    });
    fs::write(dir.join("fixture.json"), serde_json::to_vec_pretty(&meta)?)?;
    eprintln!(
        "fixture: n {n}, d {d}, k {k}, k1 {}, dedup groups {} (linked {}), add {:?} -> {}",
        chosen.k1,
        g.n_groups(),
        g.linked_pairs,
        report.action,
        dir.display()
    );
    Ok(())
}
