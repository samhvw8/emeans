//! Validate automatic k and incremental updates on synthetic nested data.
//!
//! cargo run --release -p emeans-bench --bin autok_synth -- \
//!     [--n 16000] [--seeds 7,8,9] [--criteria silhouette+db,silhouette,ch,db] [--backend cpu] \
//!     [--restarts 4] [--tolerance1 0] [--tolerance2 0.05] [--nepochs 5] [--init pp|random]
//!     [--level1 bottomup|topdown] [--resplit 2] [--broad 8] [--subs 3,12]
//!     [--no-incremental] [--verbose]
//!
//! Prints markdown tables:
//! 1. Auto-k accuracy per data seed and criterion: chosen k1 vs 8 broad groups,
//!    level-2 k vs the number of true sub groups whose majority lands in that
//!    level-1 group (a merged group holds both broad groups' subs), ARI/NMI
//!    of level 1 vs broad groups and of the leaves vs sub groups (noise rows excluded).
//! 2. Incremental: train on 80%, add 20% in 4 chunks; action, label stability of
//!    old rows, objective, and a full retrain on 100% for comparison.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Context;
use emeans::auto::Level1Method;
use emeans::auto::{GroupDecision, label_stability};
use emeans::metrics::{ari, nmi};
use emeans::synth::{Nested, NestedSpec, nested};
use emeans::{AutoConfig, Backend, Criterion, Init, Labels, auto_cluster};

struct Args {
    n: usize,
    seeds: Vec<u32>,
    criteria: Vec<(Criterion, Criterion)>,
    backend: Option<Backend>,
    incremental: bool,
    verbose: bool,
    restarts: Option<usize>,
    tolerance1: Option<f64>,
    tolerance2: Option<f64>,
    nepochs: Option<f64>,
    init: Option<Init>,
    level1: Option<Level1Method>,
    resplit: Option<usize>,
    broad: usize,
    subs: (usize, usize),
}

fn parse() -> anyhow::Result<Args> {
    let mut a = Args {
        n: 16_000,
        seeds: vec![7, 8, 9],
        criteria: vec![(Criterion::Silhouette, Criterion::DaviesBouldin)],
        backend: None,
        incremental: true,
        verbose: false,
        restarts: None,
        tolerance1: None,
        tolerance2: None,
        nepochs: None,
        init: None,
        level1: None,
        resplit: None,
        broad: 8,
        subs: (3, 12),
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        match v[i].as_str() {
            "--n" => {
                a.n = v[i + 1].parse()?;
                i += 1;
            }
            "--seeds" => {
                a.seeds = v[i + 1]
                    .split(',')
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?;
                i += 1;
            }
            "--criteria" => {
                let one = |s: &str| match s {
                    "silhouette" => Ok(Criterion::Silhouette),
                    "ch" => Ok(Criterion::CalinskiHarabasz),
                    "db" => Ok(Criterion::DaviesBouldin),
                    _ => Err(anyhow::anyhow!("unknown criterion {s}")),
                };
                // "a" = a at both levels; "a+b" = a at level 1, b at level 2.
                a.criteria = v[i + 1]
                    .split(',')
                    .map(|s| match s.split_once('+') {
                        Some((l1, l2)) => Ok((one(l1)?, one(l2)?)),
                        None => Ok((one(s)?, one(s)?)),
                    })
                    .collect::<anyhow::Result<_>>()?;
                i += 1;
            }
            "--backend" => {
                a.backend = Some(emeans_bench::backend(&v[i + 1])?);
                i += 1;
            }
            "--restarts" => {
                a.restarts = Some(v[i + 1].parse()?);
                i += 1;
            }
            "--tolerance1" => {
                a.tolerance1 = Some(v[i + 1].parse()?);
                i += 1;
            }
            "--tolerance2" => {
                a.tolerance2 = Some(v[i + 1].parse()?);
                i += 1;
            }
            "--nepochs" => {
                a.nepochs = Some(v[i + 1].parse()?);
                i += 1;
            }
            "--init" => {
                a.init = Some(match v[i + 1].as_str() {
                    "pp" => Init::KMeansPlusPlus,
                    "random" => Init::RandomRows,
                    o => anyhow::bail!("unknown init {o}"),
                });
                i += 1;
            }
            "--level1" => {
                a.level1 = Some(match v[i + 1].as_str() {
                    "bottomup" => Level1Method::BottomUp,
                    "topdown" => Level1Method::TopDown,
                    o => anyhow::bail!("unknown level-1 method {o}"),
                });
                i += 1;
            }
            "--resplit" => {
                a.resplit = Some(v[i + 1].parse()?);
                i += 1;
            }
            "--broad" => {
                a.broad = v[i + 1].parse()?;
                i += 1;
            }
            "--subs" => {
                let (lo, hi) = v[i + 1].split_once(',').context("--subs lo,hi")?;
                a.subs = (lo.parse()?, hi.parse()?);
                i += 1;
            }
            "--no-incremental" => a.incremental = false,
            "--verbose" => a.verbose = true,
            s => anyhow::bail!("unknown argument {s}"),
        }
        i += 1;
    }
    Ok(a)
}

/// Rows that belong to a group (noise excluded).
fn clean(data: &Nested) -> Vec<usize> {
    (0..data.broad.len())
        .filter(|&i| data.broad[i] >= 0)
        .collect()
}

fn scores(data: &Nested, l: &Labels, rows: &[usize]) -> (f64, f64, f64, f64) {
    let tb: Vec<u32> = rows.iter().map(|&i| data.broad[i] as u32).collect();
    let ts: Vec<u32> = rows.iter().map(|&i| data.sub[i] as u32).collect();
    let p1: Vec<u32> = rows.iter().map(|&i| l.l1[i]).collect();
    let codes = l.leaf_codes();
    let p2: Vec<u32> = rows.iter().map(|&i| codes[i]).collect();
    (ari(&tb, &p1), nmi(&tb, &p1), ari(&ts, &p2), nmi(&ts, &p2))
}

fn config(args: &Args, criteria: (Criterion, Criterion)) -> AutoConfig {
    let mut c = AutoConfig {
        criterion1: criteria.0,
        criterion2: criteria.1,
        verbose: args.verbose,
        ..Default::default()
    };
    if let Some(b) = args.backend {
        c.backend = b;
    }
    if let Some(r) = args.restarts {
        c.restarts = r;
    }
    if let Some(t) = args.tolerance1 {
        c.level1_tolerance = t;
    }
    if let Some(t) = args.tolerance2 {
        c.level2_tolerance = t;
    }
    if let Some(e) = args.nepochs {
        c.nepochs = e;
    }
    if let Some(i) = args.init {
        c.init = i;
    }
    if let Some(l) = args.level1 {
        c.level1_method = l;
    }
    if let Some(r) = args.resplit {
        c.resplit_passes = r;
    }
    c
}

fn criteria_name(c: (Criterion, Criterion)) -> String {
    if c.0 == c.1 {
        criterion_name(c.0).to_string()
    } else {
        format!("{} + {}", criterion_name(c.0), criterion_name(c.1))
    }
}

fn criterion_name(c: Criterion) -> &'static str {
    match c {
        Criterion::Silhouette => "silhouette",
        Criterion::CalinskiHarabasz => "calinski-harabasz",
        Criterion::DaviesBouldin => "davies-bouldin",
    }
}

fn main() -> anyhow::Result<()> {
    let args = parse()?;
    let c0 = config(&args, args.criteria[0]);
    println!(
        "Settings: level 1 {:?}, resplit passes {}, restarts {}, nepochs {}, level-1 tolerance {}, level-2 tolerance {}, init {:?}, backend {:?}; data {} broad groups of {:?} subs\n",
        c0.level1_method,
        c0.resplit_passes,
        c0.restarts,
        c0.nepochs,
        c0.level1_tolerance,
        c0.level2_tolerance,
        c0.init,
        c0.backend,
        args.broad,
        args.subs
    );
    println!(
        "## Auto-k on nested synthetic data (n={}, d=768, {} broad groups)\n",
        args.n, args.broad
    );
    println!(
        "| data seed | criterion | true subs per broad | k1 | level-2 k (true) per group | exact k2 | |k2 err| mean | leaves (true) | resplits | flat | ARI L1 | NMI L1 | ARI leaves | NMI leaves | seconds |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut first: Option<(Nested, AutoConfig)> = None;
    for &seed in &args.seeds {
        let data = nested(&NestedSpec {
            n: args.n,
            seed,
            n_broad: args.broad,
            sub_range: args.subs,
            ..Default::default()
        });
        let rows = clean(&data);
        if args.verbose {
            // Criterion values of the true partitions (noise rows as one extra cluster).
            let nb = data.subs_per_broad.len() as u32;
            let ns: u32 = data.subs_per_broad.iter().sum::<usize>() as u32;
            let tb: Vec<u32> = data
                .broad
                .iter()
                .map(|&b| if b < 0 { nb } else { b as u32 })
                .collect();
            let ts: Vec<u32> = data
                .sub
                .iter()
                .map(|&b| if b < 0 { ns } else { b as u32 })
                .collect();
            let ib = emeans::metrics::internal_scores(&data.x, data.d, &tb, nb as usize + 1)?;
            let is = emeans::metrics::internal_scores(&data.x, data.d, &ts, ns as usize + 1)?;
            eprintln!(
                "truth seed {seed}: broad sil={:.4} db={:.3} ch={:.1} obj={:.4}; subs sil={:.4} db={:.3} ch={:.1} obj={:.4}",
                ib.silhouette,
                ib.davies_bouldin,
                ib.calinski_harabasz,
                ib.objective,
                is.silhouette,
                is.davies_bouldin,
                is.calinski_harabasz,
                is.objective
            );
        }
        for &crit in &args.criteria {
            let cfg = config(&args, crit);
            let t = Instant::now();
            let (h, l) = auto_cluster(&data.x, data.d, &cfg)?;
            let secs = t.elapsed().as_secs_f64();
            let ck = h.chosen_k();
            // The true level-2 count of a level-1 group: the true sub groups
            // whose majority of rows lands in it (a merged group holds both
            // broad groups' subs).
            let mut slots_of_sub: HashMap<i32, HashMap<u32, usize>> = HashMap::new();
            for &i in &rows {
                *slots_of_sub
                    .entry(data.sub[i])
                    .or_default()
                    .entry(l.slot1[i])
                    .or_default() += 1;
            }
            let mut true_k2 = vec![0usize; ck.k1];
            for m in slots_of_sub.values() {
                let (&g, _) = m.iter().max_by_key(|(g, c)| (**c, -(**g as i64))).unwrap();
                true_k2[g as usize] += 1;
            }
            let mut cells = Vec::new();
            let (mut exact, mut err, mut cnt) = (0, 0usize, 0);
            for (g, &truth) in true_k2.iter().enumerate() {
                if truth == 0 {
                    cells.push(format!("{}(0)", ck.k2[g]));
                    continue;
                }
                let tag = match &h.groups[g].decision {
                    GroupDecision::Split { .. } => "",
                    GroupDecision::NoStructure { .. } => "n",
                    GroupDecision::TooSmall => "s",
                    GroupDecision::Empty => "e",
                };
                cells.push(format!("{}{tag}({truth})", ck.k2[g]));
                exact += (ck.k2[g] == truth) as usize;
                err += ck.k2[g].abs_diff(truth);
                cnt += 1;
            }
            let (a1, n1, a2, n2) = scores(&data, &l, &rows);
            println!(
                "| {seed} | {} | {:?} | {} | {} | {exact}/{cnt} | {:.2} | {} ({}) | {} | {} | {a1:.3} | {n1:.3} | {a2:.3} | {n2:.3} | {secs:.1} |",
                criteria_name(crit),
                data.subs_per_broad,
                ck.k1,
                cells.join(" "),
                err as f64 / cnt.max(1) as f64,
                ck.k2.iter().sum::<usize>(),
                data.subs_per_broad.iter().sum::<usize>(),
                h.diagnostics.resplits,
                if h.diagnostics.flat { "yes" } else { "no" },
            );
        }
        if first.is_none() {
            first = Some((data, config(&args, args.criteria[0])));
        }
    }
    println!(
        "\nLevel-2 cells: chosen k, a tag (n = no structure, s = too small, e = empty), and in parentheses the true sub groups whose majority lands in the group (0 = a group of noise rows, not counted)."
    );

    if !args.incremental {
        return Ok(());
    }
    let (data, cfg) = first.unwrap();
    let d = data.d;
    let n = data.x.len() / d;
    let n0 = n * 8 / 10;
    let rows = clean(&data);
    println!(
        "\n## Incremental: train on {n0} rows, add {} rows in 4 chunks (data seed {})\n",
        n - n0,
        args.seeds[0]
    );
    let t = Instant::now();
    let (mut h, l0) = auto_cluster(&data.x[..n0 * d], d, &cfg)?;
    let train_secs = t.elapsed().as_secs_f64();
    let dir = std::env::temp_dir().join(format!("emeans-autok-synth-{}", std::process::id()));
    h.save(&dir)?;
    h = emeans::Hierarchy::load(&dir, cfg.backend)?;
    std::fs::remove_dir_all(&dir)?;
    println!(
        "| step | rows | action | new outliers | pending | old rows with unchanged leaf id | objective (all rows so far) | ARI leaves (clean rows so far) | seconds |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let rows0: Vec<usize> = rows.iter().copied().filter(|&i| i < n0).collect();
    println!(
        "| train 80% | {n0} | auto_cluster (k1={}) | - | - | - | {:.4} | {:.3} | {train_secs:.1} |",
        h.chosen_k().k1,
        l0.objective(),
        scores(&data, &l0, &rows0).2
    );
    let chunk = (n - n0).div_ceil(4);
    let mut seen = n0;
    let mut last = l0.clone();
    while seen < n {
        let end = (seen + chunk).min(n);
        let t = Instant::now();
        let r = h.add(&data.x[..end * d], seen)?;
        let secs = t.elapsed().as_secs_f64();
        let old = label_stability(&l0, &r.labels.slice(0..n0));
        let rs: Vec<usize> = rows.iter().copied().filter(|&i| i < end).collect();
        println!(
            "| add {}..{} | {end} | {:?} | {} | {} | {:.4} | {:.4} | {:.3} | {secs:.2} |",
            seen,
            end,
            r.action,
            r.new_outliers,
            r.pending,
            old,
            r.labels.objective(),
            scores(&data, &r.labels, &rs).2
        );
        seen = end;
        last = r.labels;
    }
    let t = Instant::now();
    let (hf, lf) = auto_cluster(&data.x, d, &cfg)?;
    println!(
        "| full retrain 100% | {n} | auto_cluster (k1={}) | - | - | {:.4} (ids not matched) | {:.4} | {:.3} | {:.1} |",
        hf.chosen_k().k1,
        label_stability(&l0, &lf.slice(0..n0)),
        lf.objective(),
        scores(&data, &lf, &rows).2,
        t.elapsed().as_secs_f64()
    );
    println!(
        "\nIncremental objective / full-retrain objective: {:.4}",
        last.objective() / lf.objective()
    );
    Ok(())
}
