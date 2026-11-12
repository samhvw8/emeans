//! Behaviour tests for the e-means model: streaming API, save/load, resume,
//! subsampling, spherical invariants and loud failures.

mod common;

use emeans_core::{Config, CpuAssigner, CpuKernel, Emeans, Rows};

fn cpu() -> Box<CpuAssigner> {
    Box::new(CpuAssigner::new(CpuKernel::Blocked))
}

fn model(cfg: Config) -> Emeans {
    Emeans::with_assigner(cfg, cpu()).unwrap()
}

#[test]
fn streaming_api_reproduces_train_bit_for_bit() {
    let (n, d, k) = (3000, 12, 20);
    let x = common::mixture(n, d, 15, 0.6, false, 3);
    let mut cfg = Config::new(d, k);
    cfg.nepochs = 2.5;
    let mut a = model(cfg.clone());
    let trained = a.train(&x).unwrap();

    // Same draws as train(): init from RandomState(seed).permutation, epoch orders
    // from RandomState(epoch_seed). Feed each batch in two uneven pieces.
    let mut b = model(cfg.clone());
    let perm = emeans_core::rng::RandomState::new(cfg.seed).permutation(n);
    let init: Vec<f32> = perm[..k]
        .iter()
        .flat_map(|&i| x[i as usize * d..(i as usize + 1) * d].to_vec())
        .collect();
    b.begin(n, &init).unwrap();
    let recipe = b.recipe().unwrap().clone();
    for (e, &rows) in recipe.epoch_rows.iter().enumerate() {
        let order =
            emeans_core::rng::RandomState::new(emeans_core::recipe::epoch_seed(cfg.seed, e))
                .permutation(n);
        for s in (0..rows).step_by(recipe.batch_size) {
            let ids = &order[s..(s + recipe.batch_size).min(rows)];
            let cut = ids.len() / 3;
            let first = b
                .accumulate_rows(Rows::Indexed {
                    data: &x,
                    ids: &ids[..cut],
                })
                .unwrap();
            assert!(!first.complete || cut == ids.len());
            let dense: Vec<f32> = ids[cut..]
                .iter()
                .flat_map(|&i| x[i as usize * d..(i as usize + 1) * d].to_vec())
                .collect();
            let second = b.accumulate(&dense).unwrap();
            assert!(second.complete);
            b.update().unwrap();
        }
        b.end_epoch().unwrap();
    }
    let c = b.finalize().unwrap();
    assert_eq!(c, &trained.centroids[..]);
    // Objectives are f64 sums per piece, so splitting a batch changes the last bits.
    for (p, q) in a.obj().iter().zip(b.obj()) {
        assert!((p - q).abs() <= 1e-12 * p.abs(), "{p} vs {q}");
    }
}

#[test]
fn fractional_epochs_and_stats() {
    let (n, d, k) = (2000, 6, 10);
    let x = common::mixture(n, d, 8, 0.5, false, 9);
    let mut cfg = Config::new(d, k);
    cfg.nepochs = 2.5;
    let mut m = model(cfg);
    m.train(&x).unwrap();
    assert_eq!(m.recipe().unwrap().epoch_rows, vec![2000, 1000, 2000]);
    assert_eq!(m.iteration_stats().len(), 3);
    for s in m.iteration_stats() {
        assert!(s.obj.is_finite() && s.obj > 0.0);
        assert!(s.imbalance_factor >= 1.0 - 1e-9);
    }
    let mut cfg = Config::new(d, k);
    cfg.nepochs = 0.5;
    let mut m = model(cfg);
    m.train(&x).unwrap();
    assert_eq!(m.recipe().unwrap().epoch_rows, vec![1000]);
}

#[test]
fn spherical_centroids_are_unit_and_scores_are_cosines() {
    let (n, d, k) = (2500, 32, 16);
    let x = common::mixture(n, d, 10, 0.4, true, 5);
    let mut cfg = Config::new(d, k);
    cfg.spherical = true;
    let mut m = model(cfg);
    let t = m.train(&x).unwrap();
    for c in t.centroids.chunks(d) {
        let norm = c.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
    }
    let a = m.assign(&x).unwrap();
    assert!(a.scores.iter().all(|&s| (-1.0001..=1.0001).contains(&s)));
    let obj = a.mean_objective(true);
    assert!(obj > 0.0 && obj < 2.0);
}

#[test]
fn dead_clusters_are_revived_by_splitting() {
    let (n, d, k) = (4000, 4, 16);
    let x = common::mixture(n, d, 16, 0.3, false, 11);
    let mut init = vec![0f32; k * d];
    // 15 centroids far from all data: none receives rows, all must be revived.
    for j in 1..k {
        for t in 0..d {
            init[j * d + t] = 1e3 + j as f32;
        }
    }
    init[..d].copy_from_slice(&x[..d]);
    let mut m = model(Config::new(d, k));
    m.train_from(&x, Some(&init)).unwrap();
    assert!(m.n_splits() >= 15, "splits {}", m.n_splits());
    let used: std::collections::HashSet<u32> = m.assign(&x).unwrap().labels.into_iter().collect();
    assert!(used.len() >= 12, "only {} clusters used", used.len());
}

#[test]
fn max_points_per_centroid_subsamples() {
    let (n, d, k) = (5000, 5, 10);
    let x = common::mixture(n, d, 10, 0.5, false, 2);
    let mut cfg = Config::new(d, k);
    cfg.max_points_per_centroid = Some(100);
    let mut m = model(cfg);
    let t = m.train(&x).unwrap();
    let labels = t.labels.unwrap();
    assert_eq!(labels.iter().filter(|&&l| l == -1).count(), n - 1000);
    assert!(labels.iter().all(|&l| l >= -1 && l < k as i32));
    assert_eq!(m.recipe().unwrap().n, 1000);
}

#[test]
fn save_load_roundtrip_and_resume() {
    let (n, d, k) = (3000, 16, 12);
    let x = common::mixture(n, d, 12, 0.5, true, 4);
    let mut cfg = Config::new(d, k);
    cfg.spherical = true;
    let mut m = model(cfg);
    m.train(&x).unwrap();
    let dir = std::env::temp_dir().join(format!("emeans-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.emeans");
    m.save(&path).unwrap();
    let mut l = Emeans::load_with(&path, cpu()).unwrap();
    assert_eq!(l.centroids(), m.centroids());
    assert_eq!(l.ema_state(), m.ema_state());
    assert_eq!(l.recipe(), m.recipe());
    assert_eq!(l.config(), m.config());
    assert_eq!(l.obj(), m.obj());
    let (a, b) = (m.assign(&x).unwrap(), l.assign(&x).unwrap());
    assert_eq!(a.labels, b.labels);
    assert_eq!(a.scores, b.scores);

    // Continue training on new data from the saved EMA state.
    let more = common::mixture(1500, d, 12, 0.5, true, 40);
    let before = l.assign(&more).unwrap().mean_objective(true);
    l.config_mut().unwrap().nepochs = 2.0;
    l.train_resume(&more).unwrap();
    let after = l.assign(&more).unwrap().mean_objective(true);
    assert!(
        after <= before * 1.05,
        "resume made it worse: {before} -> {after}"
    );
    assert_eq!(l.iteration_stats().len(), 2);

    // A truncated file fails loudly.
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
    assert!(Emeans::load(&path).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn misuse_fails_loudly() {
    let (n, d, k) = (1000, 4, 8);
    let x = common::mixture(n, d, 8, 0.5, false, 1);
    let mut m = model(Config::new(d, k));
    assert!(m.update().is_err(), "update without a run");
    let mut m = model(Config::new(d, k));
    m.begin(n, &x[..k * d]).unwrap();
    let b = m.local_batch_size().unwrap();
    // More rows than the logical batch.
    assert!(m.accumulate(&x[..(b + 1) * d]).is_err());
    // The failed object refuses further work.
    assert!(m.accumulate(&x[..d]).is_err());

    let mut m = model(Config::new(d, k));
    m.begin(n, &x[..k * d]).unwrap();
    m.accumulate(&x[..d]).unwrap();
    assert!(m.update().is_err(), "incomplete batch");

    let mut m = model(Config::new(d, k));
    m.begin(n, &x[..k * d]).unwrap();
    assert!(m.end_epoch().is_err(), "epoch ended early");

    let mut m = model(Config::new(d, k));
    m.begin(n, &x[..k * d]).unwrap();
    assert!(m.finalize().is_err(), "finalize before the last epoch");
    assert!(m.save(std::env::temp_dir().join("never.emeans")).is_err());

    assert!(
        model(Config::new(d, k)).train(&x[..(k - 1) * d]).is_err(),
        "n < k"
    );
}
