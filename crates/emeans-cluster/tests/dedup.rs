//! Exact kNN (CPU and Metal) and near-duplicate grouping.

use emeans_cluster::Backend;
use emeans_cluster::dedup::{DedupConfig, DedupInput, dedup};
use emeans_cluster::knn::{NONE, knn, knn_cpu, knn_self};
use emeans_cluster::metrics::ari;
use emeans_cluster::synth::{NestedSpec, Normal, nested, with_bursts};
use emeans_core::CpuKernel;
use emeans_core::model::normalize_rows;

// Accelerate is macOS only; the portable blocked kernel elsewhere.
#[cfg(target_os = "macos")]
const CPU: Backend = Backend::Cpu(CpuKernel::Accelerate);
#[cfg(not(target_os = "macos"))]
const CPU: Backend = Backend::Cpu(CpuKernel::Blocked);

fn random_unit(n: usize, d: usize, seed: u32) -> Vec<f32> {
    let mut r = Normal::new(seed);
    let mut x: Vec<f32> = (0..n * d).map(|_| r.sample()).collect();
    normalize_rows(&mut x, d);
    x
}

/// f64 brute force: best k ids per row (score desc, id asc), self excluded when asked.
fn brute(q: &[f32], x: &[f32], d: usize, k: usize, exclude: bool) -> Vec<Vec<(f64, u32)>> {
    let (m, n) = (q.len() / d, x.len() / d);
    (0..m)
        .map(|i| {
            let mut v: Vec<(f64, u32)> = (0..n)
                .filter(|&j| !(exclude && i == j))
                .map(|j| {
                    let s = (0..d)
                        .map(|t| q[i * d + t] as f64 * x[j * d + t] as f64)
                        .sum();
                    (s, j as u32)
                })
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            v.truncate(k);
            v
        })
        .collect()
}

fn check(nb: &emeans_core::Neighbors, want: &[Vec<(f64, u32)>], k: usize) {
    for (i, w) in want.iter().enumerate() {
        let (ids, sims) = nb.row(i);
        for t in 0..k {
            if t >= w.len() {
                assert_eq!(ids[t], NONE, "row {i} slot {t} should be padding");
                assert_eq!(sims[t], f32::NEG_INFINITY);
                continue;
            }
            assert!(
                (sims[t] as f64 - w[t].0).abs() < 1e-4,
                "row {i} slot {t}: {} vs {}",
                sims[t],
                w[t].0
            );
            if ids[t] != w[t].1 {
                // Only a tie may reorder ids.
                let other = w.iter().find(|p| p.1 == ids[t]).map(|p| p.0);
                assert!(
                    other.is_some_and(|s| (s - w[t].0).abs() < 1e-5),
                    "row {i} slot {t}: id {} vs {}",
                    ids[t],
                    w[t].1
                );
            }
        }
    }
}

#[test]
fn cpu_knn_matches_brute_force() {
    for &(n, d, k) in &[(300, 768, 10), (257, 13, 16), (5, 8, 10), (1, 8, 3)] {
        let x = random_unit(n, d, n as u32 + d as u32);
        let nb = knn_cpu(&x, &x, d, k, true).unwrap();
        check(&nb, &brute(&x, &x, d, k, true), k);
        let q = random_unit(37, d, 99);
        let nb = knn_cpu(&q, &x, d, k, false).unwrap();
        check(&nb, &brute(&q, &x, d, k, false), k);
    }
}

/// The GPU backends' top-k op (layer 1) through `knn` / `knn_self`.
#[test]
fn gpu_knn_matches_brute_force() {
    let mut gpus: Vec<Backend> = Vec::new();
    if cfg!(target_os = "macos") {
        gpus.push(Backend::MetalFused);
    }
    if cfg!(feature = "cuda") {
        gpus.push(Backend::Cuda);
    }
    for backend in gpus {
        if emeans_cluster::make_assigner(backend).is_err() {
            eprintln!("skipping {backend:?}: no device");
            continue;
        }
        for &(n, d, k) in &[(700, 768, 10), (130, 13, 16), (65, 64, 1), (5, 8, 10)] {
            let x = random_unit(n, d, 3 * n as u32 + d as u32);
            let nb = knn_self(&x, d, k, backend).unwrap();
            check(&nb, &brute(&x, &x, d, k, true), k);
            let q = random_unit(70, d, 5);
            let nb = knn(&q, &x, d, k, backend).unwrap();
            check(&nb, &brute(&q, &x, d, k, false), k);
        }
    }
}

fn small_bursts() -> (emeans_cluster::synth::WithBursts, usize) {
    let base = nested(&NestedSpec {
        n: 800,
        d: 128,
        n_broad: 4,
        sub_range: (2, 3),
        seed: 21,
        ..Default::default()
    });
    let d = 128;
    (with_bursts(&base.x, d, 40, (1, 6), 0.143, 5), d)
}

#[test]
fn dedup_recovers_injected_bursts() {
    let (w, d) = small_bursts();
    let r = dedup(
        &w.x,
        d,
        &DedupInput::default(),
        &DedupConfig::default(),
        CPU,
    )
    .unwrap();
    let truth = &w.truth;
    assert!(
        (ari(truth, &r.group_of) - 1.0).abs() < 1e-12,
        "groups differ from the injected bursts"
    );
    assert_eq!(r.n_groups(), 800);
    // Every representative is a member of its own group.
    for (g, &rep) in r.representatives.iter().enumerate() {
        assert_eq!(r.group_of[rep as usize], g as u32);
    }
    let labels: Vec<u32> = (0..r.n_groups() as u32).map(|g| g * 10).collect();
    let all = r.expand(&labels);
    for (i, &l) in all.iter().enumerate() {
        assert_eq!(l, r.group_of[i] * 10);
    }
    assert_eq!(r.gather(&w.x, d).len(), r.n_groups() * d);
}

#[test]
fn dedup_cuts_chains() {
    // A random walk: each step has cosine ~0.976 with the previous one, the
    // ends are far apart. Single linkage would make one group.
    let d = 256;
    let mut rng = Normal::new(8);
    let mut x = random_unit(1, d, 1);
    for t in 1..40 {
        let prev = x[(t - 1) * d..t * d].to_vec();
        let mut v: Vec<f32> = prev
            .iter()
            .zip(rng.direction(d))
            .map(|(p, z)| p + 0.22 * z)
            .collect();
        normalize_rows(&mut v, d);
        x.extend_from_slice(&v);
    }
    let cfg = DedupConfig::default();
    let r = dedup(&x, d, &DedupInput::default(), &cfg, CPU).unwrap();
    let dot = |a: usize, b: usize| -> f32 { (0..d).map(|t| x[a * d + t] * x[b * d + t]).sum() };
    let ends = dot(0, 39);
    assert!(ends < 0.8, "the walk should drift (ends cos {ends})");
    assert!(r.n_groups() > 1, "the chain was not cut");
    assert!(r.split_components >= 1);
    // Bounded diameter: any two members within twice the threshold angle.
    let t = cfg.sim_threshold;
    let min_pair = 2.0 * t * t - 1.0 - 1e-4;
    for g in 0..r.n_groups() {
        let m: Vec<usize> = (0..40).filter(|&i| r.group_of[i] == g as u32).collect();
        for &a in &m {
            for &b in &m {
                assert!(
                    dot(a, b) >= min_pair,
                    "group {g}: cos({a},{b}) = {}",
                    dot(a, b)
                );
            }
        }
    }
}

#[test]
fn bursts_time_gap_and_priority() {
    let d = 64;
    let mut x = random_unit(4, d, 77);
    // Row 4 is a near copy of row 0.
    let mut c: Vec<f32> = x[..d].iter().map(|v| v + 0.001).collect();
    normalize_rows(&mut c, d);
    x.extend_from_slice(&c);
    let t = [0.0, 10.0, 20.0, 30.0, 86_400.0];
    let cfg = DedupConfig::default();

    let r = dedup(&x, d, &DedupInput::default(), &cfg, CPU).unwrap();
    assert_eq!(r.group_of[0], r.group_of[4]);
    assert_eq!(r.n_groups(), 4);

    // A one-minute gap rule keeps a copy taken a day later apart.
    let gap = DedupConfig {
        max_gap_secs: Some(60.0),
        ..cfg.clone()
    };
    let inp = DedupInput {
        created_unix: Some(&t),
        ..Default::default()
    };
    let r = dedup(&x, d, &inp, &gap, CPU).unwrap();
    assert_ne!(r.group_of[0], r.group_of[4]);

    // Unknown time: the rule does not apply.
    let t_nan = [0.0, 10.0, 20.0, 30.0, f64::NAN];
    let inp = DedupInput {
        created_unix: Some(&t_nan),
        ..Default::default()
    };
    let r = dedup(&x, d, &inp, &gap, CPU).unwrap();
    assert_eq!(r.group_of[0], r.group_of[4]);

    // Burst ids group unrelated rows 1 and 2.
    let burst = [None, Some(7), Some(7), None, None];
    let inp = DedupInput {
        burst: Some(&burst),
        ..Default::default()
    };
    let r = dedup(&x, d, &inp, &cfg, CPU).unwrap();
    assert_eq!(r.group_of[1], r.group_of[2]);
    assert_eq!(r.n_groups(), 3);

    // Priority picks the representative.
    for fav in [0usize, 4] {
        let mut p = [0f32; 5];
        p[fav] = 1.0;
        let inp = DedupInput {
            priority: Some(&p),
            ..Default::default()
        };
        let r = dedup(&x, d, &inp, &cfg, CPU).unwrap();
        assert_eq!(r.representatives[r.group_of[0] as usize], fav as u32);
    }
}

/// Unit rows built from a few basis directions: `parts` = (axis, weight).
fn crafted(d: usize, parts: &[(usize, f32)]) -> Vec<f32> {
    let mut v = vec![0f32; d];
    for &(a, w) in parts {
        v[a] = w;
    }
    normalize_rows(&mut v, d);
    v
}

#[test]
fn burst_joins_a_star_only_when_every_row_fits() {
    // Anchor A with three rim rows at cosine 0.96 (0.9216 to each other), and
    // a burst of two rows: b1 on the rim, b2 unrelated to A.
    let d = 32;
    let mut x = crafted(d, &[(0, 1.0)]);
    for a in 1..=3 {
        x.extend(crafted(d, &[(0, 0.96), (a, 0.28)]));
    }
    x.extend(crafted(d, &[(0, 0.96), (4, 0.28)])); // b1 = row 4
    x.extend(crafted(d, &[(9, 1.0)])); // b2 = row 5
    let burst = [None, None, None, None, Some(1), Some(1)];
    let inp = DedupInput {
        burst: Some(&burst),
        ..Default::default()
    };
    let cfg = DedupConfig::default();
    let r = dedup(&x, d, &inp, &cfg, CPU).unwrap();
    assert_eq!(r.group_of[4], r.group_of[5], "a burst stays together");
    assert_ne!(
        r.group_of[5], r.group_of[0],
        "a burst row unrelated to the anchor was pulled into its group"
    );
    assert_eq!(r.anchors.len(), r.n_groups());
    assert_eq!(r.anchors[r.group_of[0] as usize], 0);
    // The documented bound: every member outside the anchor's own burst is
    // within the threshold of the anchor.
    let dot = |a: usize, b: usize| -> f32 { (0..d).map(|t| x[a * d + t] * x[b * d + t]).sum() };
    for i in 0..6 {
        let a = r.anchors[r.group_of[i] as usize] as usize;
        assert_eq!(r.group_of[a], r.group_of[i], "anchor outside its group");
        let same_burst = burst[i].is_some() && burst[i] == burst[a];
        assert!(
            same_burst || dot(i, a) >= cfg.sim_threshold - 1e-6,
            "row {i}: cos to anchor {a} = {}",
            dot(i, a)
        );
    }
}
