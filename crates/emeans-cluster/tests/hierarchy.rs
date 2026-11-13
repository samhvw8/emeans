//! Automatic k (two levels) and incremental updates on small synthetic data.

use emeans_cluster::auto::label_stability;
use emeans_cluster::incremental::IncrementalConfig;
use emeans_cluster::metrics::ari;
use emeans_cluster::synth::{Nested, NestedSpec, nested};
use emeans_cluster::{Action, AutoConfig, Backend, Hierarchy, auto_cluster};
use emeans_core::CpuKernel;

// Accelerate is macOS only; the portable blocked kernel elsewhere.
#[cfg(target_os = "macos")]
const CPU: Backend = Backend::Cpu(CpuKernel::Accelerate);
#[cfg(not(target_os = "macos"))]
const CPU: Backend = Backend::Cpu(CpuKernel::Blocked);

fn data() -> Nested {
    nested(&NestedSpec {
        n: 3000,
        d: 96,
        n_broad: 4,
        sub_range: (2, 4),
        min_sub: 60,
        noise_frac: 0.0,
        seed: 31,
        ..Default::default()
    })
}

fn cfg() -> AutoConfig {
    AutoConfig {
        k1_range: (2, 10),
        k2_max: 8,
        backend: CPU,
        ..Default::default()
    }
}

fn broad_ari(data: &Nested, l1: &[u32]) -> f64 {
    let t: Vec<u32> = data.broad.iter().map(|&b| b as u32).collect();
    ari(&t, l1)
}

fn leaf_ari(data: &Nested, codes: &[u32]) -> f64 {
    let t: Vec<u32> = data.sub.iter().map(|&b| b as u32).collect();
    ari(&t, codes)
}

#[test]
fn auto_cluster_finds_both_levels() {
    let data = data();
    let (h, l) = auto_cluster(&data.x, data.d, &cfg()).unwrap();
    let ck = h.chosen_k();
    assert_eq!(ck.k1, 4, "k1");
    assert_eq!(h.diagnostics.stage_a_level1.len(), 9);
    assert!(!h.diagnostics.level1.is_empty() && !h.diagnostics.flat);
    assert!(broad_ari(&data, &l.l1) > 0.98, "level-1 ARI");
    let leaves: usize = ck.k2.iter().sum();
    let truth: usize = data.subs_per_broad.iter().sum();
    assert!(
        leaves.abs_diff(truth) <= 2,
        "{leaves} leaves vs {truth} true sub groups"
    );
    assert!(leaf_ari(&data, &l.leaf_codes()) > 0.85, "leaf ARI");
    // Labels are consistent with the models.
    let again = h.assign(&data.x).unwrap();
    assert_eq!(again, l);
}

#[test]
fn incremental_adds_keep_ids_and_survive_save_load() {
    let data = data();
    let d = data.d;
    let n = data.x.len() / d;
    let n0 = n * 8 / 10;
    let (mut h, l0) = auto_cluster(&data.x[..n0 * d], d, &cfg()).unwrap();

    // Save, load, same labels.
    let dir = std::env::temp_dir().join(format!("emeans-hier-test-{}", std::process::id()));
    h.save(&dir).unwrap();
    let h2 = Hierarchy::load(&dir, CPU).unwrap();
    assert_eq!(h2.assign(&data.x[..n0 * d]).unwrap(), l0);
    assert_eq!(h2.chosen_k(), h.chosen_k());
    std::fs::remove_dir_all(&dir).unwrap();

    // The same through one byte blob; a cut or padded blob fails.
    let blob = h.to_bytes().unwrap();
    let h3 = Hierarchy::from_bytes(&blob, CPU).unwrap();
    assert_eq!(h3.assign(&data.x[..n0 * d]).unwrap(), l0);
    assert_eq!(h3.chosen_k(), h.chosen_k());
    assert_eq!(h3.to_bytes().unwrap(), blob);
    assert!(Hierarchy::from_bytes(&blob[..blob.len() - 1], CPU).is_err());
    let mut padded = blob.clone();
    padded.push(0);
    assert!(Hierarchy::from_bytes(&padded, CPU).is_err());

    // Rows are append-only.
    assert!(h.add(&data.x, n0 + 1).is_err());

    let chunk = (n - n0) / 4;
    let mut seen = n0;
    let mut actions = Vec::new();
    let mut last = None;
    while seen < n {
        let end = (seen + chunk).min(n);
        let r = h.add(&data.x[..end * d], seen).unwrap();
        assert_eq!(r.labels.len(), end);
        actions.push(r.action);
        seen = end;
        last = Some(r.labels);
    }
    let last = last.unwrap();
    assert_eq!(h.n_seen(), n);
    // 150 rows per chunk against 2400 is 6%: the second chunk passes 10% and updates.
    assert!(actions.contains(&Action::EmaUpdate), "{actions:?}");
    assert!(!actions.contains(&Action::FullRetrain), "{actions:?}");
    let stable = label_stability(&l0, &last.slice(0..n0));
    assert!(stable > 0.97, "label stability {stable}");
    assert!(leaf_ari(&data, &last.leaf_codes()) > 0.85);
}

#[test]
fn full_retrain_carries_ids() {
    let data = data();
    let d = data.d;
    let n = data.x.len() / d;
    let n0 = n / 2;
    let mut c = cfg();
    c.incremental = IncrementalConfig {
        retrain_added_fraction: 0.5,
        ..Default::default()
    };
    let (mut h, l0) = auto_cluster(&data.x[..n0 * d], d, &c).unwrap();
    let r = h.add(&data.x, n0).unwrap();
    assert_eq!(r.action, Action::FullRetrain);
    // The same broad groups exist before and after, so level-1 ids carry over.
    let stable1 = (0..n0).filter(|&i| l0.l1[i] == r.labels.l1[i]).count() as f64 / n0 as f64;
    assert!(stable1 > 0.98, "level-1 id stability {stable1}");
    let stable = label_stability(&l0, &r.labels.slice(0..n0));
    assert!(stable > 0.9, "leaf id stability {stable}");
}

#[test]
fn kmeans_pp_picks_distinct_rows_across_groups() {
    let data = data();
    let d = data.d;
    let picks = emeans_cluster::auto::kmeans_pp(&data.x, d, 12, 5);
    assert_eq!(picks.len(), 12);
    let mut u = picks.clone();
    u.sort_unstable();
    u.dedup();
    assert_eq!(u.len(), 12, "duplicate picks");
    // D^2 sampling reaches every broad group within the first 8 picks.
    let groups: std::collections::HashSet<i32> =
        picks[..8].iter().map(|&i| data.broad[i as usize]).collect();
    assert_eq!(groups.len(), 4, "{groups:?}");
    // Same seed, same picks; more rows than kmax are never returned.
    assert_eq!(emeans_cluster::auto::kmeans_pp(&data.x, d, 12, 5), picks);
    assert_eq!(
        emeans_cluster::auto::kmeans_pp(&data.x[..3 * d], d, 12, 5).len(),
        3
    );
}

#[test]
fn add_photos_joins_near_duplicate_groups() {
    use emeans_cluster::dedup::{DedupConfig, DedupInput, dedup};
    use emeans_cluster::synth::{Normal, with_bursts};
    use emeans_core::model::normalize_rows;
    let data = data();
    let d = data.d;
    let n_base = 1500;
    let old = with_bursts(&data.x[..n_base * d], d, 60, (1, 4), 0.1, 3);
    let n_old = old.x.len() / d;
    let mut dd = dedup(
        &old.x,
        d,
        &DedupInput::default(),
        &DedupConfig::default(),
        CPU,
    )
    .unwrap();
    let (mut h, _) = auto_cluster(&dd.gather(&old.x, d), d, &cfg()).unwrap();
    h.dedup = Some(DedupConfig::default());
    let n_reps_old = dd.n_groups();

    let mut rng = Normal::new(11);
    let mut near = |src: &[f32]| -> Vec<f32> {
        let mut v: Vec<f32> = src
            .iter()
            .zip(rng.direction(d))
            .map(|(p, z)| p + 0.05 * z)
            .collect();
        normalize_rows(&mut v, d);
        v
    };
    let mut photos = old.x.clone();
    // (a) 20 near copies of old photos 0..20.
    for i in 0..20 {
        let v = near(&old.x[i * d..(i + 1) * d]);
        photos.extend_from_slice(&v);
    }
    // (b) 10 new photos, each followed by a near copy of itself.
    for i in 0..10 {
        let src = &data.x[(2000 + i) * d..(2001 + i) * d];
        photos.extend_from_slice(src);
        let v = near(src);
        photos.extend_from_slice(&v);
    }
    // (c) 30 fresh photos.
    photos.extend_from_slice(&data.x[2010 * d..2040 * d]);
    let n = photos.len() / d;

    let r = h
        .add_photos(&photos, n_old, &mut dd, &DedupInput::default(), CPU)
        .unwrap();
    assert_eq!(r.joined_existing, 20);
    assert_eq!(r.new_groups, 40);
    assert_eq!(r.leaf.len(), n);
    assert_eq!(dd.group_of.len(), n);
    assert_eq!(dd.n_groups(), n_reps_old + 40);
    assert_eq!(h.n_seen(), dd.n_groups());
    for i in 0..20 {
        let new = n_old + i;
        assert_eq!(
            dd.group_of[new], dd.group_of[i],
            "copy {i} did not join its source"
        );
        assert_eq!(r.leaf[new], r.leaf[i]);
    }
    for p in 0..10 {
        let (a, b) = (n_old + 20 + 2 * p, n_old + 21 + 2 * p);
        assert_eq!(dd.group_of[a], dd.group_of[b], "new pair {p} split");
        assert!(dd.group_of[a] as usize >= n_reps_old);
        assert_eq!(r.leaf[a], r.leaf[b]);
    }
    let fresh: std::collections::HashSet<u32> = (n_old + 40..n).map(|i| dd.group_of[i]).collect();
    assert_eq!(fresh.len(), 30);
    assert!(fresh.iter().all(|&g| g as usize >= n_reps_old));
    // Group sizes add up, and wrong offsets fail loudly.
    assert_eq!(dd.sizes.iter().map(|&s| s as usize).sum::<usize>(), n);
    assert!(
        h.add_photos(&photos, n_old, &mut dd, &DedupInput::default(), CPU)
            .is_err()
    );
}

#[test]
fn add_photos_follows_burst_ids() {
    use emeans_cluster::dedup::{DedupConfig, DedupInput, dedup};
    let data = data();
    let d = data.d;
    let n_old = 800;
    let old = &data.x[..n_old * d];
    let mut burst: Vec<Option<u32>> = vec![None; n_old + 2];
    burst[5] = Some(9);
    let inp_old = DedupInput {
        burst: Some(&burst[..n_old]),
        ..Default::default()
    };
    let mut dd = dedup(old, d, &inp_old, &DedupConfig::default(), CPU).unwrap();
    let (mut h, _) = auto_cluster(&dd.gather(old, d), d, &cfg()).unwrap();
    // Two unrelated new photos; the first shares row 5's burst id.
    burst[n_old] = Some(9);
    let photos = data.x[..(n_old + 2) * d].to_vec();
    let inp = DedupInput {
        burst: Some(&burst),
        ..Default::default()
    };
    let r = h.add_photos(&photos, n_old, &mut dd, &inp, CPU).unwrap();
    assert_eq!(r.joined_existing, 1);
    assert_eq!(dd.group_of[n_old], dd.group_of[5]);
    assert_ne!(dd.group_of[n_old + 1], dd.group_of[5]);
    assert_eq!(r.leaf[n_old], r.leaf[5]);
}

#[test]
fn flat_data_stays_flat() {
    // Six groups with no broad structure above them: one sub group per broad group.
    let data = nested(&NestedSpec {
        n: 2400,
        d: 96,
        n_broad: 6,
        sub_range: (1, 1),
        min_sub: 100,
        noise_frac: 0.0,
        seed: 41,
        ..Default::default()
    });
    let (h, l) = auto_cluster(&data.x, data.d, &cfg()).unwrap();
    let ck = h.chosen_k();
    assert!(h.diagnostics.flat, "grouped flat data: k1 {}", ck.k1);
    assert_eq!(ck.k1, 6);
    assert!(ck.k2.iter().all(|&k| k == 1), "{:?}", ck.k2);
    assert!(broad_ari(&data, &l.l1) > 0.98);
}

#[test]
fn resplit_recovers_leaves_of_merged_groups() {
    // Force the top-down pass to merge broad groups (k1 = 2) and allow at most
    // 3 leaves per group, so leaves hold several true sub groups.
    let data = data();
    let base = AutoConfig {
        k1_range: (2, 2),
        k2_max: 3,
        level1_method: emeans_cluster::auto::Level1Method::TopDown,
        ..cfg()
    };
    let off = AutoConfig {
        resplit_passes: 0,
        ..base.clone()
    };
    let (h0, l0) = auto_cluster(&data.x, data.d, &off).unwrap();
    let (h1, l1) = auto_cluster(&data.x, data.d, &base).unwrap();
    let (a0, a1) = (
        leaf_ari(&data, &l0.leaf_codes()),
        leaf_ari(&data, &l1.leaf_codes()),
    );
    assert_eq!(h0.diagnostics.resplits, 0);
    assert!(h1.diagnostics.resplits > 0, "no leaf was split again");
    assert!(
        a1 > a0 + 0.1 && a1 > 0.85,
        "leaf ARI {a0:.3} without re-split, {a1:.3} with"
    );
}

#[test]
fn level1_skips_cuts_with_one_giant_group() {
    // One broad group (its sub groups are the leaves) plus a small tight group
    // far away: the best silhouette cut is "giant group + small group", which
    // holds no grouping. The guard must reject it.
    let base = nested(&NestedSpec {
        n: 2900,
        d: 96,
        n_broad: 1,
        sub_range: (5, 6),
        min_sub: 60,
        noise_frac: 0.0,
        seed: 7,
        ..Default::default()
    });
    let d = base.d;
    let mut x = base.x.clone();
    let mut state = 12345u64;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    };
    let center: Vec<f32> = (0..d).map(|_| unif()).collect();
    for _ in 0..100 {
        let mut r: Vec<f32> = center.iter().map(|&c| c + 0.02 * unif()).collect();
        let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        r.iter_mut().for_each(|v| *v /= norm);
        x.extend_from_slice(&r);
    }
    let n = x.len() / d;
    let off = AutoConfig {
        level1_max_share: 1.0,
        ..cfg()
    };
    let (h_off, l_off) = auto_cluster(&x, d, &off).unwrap();
    let largest = |l1: &[u32]| {
        let mut s = std::collections::HashMap::new();
        l1.iter().for_each(|&g| *s.entry(g).or_insert(0usize) += 1);
        *s.values().max().unwrap() as f64 / n as f64
    };
    assert!(
        !h_off.diagnostics.flat && largest(&l_off.l1) > 0.9,
        "without the guard the giant-group cut should win (precondition)"
    );
    let (h, l) = auto_cluster(&x, d, &cfg()).unwrap();
    assert!(!h.diagnostics.level1_rejected.is_empty());
    assert!(
        h.diagnostics.flat || largest(&l.l1) <= 0.5,
        "level 1 still has a group with {:.0}% of rows",
        100.0 * largest(&l.l1)
    );
}

/// A unit row with the given (axis, weight) parts.
fn crafted(d: usize, parts: &[(usize, f32)]) -> Vec<f32> {
    let mut v = vec![0f32; d];
    for &(a, w) in parts {
        v[a] = w;
    }
    emeans_core::model::normalize_rows(&mut v, d);
    v
}

/// `add_photos` and a fresh dedup of the same photos apply one rule: a photo
/// joins a group when it is linked to a member and within the threshold of the
/// group's anchor (the star center), whatever the representative is.
#[test]
fn add_photos_agrees_with_a_fresh_dedup() {
    use emeans_cluster::dedup::{DedupConfig, DedupInput, dedup};
    let data = data();
    let d = data.d;
    let n_bg = 800;
    // Background rows spread over the first 80 axes only, so the crafted rows
    // (axes 80..) have cosine 0 with them.
    let mut old: Vec<f32> = Vec::new();
    for i in 0..n_bg {
        let mut r = data.x[i * d..(i + 1) * d].to_vec();
        r[80..].fill(0.0);
        emeans_core::model::normalize_rows(&mut r, d);
        old.extend(r);
    }
    // Anchor A (row 800) and six rim rows at cosine 0.96 to A, 0.9216 to each
    // other: only A links to all. R1 (row 801) is a favorite, so it becomes
    // the representative although A is the star center.
    old.extend(crafted(d, &[(80, 1.0)]));
    for a in 81..=86 {
        old.extend(crafted(d, &[(80, 0.96), (a, 0.28)]));
    }
    let n_old = old.len() / d;
    let (a_row, r1) = (n_bg, n_bg + 1);
    let mut photos = old.clone();
    // N1: on the rim (0.96 to A, 0.9216 to R1).
    photos.extend(crafted(d, &[(80, 0.96), (87, 0.28)]));
    // N2: 0.96 to R1 but only 0.9216 to A.
    let r1v = photos[r1 * d..(r1 + 1) * d].to_vec();
    let mut n2: Vec<f32> = r1v.iter().map(|v| 0.96 * v).collect();
    n2[88] = 0.28;
    photos.extend(n2);
    // N3 + N4: one new burst; N3 is on the rim, N4 is unrelated to A.
    photos.extend(crafted(d, &[(80, 0.96), (89, 0.28)]));
    photos.extend(crafted(d, &[(90, 1.0)]));
    let n = photos.len() / d;
    let (n1, n2, n3, n4) = (n_old, n_old + 1, n_old + 2, n_old + 3);
    let mut priority = vec![0f32; n];
    priority[r1] = 1.0;
    let mut burst: Vec<Option<u32>> = vec![None; n];
    burst[n3] = Some(4);
    burst[n4] = Some(4);
    let cfg = DedupConfig::default();
    let inp_old = DedupInput {
        priority: Some(&priority[..n_old]),
        burst: Some(&burst[..n_old]),
        ..Default::default()
    };
    let mut dd = dedup(&old, d, &inp_old, &cfg, CPU).unwrap();
    let ga = dd.group_of[a_row] as usize;
    assert_eq!(dd.sizes[ga], 7, "precondition: A and its rim are one group");
    assert_eq!(dd.representatives[ga], r1 as u32, "precondition");
    assert_eq!(dd.anchors[ga], a_row as u32, "precondition");
    let (mut h, _) = auto_cluster(&dd.gather(&old, d), d, &cfg_small()).unwrap();
    h.dedup = Some(cfg.clone());

    let inp = DedupInput {
        priority: Some(&priority),
        burst: Some(&burst),
        ..Default::default()
    };
    h.add_photos(&photos, n_old, &mut dd, &inp, CPU).unwrap();
    let fresh = dedup(&photos, d, &inp, &cfg, CPU).unwrap();

    assert_eq!(
        dd.group_of[n1], dd.group_of[a_row],
        "N1 is within the threshold of the anchor"
    );
    assert_ne!(
        dd.group_of[n2], dd.group_of[a_row],
        "N2 is outside the threshold of the anchor"
    );
    assert_eq!(
        dd.group_of[n3], dd.group_of[n4],
        "a new burst stays together"
    );
    assert_ne!(dd.group_of[n4], dd.group_of[a_row]);
    // Same partition as a fresh dedup (both number groups by lowest row).
    assert_eq!(dd.group_of, fresh.group_of);
    assert_eq!(dd.sizes, fresh.sizes);
    assert_eq!(dd.anchors, fresh.anchors);
    assert_eq!(dd.representatives, fresh.representatives);
}

fn cfg_small() -> AutoConfig {
    AutoConfig {
        k1_range: (2, 6),
        k2_max: 4,
        restarts: 1,
        backend: CPU,
        ..Default::default()
    }
}

/// Rows of one tight group around a random direction far from the data of `data()`.
fn new_group(d: usize, rows: usize, seed: u32) -> Vec<f32> {
    let mut rng = emeans_cluster::synth::Normal::new(seed);
    let mut center = rng.direction(d);
    emeans_core::model::normalize_rows(&mut center, d);
    let mut x = Vec::with_capacity(rows * d);
    for _ in 0..rows {
        let mut v: Vec<f32> = center
            .iter()
            .zip(rng.direction(d))
            .map(|(c, z)| c + 0.5 * z)
            .collect();
        emeans_core::model::normalize_rows(&mut v, d);
        x.extend(v);
    }
    x
}

/// The outlier flag is not true by construction: clean training rows are
/// rarely flagged, held-out rows of the same groups are flagged at about the
/// same rate, and rows that belong to no group are all flagged.
#[test]
fn outlier_flag_carries_information() {
    let data = data();
    let d = data.d;
    let n = data.x.len() / d;
    let n0 = n * 8 / 10;
    let (h, l0) = auto_cluster(&data.x[..n0 * d], d, &cfg()).unwrap();
    let rate = |o: &[bool]| o.iter().filter(|&&b| b).count() as f64 / o.len() as f64;
    let train = rate(&l0.outlier);
    assert!(
        train < 0.01,
        "{:.2}% of clean training rows are flagged",
        100.0 * train
    );
    let held = h.assign(&data.x[n0 * d..]).unwrap();
    assert!(
        rate(&held.outlier) < 0.02,
        "{:.2}% of held-out rows of the same groups are flagged",
        100.0 * rate(&held.outlier)
    );
    let far = h.assign(&new_group(d, 50, 77)).unwrap();
    assert!(
        far.outlier.iter().all(|&o| o),
        "rows of an unseen group must be flagged"
    );
}

/// An update forms a new leaf from pending outliers that are one coherent
/// group, instead of absorbing them into old leaves; old ids stay.
#[test]
fn update_grows_a_leaf_for_a_new_group() {
    let data = data();
    let d = data.d;
    let n = data.x.len() / d;
    let n0 = n * 8 / 10;
    let mut c = cfg();
    // Update on 20 pending outliers; keep the full retrain out of this test.
    c.incremental = IncrementalConfig {
        update_min_outliers: 20,
        retrain_outlier_fraction: 0.9,
        ..Default::default()
    };
    let (mut h, l0) = auto_cluster(&data.x[..n0 * d], d, &c).unwrap();
    let old_ids: std::collections::HashSet<(u32, u32)> = (0..n0).map(|i| l0.leaf(i)).collect();
    let leaves0: usize = h.chosen_k().k2.iter().sum();

    // 340 more rows of the known groups, then 60 rows of a new group.
    let n_known = 340;
    let n_new = 60;
    let mut x = data.x[..(n0 + n_known) * d].to_vec();
    x.extend(new_group(d, n_new, 99));
    let first_new = n0 + n_known;
    let r = h.add(&x, n0).unwrap();
    assert_eq!(r.action, Action::EmaUpdate, "precondition");
    assert_eq!(r.new_leaves.len(), 1, "{:?}", r.new_leaves);
    let leaf = r.new_leaves[0];
    assert!(!old_ids.contains(&leaf), "the new leaf reuses an old id");
    let in_leaf = |i: usize| r.labels.leaf(i) == leaf;
    let hits = (first_new..first_new + n_new)
        .filter(|&i| in_leaf(i))
        .count();
    assert!(
        hits >= n_new * 95 / 100,
        "{hits} of {n_new} new rows in the new leaf"
    );
    let strays = (0..first_new).filter(|&i| in_leaf(i)).count();
    assert!(
        strays <= 3,
        "{strays} rows of known groups moved to the new leaf"
    );
    assert_eq!(h.chosen_k().k2.iter().sum::<usize>(), leaves0 + 1);
    let stable = label_stability(&l0, &r.labels.slice(0..n0));
    assert!(stable > 0.97, "label stability {stable}");
    // The grown hierarchy survives save and load.
    let dir = std::env::temp_dir().join(format!("emeans-grow-test-{}", std::process::id()));
    h.save(&dir).unwrap();
    let h2 = Hierarchy::load(&dir, CPU).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(h2.assign(&x).unwrap(), r.labels);

    // Scattered outliers (no two alike) do not form a leaf.
    let mut rng = emeans_cluster::synth::Normal::new(5);
    let mut y = x.clone();
    for _ in 0..30 {
        let mut v = rng.direction(d);
        emeans_core::model::normalize_rows(&mut v, d);
        y.extend(v);
    }
    let r2 = h.add(&y, x.len() / d).unwrap();
    assert_eq!(r2.action, Action::EmaUpdate, "precondition");
    assert!(r2.new_leaves.is_empty(), "{:?}", r2.new_leaves);
}
