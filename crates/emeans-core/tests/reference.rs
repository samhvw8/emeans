//! Exact-match tests against values computed by the faiss Python reference
//! (bench/py/fixtures.py writes tests/fixtures/reference.json).

use std::path::PathBuf;

use emeans_core::recipe::{
    BatchLaw, epoch_rows, resolve_batch_and_step, resolve_unused_threshold, scheduled_alpha,
};
use emeans_core::rng::RandomState;
use emeans_core::{Config, Emeans};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn reference() -> Value {
    serde_json::from_slice(&std::fs::read(fixtures().join("reference.json")).unwrap()).unwrap()
}

fn law(s: &str) -> BatchLaw {
    s.parse().unwrap()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn recipes_match_python_exactly() {
    let r = reference();
    let mut checked = 0;
    for c in r["recipes"].as_array().unwrap() {
        let n = c["n"].as_u64().unwrap() as usize;
        let k = c["k"].as_u64().unwrap() as usize;
        let nepochs = c["nepochs"].as_f64().unwrap();
        let d = c["d"].as_u64().unwrap() as usize;
        let (b, a0) = if c["batch_law"] == "given" {
            resolve_batch_and_step(
                n,
                k,
                nepochs,
                BatchLaw::Throughput,
                Some(c["given_batch_size"].as_u64().unwrap() as usize),
                c["given_alpha0"].as_f64(),
            )
        } else {
            let (resolved, _) = law(c["batch_law"].as_str().unwrap()).resolve(d);
            assert_eq!(resolved, law(c["resolved_law"].as_str().unwrap()), "{c}");
            let (b, a0) = resolve_batch_and_step(n, k, nepochs, resolved, None, None);
            let rows = epoch_rows(n, nepochs);
            assert_eq!(rows, usizes(&c["epoch_rows"]), "epoch_rows {c}");
            let steps: Vec<usize> = rows.iter().map(|r| r.div_ceil(b)).collect();
            assert_eq!(steps, usizes(&c["epoch_steps"]), "epoch_steps {c}");
            assert_eq!(
                steps.iter().sum::<usize>(),
                c["total_steps"].as_u64().unwrap() as usize
            );
            assert_eq!(
                n.div_ceil(b),
                c["steps_per_pass"].as_u64().unwrap() as usize
            );
            assert_eq!(
                resolve_unused_threshold(None, d).unwrap(),
                c["unused_threshold"].as_f64().unwrap()
            );
            (b, a0)
        };
        assert_eq!(
            b,
            c["batch_size"].as_u64().unwrap() as usize,
            "batch_size {c}"
        );
        assert_eq!(a0, c["alpha0"].as_f64().unwrap(), "alpha0 bits {c}");
        checked += 1;
    }
    assert!(checked > 1400);
}

#[test]
fn recipe_struct_matches_resolve() {
    let mut m = Emeans::new(Config::new(768, 1024)).unwrap();
    m.begin(200_000, &vec![0.5; 768 * 1024]).unwrap();
    let r = m.recipe().unwrap();
    assert_eq!(r.batch_law, Some(BatchLaw::Throughput));
    assert_eq!(r.epoch_rows, vec![200_000; 5]);
    assert_eq!(r.steps_per_pass, 200_000usize.div_ceil(r.batch_size));
    assert_eq!(r.total_steps, 5 * r.steps_per_pass);
    assert_eq!(r.name, "canonical_v1");
}

#[test]
fn cosine_schedule_matches_python_exactly() {
    for c in reference()["alphas"].as_array().unwrap() {
        let a = scheduled_alpha(
            c["alpha0"].as_f64().unwrap(),
            c["step"].as_u64().unwrap() as usize,
            c["total"].as_u64().unwrap() as usize,
        );
        assert_eq!(a, c["alpha"].as_f64().unwrap(), "{c}");
    }
}

#[test]
fn mt19937_matches_numpy_random_state() {
    for c in reference()["mt19937"].as_array().unwrap() {
        let seed = c["seed"].as_u64().unwrap() as u32;
        if let Some(rs) = c.get("random_sample") {
            let mut r = RandomState::new(seed);
            for v in rs.as_array().unwrap() {
                assert_eq!(r.random_sample(), v.as_f64().unwrap(), "seed {seed}");
            }
            continue;
        }
        let n = c["n"].as_u64().unwrap() as usize;
        let perm = RandomState::new(seed).permutation(n);
        let head: Vec<usize> = perm.iter().take(64).map(|&v| v as usize).collect();
        assert_eq!(head, usizes(&c["head"]), "seed {seed} n {n}");
        let m = n.min(5000);
        let sum: i64 = perm[..m]
            .iter()
            .enumerate()
            .map(|(i, &v)| i as i64 * v as i64)
            .sum();
        assert_eq!(sum, c["sum"].as_i64().unwrap(), "seed {seed} n {n}");
    }
}

/// Train on the fixture data with the same seed: per-epoch objectives, split
/// counts, last-pass labels and final centroids must match faiss NumpyEmeans.
/// The two runs draw the same random numbers; only float summation order differs.
#[test]
fn training_matches_faiss_numpy_emeans() {
    for c in reference()["runs"].as_array().unwrap() {
        let d = c["d"].as_u64().unwrap() as usize;
        let k = c["k"].as_u64().unwrap() as usize;
        let x = emeans_core::io::read_f32_matrix(
            fixtures().join(format!("{}.f32", c["data"].as_str().unwrap())),
            d,
        )
        .unwrap();
        let mut cfg = Config::new(d, k);
        cfg.nepochs = c["nepochs"].as_f64().unwrap();
        cfg.batch_law = law(c["batch_law"].as_str().unwrap());
        cfg.seed = c["seed"].as_u64().unwrap() as u32;
        cfg.spherical = c["spherical"].as_bool().unwrap();
        // The blocked kernel needs no external BLAS, so the test runs anywhere.
        let mut m = Emeans::with_assigner(
            cfg,
            Box::new(emeans_core::CpuAssigner::new(
                emeans_core::CpuKernel::Blocked,
            )),
        )
        .unwrap();
        let t = m.train(&x).unwrap();
        let r = m.recipe().unwrap();
        assert_eq!(r.batch_size, c["batch_size"].as_u64().unwrap() as usize);
        assert_eq!(r.alpha0, c["alpha0"].as_f64().unwrap());
        let st = m.iteration_stats();
        let nsplit: Vec<usize> = st.iter().map(|s| s.nsplit).collect();
        assert_eq!(nsplit, usizes(&c["nsplit"]), "splits {}", c["data"]);
        for (s, o) in st.iter().zip(c["obj"].as_array().unwrap()) {
            let o = o.as_f64().unwrap();
            assert!((s.obj - o).abs() <= 1e-5 * o.abs(), "obj {} vs {o}", s.obj);
        }
        for (s, o) in st.iter().zip(c["imbalance"].as_array().unwrap()) {
            assert!((s.imbalance_factor - o.as_f64().unwrap()).abs() < 1e-9);
        }
        let labels = t.labels.unwrap();
        let want: Vec<i32> = c["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(labels, want, "last-pass labels");
        for (a, b) in t.centroids.iter().zip(c["centroids"].as_array().unwrap()) {
            let b = b.as_f64().unwrap() as f32;
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "centroid {a} vs {b}"
            );
        }
        let asg = m.assign(&x).unwrap();
        let want: Vec<u32> = c["assign_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(asg.labels, want);
        for (a, b) in asg
            .scores
            .iter()
            .zip(c["assign_scores"].as_array().unwrap())
        {
            let b = b.as_f64().unwrap() as f32;
            assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "score {a} vs {b}");
        }
    }
}
