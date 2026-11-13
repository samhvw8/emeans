//! The common layer reproduces the facade's results byte for byte: the same
//! check every binding runs, here without a language in between.

use std::fs;
use std::path::{Path, PathBuf};

use emeans_bindings_common as api;
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("emeans-fixture-{}", std::process::id()));
    api::fixture::write_fixture(&dir).unwrap();
    dir
}

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain numeric data, read as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn f32s(dir: &Path, name: &str) -> Vec<f32> {
    fs::read(dir.join(name))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn same(dir: &Path, name: &str, got: &[u8]) {
    let want = fs::read(dir.join(name)).unwrap();
    assert!(want == got, "{name} differs from the facade's result");
}

fn same_labels(dir: &Path, prefix: &str, l: &api::LabelsOutput) {
    same(dir, &format!("{prefix}.l1.u32"), bytes(&l.l1));
    same(dir, &format!("{prefix}.l2.u32"), bytes(&l.l2));
    same(dir, &format!("{prefix}.sim1.f32"), bytes(&l.sim1));
    same(dir, &format!("{prefix}.sim.f32"), bytes(&l.sim));
    same(dir, &format!("{prefix}.outlier.u8"), &l.outlier);
}

#[test]
fn common_layer_equals_the_facade() {
    let dir = fixture_dir();
    let meta: Value = serde_json::from_slice(&fs::read(dir.join("fixture.json")).unwrap()).unwrap();
    let (d, k, n0) = (16usize, 8usize, meta["n0"].as_u64().unwrap() as usize);
    let x = f32s(&dir, "x.f32");

    let t = api::train(&x, d, k, Some(&meta["train"]["config"])).unwrap();
    same(&dir, "train.centroids.f32", bytes(&t.centroids));
    same(
        &dir,
        "train.labels.i32",
        bytes(t.labels.as_deref().unwrap()),
    );
    assert_eq!(t.objective, meta["train"]["objective"].as_f64().unwrap());
    assert_eq!(t.exact_objective, None);
    let mut exact_cfg = meta["train"]["config"].clone();
    exact_cfg["exact_objective"] = true.into();
    let te = api::train(&x, d, k, Some(&exact_cfg)).unwrap();
    same(&dir, "train.centroids.f32", bytes(&te.centroids));
    assert_eq!(
        te.exact_objective,
        meta["train"]["exact_objective"].as_f64()
    );

    let a = api::assign(&x, &t.centroids, d, "inner_product", "cpu").unwrap();
    same(&dir, "assign.labels.u32", bytes(&a.labels));
    same(&dir, "assign.scores.f32", bytes(&a.scores));

    // The model handle: train, assign with the model, then through bytes and a file.
    let (m, t2) = api::train_model(&x, d, k, Some(&meta["train"]["config"])).unwrap();
    same(
        &dir,
        "train.centroids.f32",
        bytes(api::model_centroids(&m).unwrap()),
    );
    assert_eq!(t2.objective, t.objective);
    assert_eq!(
        api::model_objective(&m, &x).unwrap(),
        meta["train"]["exact_objective"].as_f64().unwrap()
    );
    let a = api::model_assign(&m, &x).unwrap();
    same(&dir, "assign.labels.u32", bytes(&a.labels));
    same(&dir, "assign.scores.f32", bytes(&a.scores));
    let m2 = api::model_from_bytes(&api::model_to_bytes(&m).unwrap(), "cpu").unwrap();
    same(
        &dir,
        "assign.scores.f32",
        bytes(&api::model_assign(&m2, &x).unwrap().scores),
    );
    let file = dir.join("model.emeans");
    api::model_save(&m, file.to_str().unwrap()).unwrap();
    let m3 = api::model_load(file.to_str().unwrap(), "cpu").unwrap();
    same(
        &dir,
        "assign.labels.u32",
        bytes(&api::model_assign(&m3, &x).unwrap().labels),
    );
    assert_eq!(api::model_info(&m3)["k"], k);
    assert!(api::model_from_bytes(b"not a model", "cpu").is_err());

    let (mut h, l) = api::auto_cluster(&x[..n0 * d], d, Some(&meta["auto"]["config"])).unwrap();
    same_labels(&dir, "auto", &l);
    // Through save and load, then add: the reloaded model must behave the same.
    let saved = dir.join("hierarchy");
    api::hierarchy_save(&h, saved.to_str().unwrap()).unwrap();
    let mut h2 = api::hierarchy_load(saved.to_str().unwrap(), "cpu").unwrap();
    same_labels(
        &dir,
        "add",
        &api::hierarchy_add(&mut h2, &x).unwrap().labels,
    );
    let blob = api::hierarchy_to_bytes(&h).unwrap();
    let mut h3 = api::hierarchy_from_bytes(&blob, "cpu").unwrap();
    same_labels(
        &dir,
        "add",
        &api::hierarchy_add(&mut h3, &x).unwrap().labels,
    );
    assert!(api::hierarchy_from_bytes(&blob[..blob.len() / 2], "cpu").is_err());
    let r = api::hierarchy_add(&mut h, &x).unwrap();
    same_labels(&dir, "add", &r.labels);
    assert_eq!(r.action, meta["add"]["action"].as_str().unwrap());
    // Append-only: the same rows again are not new rows.
    assert!(api::hierarchy_add(&mut h, &x).is_err());

    let created: Vec<f64> = fs::read(dir.join("created.f64"))
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    let priority = f32s(&dir, "priority.f32");
    let side = api::DedupSide {
        created: Some(&created),
        burst: None,
        priority: Some(&priority),
    };
    let g = api::dedup(&x, d, Some(&meta["dedup"]["config"]), side).unwrap();
    same(&dir, "dedup.group_of.u32", bytes(&g.group_of));
    same(&dir, "dedup.representatives.u32", bytes(&g.representatives));
    same(&dir, "dedup.sizes.u32", bytes(&g.sizes));
    same(&dir, "dedup.nn_sim.f32", bytes(&g.nn_sim));

    fs::remove_dir_all(&dir).unwrap();
}
