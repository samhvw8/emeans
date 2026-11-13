//! The plain API that the Python, Node.js and Go bindings wrap.
//!
//! Each binding converts its own types to these calls and back, so the rules
//! live here once:
//!
//! - **Configs are JSON objects of overrides.** A call starts from the
//!   library defaults ([`default_config`]) and replaces only the keys given.
//!   An unknown key is an error that names it, so a typo never runs with a
//!   default. Nested objects (`incremental` in the auto config) merge the same way.
//! - **`backend`** is one more key in every config (`"cpu"`, `"metal"`,
//!   `"cuda"`, ...; see [`emeans::Backend::NAMES`]). It is not part of the saved model.
//! - **Shapes:** data is row-major `n x d` `f32`; `n` comes from the length.
//! - Results are plain structs of vectors, ready to hand to a buffer type.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

pub use emeans;

#[doc(hidden)]
pub mod fixture;
use emeans::{
    AddReport, AutoConfig, Backend, Config, DedupConfig, DedupInput, DedupResult, Emeans,
    EpochStats, Hierarchy, Labels, Metric,
};

/// The config kinds [`default_config`] knows.
pub const KINDS: &[&str] = &["train", "auto", "dedup"];

/// Backend used when a config gives none.
pub const DEFAULT_BACKEND: &str = "cpu";

/// The default config of `kind` (`train`, `auto` or `dedup`) as a JSON object,
/// including `"backend": "cpu"`. `train` leaves out `d` and `k` (they are
/// arguments) and adds `"exact_objective": false` (see [`TrainOutput`]).
pub fn default_config(kind: &str) -> Result<Value> {
    let mut v = match kind {
        "train" => {
            let mut v = serde_json::to_value(Config::new(1, 1))?;
            let m = v.as_object_mut().context("train config")?;
            m.remove("d");
            m.remove("k");
            m.insert("exact_objective".into(), json!(false));
            v
        }
        "auto" => serde_json::to_value(AutoConfig::default())?,
        "dedup" => serde_json::to_value(DedupConfig::default())?,
        _ => bail!("unknown config kind {kind:?} (one of {})", KINDS.join(", ")),
    };
    v.as_object_mut()
        .context("config is an object")?
        .insert("backend".into(), json!(DEFAULT_BACKEND));
    Ok(v)
}

/// Replace the keys of `base` by those of `over`, recursing into objects.
/// A key that `base` does not have is an error.
fn merge(base: &mut Map<String, Value>, over: &Map<String, Value>, path: &str) -> Result<()> {
    for (k, v) in over {
        let Some(slot) = base.get_mut(k) else {
            let mut known: Vec<&str> = base.keys().map(String::as_str).collect();
            known.sort_unstable();
            bail!(
                "unknown config key {path}{k:?} (known: {})",
                known.join(", ")
            );
        };
        match (slot, v) {
            (Value::Object(b), Value::Object(o)) => merge(b, o, &format!("{path}{k}."))?,
            (slot, v) => *slot = v.clone(),
        }
    }
    Ok(())
}

/// Defaults of `kind` with `overrides` applied, split into the typed config
/// and the backend.
fn resolve<T: DeserializeOwned>(
    kind: &str,
    overrides: Option<&Value>,
    fill: impl FnOnce(&mut Map<String, Value>) -> Result<()>,
) -> Result<(T, Backend)> {
    let mut base = default_config(kind)?;
    let m = base.as_object_mut().context("config is an object")?;
    match overrides {
        None | Some(Value::Null) => {}
        Some(Value::Object(o)) => {
            for key in ["d", "k"] {
                ensure!(
                    !(kind == "train" && o.contains_key(key)),
                    "pass {key} as an argument, not in the config"
                );
            }
            merge(m, o, "")?
        }
        Some(other) => bail!("config must be an object, got {other}"),
    }
    // After the merge, so the arguments (train's d and k) are not listed as config keys.
    fill(m)?;
    let backend_name = m
        .remove("backend")
        .context("backend")?
        .as_str()
        .context("config key \"backend\" must be a string")?
        .to_owned();
    let backend: Backend = backend_name.parse()?;
    let cfg = serde_json::from_value(base).with_context(|| format!("bad {kind} config"))?;
    Ok((cfg, backend))
}

/// Parse a JSON text of overrides (empty or `null` = defaults).
pub fn parse_overrides(text: &str) -> Result<Option<Value>> {
    let t = text.trim();
    if t.is_empty() {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(t).context("config is not valid JSON")?;
    Ok((!v.is_null()).then_some(v))
}

/// Every backend name the parser accepts.
pub fn backend_names() -> Vec<&'static str> {
    Backend::NAMES.to_vec()
}

/// The backends this build can construct on this machine (`cpu` first).
pub fn available_backends() -> Vec<String> {
    Backend::NAMES
        .iter()
        .filter(|n| n.parse::<Backend>().and_then(emeans::make_assigner).is_ok())
        .map(|n| n.to_string())
        .collect()
}

fn rows(data: &[f32], d: usize, what: &str) -> Result<usize> {
    ensure!(d > 0, "d must be positive");
    ensure!(
        data.len().is_multiple_of(d),
        "{what} has {} values, not a multiple of d = {d}",
        data.len()
    );
    Ok(data.len() / d)
}

// -------------------------------------------------------------------- train

/// Output of [`train`].
#[derive(Clone, Debug)]
pub struct TrainOutput {
    pub k: usize,
    pub d: usize,
    /// `k x d`.
    pub centroids: Vec<f32>,
    /// The last epoch's moving objective, as a mean: that epoch's `obj` (the
    /// sum, over the rows it processed, of the squared distance, or `2 - 2 cos`
    /// when spherical, from each row to its centroid when the row's batch was
    /// assigned) divided by the number of those rows. Free: training computes
    /// it. The centroids still moved after each batch, so it is a little above
    /// the exact objective of the final centroids.
    pub objective: f64,
    /// The exact objective: the same mean over all rows against the final
    /// centroids. Costs one extra assign pass over the data, so it is computed
    /// only when the config sets `"exact_objective": true`; None otherwise.
    pub exact_objective: Option<f64>,
    /// One entry per epoch.
    pub stats: Vec<EpochStats>,
    /// Last-pass label per row; -1 for rows outside the training subset.
    /// None when `capture_assignments` is off.
    pub labels: Option<Vec<i32>>,
    /// Name of the backend that ran (`cpu-accelerate`, `metal-fused`, ...).
    pub backend: String,
}

impl TrainOutput {
    /// `objective` and `stats` as JSON, for bindings that pass metadata as text.
    pub fn meta(&self) -> Value {
        json!({
            "k": self.k, "d": self.d, "objective": self.objective,
            "exact_objective": self.exact_objective,
            "stats": self.stats, "backend": self.backend,
        })
    }
}

/// Train e-means on `data` (`n x d`) with `k` centroids.
pub fn train(data: &[f32], d: usize, k: usize, config: Option<&Value>) -> Result<TrainOutput> {
    Ok(train_model(data, d, k, config)?.1)
}

/// [`train`], and keep the trained model for [`model_assign`], saving and loading.
pub fn train_model(
    data: &[f32],
    d: usize,
    k: usize,
    config: Option<&Value>,
) -> Result<(Emeans, TrainOutput)> {
    let n = rows(data, d, "data")?;
    ensure!(k > 0, "k must be positive");
    ensure!(n >= k, "need at least k = {k} rows, got {n}");
    let mut exact = false;
    let (cfg, backend): (Config, Backend) = resolve("train", config, |m| {
        exact = m
            .remove("exact_objective")
            .and_then(|v| v.as_bool())
            .context("config key \"exact_objective\" must be true or false")?;
        m.insert("d".into(), json!(d));
        m.insert("k".into(), json!(k));
        Ok(())
    })?;
    let mut model = Emeans::with_assigner(cfg, emeans::make_assigner(backend)?)?;
    let trained = model.train(data)?;
    let objective = moving_objective(&model)?;
    let exact_objective = if exact {
        Some(model_objective(&model, data)?)
    } else {
        None
    };
    let out = TrainOutput {
        k,
        d,
        centroids: trained.centroids,
        objective,
        exact_objective,
        stats: model.iteration_stats().to_vec(),
        labels: trained.labels,
        backend: model.assigner_name(),
    };
    Ok((model, out))
}

/// The last epoch's moving objective as a mean over the rows it processed.
fn moving_objective(m: &Emeans) -> Result<f64> {
    let last = m
        .iteration_stats()
        .last()
        .context("training ran no epoch")?;
    let rows = m
        .recipe()
        .and_then(|r| r.epoch_rows.get(m.iteration_stats().len() - 1).copied())
        .filter(|&r| r > 0)
        .context("no row count for the last epoch")?;
    Ok(last.obj / rows as f64)
}

// -------------------------------------------------------------------- model

/// The exact objective of a trained model on `data` (`n x d`): the mean
/// squared distance, or mean `2 - 2 cos` when spherical, of the rows to
/// their nearest final centroid. One assign pass.
pub fn model_objective(m: &Emeans, data: &[f32]) -> Result<f64> {
    rows(data, m.config().d, "data")?;
    Ok(m.assign(data)?.mean_objective(m.config().spherical))
}

/// Nearest centroid of every row of `data` (`n x d`) under a trained model,
/// with the model's metric (inner product when spherical, else squared L2)
/// and backend. Spherical models expect unit rows.
pub fn model_assign(m: &Emeans, data: &[f32]) -> Result<AssignOutput> {
    rows(data, m.config().d, "data")?;
    let a = m.assign(data)?;
    Ok(AssignOutput {
        labels: a.labels,
        scores: a.scores,
    })
}

/// The final centroids of a trained model, `k x d`.
pub fn model_centroids(m: &Emeans) -> Result<&[f32]> {
    m.centroids().context("the model is not trained")
}

/// Facts about a model: `d`, `k`, `metric`, `spherical`, `backend`, `epochs`.
pub fn model_info(m: &Emeans) -> Value {
    let c = m.config();
    let metric = if c.spherical { "inner_product" } else { "l2" };
    json!({
        "d": c.d, "k": c.k, "metric": metric, "spherical": c.spherical,
        "backend": m.assigner_name(), "epochs": m.iteration_stats().len(),
    })
}

/// Save a trained model to file `path` (written to a temporary file, then renamed).
pub fn model_save(m: &Emeans, path: &str) -> Result<()> {
    ensure!(!path.is_empty(), "path is empty");
    m.save(Path::new(path))
}

/// Load a model saved by [`model_save`] onto `backend`.
pub fn model_load(path: &str, backend: &str) -> Result<Emeans> {
    Emeans::load_with(Path::new(path), emeans::make_assigner(backend.parse()?)?)
}

/// The model file's bytes (the same layout as [`model_save`] writes).
pub fn model_to_bytes(m: &Emeans) -> Result<Vec<u8>> {
    m.to_bytes()
}

/// A model from the bytes of [`model_to_bytes`] (or of a model file), on `backend`.
pub fn model_from_bytes(bytes: &[u8], backend: &str) -> Result<Emeans> {
    Emeans::from_bytes_with(bytes, emeans::make_assigner(backend.parse()?)?)
}

// ------------------------------------------------------------------- assign

/// Parse a metric name: `l2` or `inner_product` (alias `ip`, `cosine`).
pub fn parse_metric(name: &str) -> Result<Metric> {
    Ok(match name {
        "l2" => Metric::L2,
        "inner_product" | "ip" | "cosine" => Metric::InnerProduct,
        _ => bail!("unknown metric {name:?} (l2, inner_product)"),
    })
}

/// Output of [`assign`].
#[derive(Clone, Debug)]
pub struct AssignOutput {
    pub labels: Vec<u32>,
    /// Squared L2 distance, or inner product, to the assigned centroid.
    pub scores: Vec<f32>,
}

/// Nearest centroid of every row of `data` (`n x d`) among `centroids` (`k x d`).
pub fn assign(
    data: &[f32],
    centroids: &[f32],
    d: usize,
    metric: &str,
    backend: &str,
) -> Result<AssignOutput> {
    rows(data, d, "data")?;
    let k = rows(centroids, d, "centroids")?;
    ensure!(k > 0, "centroids is empty");
    let metric = parse_metric(metric)?;
    let mut a = emeans::make_assigner(backend.parse()?)?;
    let (labels, scores) = emeans::assign::assign_all(
        a.as_mut(),
        emeans::Rows::Dense(data),
        d,
        centroids,
        metric,
        emeans::recipe::DEFAULT_ASSIGNMENT_CHUNK,
    )?;
    Ok(AssignOutput { labels, scores })
}

// ------------------------------------------------------- auto + incremental

/// Labels of rows under a hierarchy (stable ids).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LabelsOutput {
    pub l1: Vec<u32>,
    pub l2: Vec<u32>,
    pub sim1: Vec<f32>,
    pub sim: Vec<f32>,
    /// 1 when the row is below its leaf's outlier floor.
    pub outlier: Vec<u8>,
}

impl From<Labels> for LabelsOutput {
    fn from(l: Labels) -> Self {
        Self {
            outlier: l.outlier.iter().map(|&o| o as u8).collect(),
            l1: l.l1,
            l2: l.l2,
            sim1: l.sim1,
            sim: l.sim,
        }
    }
}

/// Two-level clustering with automatic k of `data` (`n x d`, unit rows).
pub fn auto_cluster(
    data: &[f32],
    d: usize,
    config: Option<&Value>,
) -> Result<(Hierarchy, LabelsOutput)> {
    rows(data, d, "data")?;
    let (mut cfg, backend): (AutoConfig, Backend) = resolve("auto", config, |_| Ok(()))?;
    cfg.backend = backend;
    let (h, labels) = emeans::auto_cluster(data, d, &cfg)?;
    Ok((h, labels.into()))
}

/// Output of [`hierarchy_add`].
#[derive(Clone, Debug)]
pub struct AddOutput {
    /// `assign_only`, `ema_update` or `full_retrain`.
    pub action: String,
    /// Labels of all rows after the action (old rows can move).
    pub labels: LabelsOutput,
    pub new_outliers: usize,
    pub pending: usize,
    pub outlier_fraction: f64,
    /// Stable `(l1, l2)` ids of leaves formed from pending outliers.
    pub new_leaves: Vec<(u32, u32)>,
}

impl AddOutput {
    pub fn meta(&self) -> Value {
        json!({
            "action": self.action, "new_outliers": self.new_outliers,
            "pending": self.pending, "outlier_fraction": self.outlier_fraction,
            "new_leaves": self.new_leaves,
        })
    }
}

impl From<AddReport> for AddOutput {
    fn from(r: AddReport) -> Self {
        let action = serde_json::to_value(r.action)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        Self {
            action,
            labels: r.labels.into(),
            new_outliers: r.new_outliers,
            pending: r.pending,
            outlier_fraction: r.outlier_fraction,
            new_leaves: r.new_leaves,
        }
    }
}

/// Label the rows of `data_all` (`n x d`) after the first `h.n_seen()` and
/// update the model when its policy says so. Rows are append-only: the first
/// `n_seen` rows must be the ones the model has seen.
pub fn hierarchy_add(h: &mut Hierarchy, data_all: &[f32]) -> Result<AddOutput> {
    let n = rows(data_all, h.d, "data")?;
    let seen = h.n_seen();
    ensure!(
        n > seen,
        "data has {n} rows; pass all {seen} rows the model has seen followed by the new ones"
    );
    Ok(h.add(data_all, seen)?.into())
}

/// Facts about a hierarchy: `d`, `n_seen`, `k1`, `k2` (per level-1 group).
pub fn hierarchy_info(h: &Hierarchy) -> Value {
    let k = h.chosen_k();
    json!({ "d": h.d, "n_seen": h.n_seen(), "k1": k.k1, "k2": k.k2 })
}

/// Save to directory `path` (replaced as a whole).
pub fn hierarchy_save(h: &Hierarchy, path: &str) -> Result<()> {
    ensure!(!path.is_empty(), "path is empty");
    h.save(Path::new(path))
}

/// Load a hierarchy saved by [`hierarchy_save`] onto `backend`.
pub fn hierarchy_load(path: &str, backend: &str) -> Result<Hierarchy> {
    Hierarchy::load(Path::new(path), backend.parse()?)
}

/// The hierarchy as one byte blob (manifest and models), for stores without files.
pub fn hierarchy_to_bytes(h: &Hierarchy) -> Result<Vec<u8>> {
    h.to_bytes()
}

/// A hierarchy from the bytes of [`hierarchy_to_bytes`], on `backend`.
pub fn hierarchy_from_bytes(bytes: &[u8], backend: &str) -> Result<Hierarchy> {
    Hierarchy::from_bytes(bytes, backend.parse()?)
}

// -------------------------------------------------------------------- dedup

/// Optional per-row side information for [`dedup`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DedupSide<'a> {
    /// Creation time (Unix seconds) per row; NaN = unknown.
    pub created: Option<&'a [f64]>,
    /// Burst id per row; negative = none. Rows of one burst share a group.
    pub burst: Option<&'a [i64]>,
    /// Representative preference per row; higher wins.
    pub priority: Option<&'a [f32]>,
}

/// Near-duplicate groups of the unit rows of `data` (`n x d`).
pub fn dedup(
    data: &[f32],
    d: usize,
    config: Option<&Value>,
    side: DedupSide<'_>,
) -> Result<DedupResult> {
    let n = rows(data, d, "data")?;
    for (name, len) in [
        ("created", side.created.map(<[f64]>::len)),
        ("burst", side.burst.map(<[i64]>::len)),
        ("priority", side.priority.map(<[f32]>::len)),
    ] {
        if let Some(len) = len {
            ensure!(len == n, "{name} has {len} values for {n} rows");
        }
    }
    let burst: Option<Vec<Option<u32>>> = side
        .burst
        .map(|b| {
            b.iter()
                .map(|&v| match v {
                    v if v < 0 => Ok(None),
                    v => u32::try_from(v)
                        .map(Some)
                        .map_err(|_| anyhow::anyhow!("burst id {v} is above u32::MAX")),
                })
                .collect()
        })
        .transpose()?;
    let (cfg, backend): (DedupConfig, Backend) = resolve("dedup", config, |_| Ok(()))?;
    let input = DedupInput {
        created_unix: side.created,
        burst: burst.as_deref(),
        priority: side.priority,
    };
    emeans::dedup(data, d, &input, &cfg, backend)
}

/// The scalar facts of a dedup result as JSON.
pub fn dedup_meta(r: &DedupResult) -> Value {
    json!({
        "n_groups": r.n_groups(),
        "linked_pairs": r.linked_pairs,
        "split_components": r.split_components,
    })
}

/// Serialize any value to a JSON string (for bindings that pass text).
pub fn to_json<T: Serialize>(v: &T) -> Result<String> {
    Ok(serde_json::to_string(v)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_keys_fail_with_their_name() {
        let e = resolve::<Config>("train", Some(&json!({"sead": 1})), |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(e.contains("\"sead\""), "{e}");
        let e = resolve::<AutoConfig>("auto", Some(&json!({"incremental": {"x": 1}})), |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(e.contains("incremental.\"x\""), "{e}");
    }

    #[test]
    fn overrides_merge_into_defaults() {
        let (c, b): (AutoConfig, Backend) = resolve(
            "auto",
            Some(&json!({"seed": 9, "k1_range": [2, 5], "incremental": {"grow_min_size": 3}})),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(
            (c.seed, c.k1_range, c.incremental.grow_min_size),
            (9, (2, 5), 3)
        );
        assert_eq!(c.incremental.update_min_outliers, 50);
        assert_eq!(b, Backend::cpu());
    }

    #[test]
    fn misuse_is_an_error() {
        let x = vec![0f32; 10];
        assert!(train(&x, 3, 2, None).is_err()); // 10 values, d = 3
        assert!(train(&x, 5, 2, Some(&json!({"k": 2}))).is_err());
        assert!(train(&x, 5, 2, Some(&json!({"backend": "tpu"}))).is_err());
        assert!(train(&x, 5, 2, Some(&json!([1]))).is_err());
        assert!(assign(&x, &x[..4], 5, "l2", "cpu").is_err());
        assert!(assign(&x, &x[..5], 5, "manhattan", "cpu").is_err());
        let s = DedupSide {
            priority: Some(&[1.0]),
            ..Default::default()
        };
        assert!(dedup(&x, 5, None, s).is_err());
    }

    #[test]
    fn objective_is_free_and_exact_is_opt_in() {
        let x: Vec<f32> = (0..4000).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
        let t = train(&x, 4, 5, None).unwrap();
        assert!(t.exact_objective.is_none());
        let t2 = train(&x, 4, 5, Some(&json!({"exact_objective": true}))).unwrap();
        assert_eq!(t.centroids, t2.centroids); // the opt-in does not change training
        let exact = t2.exact_objective.unwrap();
        // The moving objective is the last epoch's mean: near the exact one, not below by much.
        assert!(
            t.objective > 0.0 && (t.objective - exact).abs() / exact < 0.25,
            "{} {exact}",
            t.objective
        );
        assert!(train(&x, 4, 5, Some(&json!({"exact_objective": 1}))).is_err());
    }

    #[test]
    fn every_default_config_parses_back() {
        for kind in KINDS {
            let v = default_config(kind).unwrap();
            assert_eq!(v["backend"], "cpu");
        }
        assert!(default_config("nope").is_err());
        assert_eq!(available_backends()[0], "cpu");
    }
}
