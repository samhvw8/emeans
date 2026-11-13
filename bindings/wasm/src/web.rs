//! wasm-bindgen exports: the JavaScript face of [`crate::api`].
//!
//! Inputs are typed arrays (copied into wasm memory), options plain objects
//! (read through `JSON.stringify`), results typed arrays and plain objects
//! (copied out). Errors are thrown as `Error`. `Model` and `Hierarchy` hold
//! state in wasm memory: call `free()` when done, or leave it to the garbage
//! collector (wasm-bindgen registers a `FinalizationRegistry`).

use js_sys::{Array, Float32Array, Float64Array, Int32Array, JSON, Object, Reflect, Uint32Array};
use wasm_bindgen::prelude::*;

use crate::api;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);

    // Option types: plain objects, typed in the custom section below.
    #[wasm_bindgen(typescript_type = "TrainOptions")]
    pub type TrainOptions;
    #[wasm_bindgen(typescript_type = "AutoOptions")]
    pub type AutoOptions;
    #[wasm_bindgen(typescript_type = "DedupOptions")]
    pub type DedupOptions;
    #[wasm_bindgen(typescript_type = "DedupInputs")]
    pub type DedupInputs;
    #[wasm_bindgen(typescript_type = "LoadOptions")]
    pub type LoadOptions;
}

/// A Rust panic prints its message to the console before the trap.
#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| error(&format!("emeans panicked: {info}"))));
}

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &str = r#"
/**
 * CPU kernel. `gemm` (default) is the fastest. `fused` is bit-identical to native
 * `cpu-fused` on every machine, and 12-26x slower than `gemm` on wasm (no hardware FMA).
 */
export type Kernel = "fused" | "blocked" | "gemm";

/** Options of `Model.fromBytes` and `Hierarchy.fromBytes`. */
export interface LoadOptions {
  kernel?: Kernel;
}

/** Options of `train` (fields of the Rust `Config`, camelCase). Missing = default. */
export interface TrainOptions {
  kernel?: Kernel;
  /** Epoch budget; fractional values run a partial pass. Default 5. */
  nepochs?: number;
  /** Statistical batch size; null = batch law. */
  batchSize?: number | null;
  /** Initial EMA step; null = step law. */
  alpha0?: number | null;
  batchLaw?: "auto" | "throughput" | "quality";
  /** Default 1234. */
  seed?: number;
  /** Cluster unit vectors by cosine (rows must be L2-normalized). Default false. */
  spherical?: boolean;
  /** Revival threshold tau; null = 0.1 (0 when d <= 2). */
  unusedThreshold?: number | null;
  /** Train on at most this many rows per centroid (random subset). */
  maxPointsPerCentroid?: number | null;
  /** Rows per assignment call; null = 65536. */
  assignmentChunkSize?: number | null;
  /** Keep the last-pass labels (`trainLabels()`). Default true. */
  captureAssignments?: boolean;
}

/** Policy of `Hierarchy.add` (fields of the Rust `IncrementalConfig`). */
export interface IncrementalOptions {
  outlierMadK?: number;
  outlierMinSim?: number | null;
  growMinSize?: number;
  updateMinOutliers?: number;
  updateAddedFraction?: number;
  updateEpochs?: number;
  retrainAddedFraction?: number;
  retrainOutlierFraction?: number;
  matchMinSim?: number;
}

export type Criterion = "silhouette" | "calinski_harabasz" | "davies_bouldin";

/** Options of `autoCluster` (fields of the Rust `AutoConfig`). Rows must be unit vectors. */
export interface AutoOptions {
  kernel?: Kernel;
  /** Level-1 k sweep, inclusive. Default [2, 40]. */
  k1Range?: [number, number];
  k2Max?: number;
  criterion1?: Criterion;
  criterion2?: Criterion;
  level1Tolerance?: number;
  level2Tolerance?: number;
  minClusterSize?: number;
  maxClusterSize?: number;
  minSplitSize?: number;
  splitMinSilhouette?: number;
  restarts?: number;
  nepochs?: number;
  init?: "k_means_plus_plus" | "random_rows";
  level1Method?: "bottom_up" | "top_down";
  resplitPasses?: number;
  level1MaxShare?: number;
  seed?: number;
  incremental?: IncrementalOptions;
}

/** Options of `dedup` (fields of the Rust `DedupConfig`). Rows must be unit vectors. */
export interface DedupOptions {
  kernel?: Kernel;
  /** Neighbours searched per row. Default 10. */
  k?: number;
  /** Cosine at or above which two rows are near-duplicates. Default 0.95. */
  simThreshold?: number;
  repThreshold?: number | null;
  /** Largest time gap (s) for a link when both rows have a time. */
  maxGapSecs?: number | null;
}

/** Per-row side information of `dedup`; each array has one entry per row. */
export interface DedupInputs {
  /** Unix seconds; NaN = unknown. */
  createdUnix?: Float64Array | number[];
  /** Burst id; negative = none. Rows with one id always share a group. */
  burst?: Int32Array | number[];
  /** Representative preference; higher wins. */
  priority?: Float32Array | number[];
}

export interface Assignment {
  labels: Uint32Array;
  /** Squared L2 distance, or cosine (spherical). */
  scores: Float32Array;
}

export interface EpochStats {
  obj: number;
  time: number;
  imbalance_factor: number;
  nsplit: number;
}

/** Labels of rows under a `Hierarchy`. */
export interface Labels {
  /** Stable level-1 id. */
  l1: Uint32Array;
  /** Stable level-2 id inside the level-1 group. */
  l2: Uint32Array;
  /** Cosine to the level-1 centroid. */
  sim1: Float32Array;
  /** Cosine to the leaf centroid. */
  sim: Float32Array;
  /** 1 = below the leaf's outlier floor. */
  outlier: Uint8Array;
  slot1: Uint32Array;
  slot2: Uint32Array;
}

export interface ChosenK {
  k1: number;
  /** Level-2 k per level-1 group (1 = not split, 0 = empty). */
  k2: number[];
}

export interface AddReport {
  action: "assign_only" | "ema_update" | "full_retrain";
  /** Labels of all rows after the action. */
  labels: Labels;
  newOutliers: number;
  pending: number;
  outlierFraction: number;
  /** Stable ids [l1, l2] of leaves formed from pending outliers. */
  newLeaves: [number, number][];
}

export interface DedupResult {
  groupOf: Uint32Array;
  representatives: Uint32Array;
  anchors: Uint32Array;
  sizes: Uint32Array;
  /** Cosine of every row to its nearest other row. */
  nnSim: Float32Array;
  linkedPairs: number;
  splitComponents: number;
}
"#;

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

/// A JavaScript options object as JSON (None for `undefined` / `null`).
fn read_options(v: Option<JsValue>) -> Result<Option<serde_json::Value>, JsError> {
    let Some(v) = v.filter(|v| !v.is_undefined() && !v.is_null()) else {
        return Ok(None);
    };
    let s = JSON::stringify(&v)
        .map_err(|_| JsError::new("options cannot be converted to JSON"))?
        .as_string()
        .ok_or_else(|| JsError::new("options cannot be converted to JSON"))?;
    serde_json::from_str(&s)
        .map(Some)
        .map_err(|e| JsError::new(&format!("options: {e}")))
}

/// A serializable Rust value as a plain JavaScript value.
fn to_js<T: serde::Serialize>(v: &T) -> Result<JsValue, JsError> {
    let s = serde_json::to_string(v).map_err(|e| JsError::new(&e.to_string()))?;
    JSON::parse(&s).map_err(|_| JsError::new("result is not valid JSON"))
}

fn set(o: &Object, key: &str, v: impl Into<JsValue>) {
    // Reflect.set on a plain object cannot fail.
    let _ = Reflect::set(o, &JsValue::from_str(key), &v.into());
}

fn u32s(v: &[u32]) -> Uint32Array {
    Uint32Array::from(v)
}

fn f32s(v: &[f32]) -> Float32Array {
    Float32Array::from(v)
}

fn labels_obj(l: &emeans::Labels) -> Object {
    let o = Object::new();
    set(&o, "l1", u32s(&l.l1));
    set(&o, "l2", u32s(&l.l2));
    set(&o, "sim1", f32s(&l.sim1));
    set(&o, "sim", f32s(&l.sim));
    let outlier: Vec<u8> = l.outlier.iter().map(|&b| b as u8).collect();
    set(&o, "outlier", js_sys::Uint8Array::from(&outlier[..]));
    set(&o, "slot1", u32s(&l.slot1));
    set(&o, "slot2", u32s(&l.slot2));
    o
}

/// Synthetic `n x d` rows around `centers` points with the same bits on every
/// platform (integer draws, exact floats): for demos and benchmarks.
#[wasm_bindgen]
pub fn blobs(n: usize, d: usize, centers: usize, seed: u32) -> Result<Vec<f32>, JsError> {
    api::blobs(n, d, centers, seed).map_err(js_err)
}

/// The crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").into()
}

/// A trained e-means model.
#[wasm_bindgen]
pub struct Model(api::Model);

/// Train e-means with `k` centroids on `x` (`n x d`, row-major).
#[wasm_bindgen]
pub fn train(
    x: &[f32],
    d: usize,
    k: usize,
    options: Option<TrainOptions>,
) -> Result<Model, JsError> {
    let o = read_options(options.map(JsValue::from))?;
    api::train(x, d, k, o.as_ref()).map(Model).map_err(js_err)
}

#[wasm_bindgen]
impl Model {
    /// Read a model written by `toBytes` (or a native `Emeans::save` file).
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8], options: Option<LoadOptions>) -> Result<Model, JsError> {
        let o = read_options(options.map(JsValue::from))?;
        api::Model::from_bytes(bytes, o.as_ref())
            .map(Model)
            .map_err(js_err)
    }

    /// The model file's bytes (same layout as a native model file).
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.0.to_bytes().map_err(js_err)
    }

    #[wasm_bindgen(getter)]
    pub fn d(&self) -> usize {
        self.0.model.config().d
    }

    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize {
        self.0.model.config().k
    }

    /// The CPU kernel's name, for example `cpu-fused`.
    #[wasm_bindgen(getter)]
    pub fn kernel(&self) -> String {
        self.0.kernel.name().into()
    }

    /// Final centroids, `k x d`.
    pub fn centroids(&self) -> Result<Vec<f32>, JsError> {
        self.0.centroids().map_err(js_err)
    }

    /// Last-pass label of every training row (-1 = not sampled); undefined
    /// after `fromBytes` or with `captureAssignments: false`.
    #[wasm_bindgen(js_name = trainLabels)]
    pub fn train_labels(&self) -> Option<Vec<i32>> {
        self.0.labels.clone()
    }

    /// EMA state (positions `k x d`, masses `k`) that `train` continues from.
    #[wasm_bindgen(js_name = emaState, unchecked_return_type = "{ positions: Float32Array, usage: Float32Array } | undefined")]
    pub fn ema_state(&self) -> JsValue {
        match self.0.model.ema_state() {
            Some((p, u)) if !u.is_empty() => {
                let o = Object::new();
                set(&o, "positions", f32s(p));
                set(&o, "usage", f32s(u));
                o.into()
            }
            _ => JsValue::UNDEFINED,
        }
    }

    /// Per-epoch statistics of the last training run.
    #[wasm_bindgen(unchecked_return_type = "EpochStats[]")]
    pub fn stats(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.model.iteration_stats())
    }

    /// Nearest centroid of every row of `x` (`n x d`).
    #[wasm_bindgen(unchecked_return_type = "Assignment")]
    pub fn assign(&self, x: &[f32]) -> Result<JsValue, JsError> {
        let (labels, scores) = self.0.assign(x).map_err(js_err)?;
        let o = Object::new();
        set(&o, "labels", u32s(&labels));
        set(&o, "scores", f32s(&scores));
        Ok(o.into())
    }
}

/// A two-level clustering that labels new rows and updates itself.
#[wasm_bindgen]
pub struct Hierarchy(emeans::Hierarchy);

/// Two-level clustering with automatic k of the unit rows of `x` (`n x d`).
#[wasm_bindgen(js_name = autoCluster, unchecked_return_type = "{ hierarchy: Hierarchy, labels: Labels }")]
pub fn auto_cluster(x: &[f32], d: usize, options: Option<AutoOptions>) -> Result<JsValue, JsError> {
    let o = read_options(options.map(JsValue::from))?;
    let (h, l) = api::auto(x, d, o.as_ref()).map_err(js_err)?;
    let out = Object::new();
    set(&out, "hierarchy", Hierarchy(h));
    set(&out, "labels", labels_obj(&l));
    Ok(out.into())
}

#[wasm_bindgen]
impl Hierarchy {
    /// Read a hierarchy written by `toBytes`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8], options: Option<LoadOptions>) -> Result<Hierarchy, JsError> {
        let o = read_options(options.map(JsValue::from))?;
        api::hierarchy_from_bytes(bytes, o.as_ref())
            .map(Hierarchy)
            .map_err(js_err)
    }

    /// The whole hierarchy (models, ids, policy state) as one byte blob.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.0.to_bytes().map_err(js_err)
    }

    #[wasm_bindgen(getter)]
    pub fn d(&self) -> usize {
        self.0.d
    }

    /// Rows the hierarchy has seen (`add` takes this as `nOld`).
    #[wasm_bindgen(getter, js_name = nSeen)]
    pub fn n_seen(&self) -> usize {
        self.0.n_seen()
    }

    #[wasm_bindgen(js_name = chosenK, unchecked_return_type = "ChosenK")]
    pub fn chosen_k(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.chosen_k())
    }

    /// Sweep scores and decisions (the Rust `Diagnostics`, snake_case keys).
    pub fn diagnostics(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.diagnostics)
    }

    /// Level-1 centroids, `k1 x d`.
    #[wasm_bindgen(js_name = level1Centroids)]
    pub fn level1_centroids(&self) -> Vec<f32> {
        self.0.level1_centroids().to_vec()
    }

    /// Leaf centroids of level-1 group `g`, `k2 x d`.
    #[wasm_bindgen(js_name = leafCentroids)]
    pub fn leaf_centroids(&self, g: usize) -> Result<Vec<f32>, JsError> {
        if g >= self.0.groups.len() {
            return Err(JsError::new(&format!(
                "group {g} out of range ({} groups)",
                self.0.groups.len()
            )));
        }
        Ok(self.0.leaf_centroids(g))
    }

    /// Labels of the unit rows of `x` (the model does not change).
    #[wasm_bindgen(unchecked_return_type = "Labels")]
    pub fn assign(&self, x: &[f32]) -> Result<JsValue, JsError> {
        if self.0.d == 0 || !x.len().is_multiple_of(self.0.d) {
            return Err(JsError::new("x must be n x d"));
        }
        let l = self.0.assign(x).map_err(js_err)?;
        Ok(labels_obj(&l).into())
    }

    /// Label rows `nOld..` of `xAll` (all rows, old first) and update the
    /// model by its policy. `nOld` must equal `nSeen`.
    #[wasm_bindgen(unchecked_return_type = "AddReport")]
    pub fn add(&mut self, x_all: &[f32], n_old: usize) -> Result<JsValue, JsError> {
        let r = api::add(&mut self.0, x_all, n_old).map_err(js_err)?;
        let o = Object::new();
        set(&o, "action", to_js(&r.action)?);
        set(&o, "labels", labels_obj(&r.labels));
        set(&o, "newOutliers", r.new_outliers as f64);
        set(&o, "pending", r.pending as f64);
        set(&o, "outlierFraction", r.outlier_fraction);
        let leaves = Array::new();
        for (a, b) in &r.new_leaves {
            leaves.push(&Array::of2(&(*a).into(), &(*b).into()));
        }
        set(&o, "newLeaves", leaves);
        Ok(o.into())
    }
}

fn side_array<T>(input: &JsValue, key: &str, make: impl Fn(&JsValue) -> T) -> Option<T> {
    let v = Reflect::get(input, &JsValue::from_str(key)).ok()?;
    (!v.is_undefined() && !v.is_null()).then(|| make(&v))
}

/// Near-duplicate groups of the unit rows of `x` (`n x d`).
#[wasm_bindgen(unchecked_return_type = "DedupResult")]
pub fn dedup(
    x: &[f32],
    d: usize,
    options: Option<DedupOptions>,
    inputs: Option<DedupInputs>,
) -> Result<JsValue, JsError> {
    let o = read_options(options.map(JsValue::from))?;
    let side = match inputs
        .map(JsValue::from)
        .filter(|v| !v.is_undefined() && !v.is_null())
    {
        None => api::DedupSide::default(),
        Some(i) => api::DedupSide {
            created_unix: side_array(&i, "createdUnix", |v| Float64Array::new(v).to_vec()),
            burst: side_array(&i, "burst", |v| Int32Array::new(v).to_vec()),
            priority: side_array(&i, "priority", |v| Float32Array::new(v).to_vec()),
        },
    };
    let r = api::dedup(x, d, o.as_ref(), &side).map_err(js_err)?;
    let out = Object::new();
    set(&out, "groupOf", u32s(&r.group_of));
    set(&out, "representatives", u32s(&r.representatives));
    set(&out, "anchors", u32s(&r.anchors));
    set(&out, "sizes", u32s(&r.sizes));
    set(&out, "nnSim", f32s(&r.nn_sim));
    set(&out, "linkedPairs", r.linked_pairs as f64);
    set(&out, "splitComponents", r.split_components as f64);
    Ok(out.into())
}
