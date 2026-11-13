//! Node.js binding of emeans (napi-rs).
//!
//! - Typed arrays in (`Float32Array`), read in place on a worker thread;
//!   typed arrays out, handed to JS without a copy.
//! - Every compute call is async (a libuv worker), so the event loop stays free.
//! - Configs are plain objects of overrides; unknown keys reject the promise.

use std::sync::{Arc, Mutex, MutexGuard};

use emeans_bindings_common::{self as api, emeans};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

fn js_err(e: anyhow::Error) -> Error {
    Error::new(Status::InvalidArg, format!("{e:#}"))
}

// ---------------------------------------------------------------- results

/// Statistics of one training epoch.
#[napi(object, object_from_js = false)]
pub struct EpochStats {
    pub obj: f64,
    pub time: f64,
    pub imbalance_factor: f64,
    pub nsplit: u32,
}

/// Result of `train`.
#[napi(object, object_from_js = false)]
pub struct TrainResult {
    /// `k x d`, row-major.
    pub centroids: Float32Array,
    pub k: u32,
    pub d: u32,
    /// Free: the last epoch's moving objective as a mean (that epoch's `obj`
    /// divided by the rows it processed). A little above `exactObjective`.
    pub objective: f64,
    /// The mean squared distance (or `2 - 2 cos`, spherical) of all rows to
    /// the final centroids: one extra assign pass, so present only when the
    /// config sets `exact_objective: true`.
    pub exact_objective: Option<f64>,
    pub stats: Vec<EpochStats>,
    /// Last-pass label per row (-1 outside the training subset); absent when
    /// `capture_assignments` is off.
    pub labels: Option<Int32Array>,
    /// The backend that ran.
    pub backend: String,
}

/// Result of `assign`.
#[napi(object, object_from_js = false)]
pub struct Assignment {
    /// Nearest centroid per row.
    pub labels: Uint32Array,
    /// Squared distance (`l2`) or inner product per row.
    pub scores: Float32Array,
}

/// Labels of rows under a hierarchy.
#[napi(object, object_from_js = false)]
pub struct Labels {
    /// Stable level-1 id.
    pub l1: Uint32Array,
    /// Stable level-2 id inside the level-1 group.
    pub l2: Uint32Array,
    /// Cosine to the level-1 centroid.
    pub sim1: Float32Array,
    /// Cosine to the leaf centroid.
    pub sim: Float32Array,
    /// 1 when the row is below its leaf's outlier floor.
    pub outlier: Uint8Array,
}

impl From<api::LabelsOutput> for Labels {
    fn from(l: api::LabelsOutput) -> Self {
        Self {
            l1: Uint32Array::new(l.l1),
            l2: Uint32Array::new(l.l2),
            sim1: Float32Array::new(l.sim1),
            sim: Float32Array::new(l.sim),
            outlier: Uint8Array::new(l.outlier),
        }
    }
}

/// A stable leaf id.
#[napi(object, object_from_js = false)]
pub struct Leaf {
    pub l1: u32,
    pub l2: u32,
}

/// Result of `Hierarchy.add`.
#[napi(object, object_from_js = false)]
pub struct AddReport {
    #[napi(ts_type = "'assign_only' | 'ema_update' | 'full_retrain'")]
    pub action: String,
    /// Labels of all rows after the action (old rows can move).
    pub labels: Labels,
    pub new_outliers: u32,
    pub pending: u32,
    pub outlier_fraction: f64,
    pub new_leaves: Vec<Leaf>,
}

/// Result of `dedup`.
#[napi(object, object_from_js = false)]
pub struct DedupResult {
    /// Group of every row (groups numbered by their lowest row).
    pub group_of: Uint32Array,
    /// Representative row of every group.
    pub representatives: Uint32Array,
    /// Anchor (star center) row of every group.
    pub anchors: Uint32Array,
    /// Members per group.
    pub sizes: Uint32Array,
    /// Cosine of every row to its nearest other row.
    pub nn_sim: Float32Array,
    pub n_groups: u32,
    pub linked_pairs: u32,
    pub split_components: u32,
}

/// Result of `autoCluster`.
#[napi(object, object_from_js = false)]
pub struct AutoResult {
    pub hierarchy: Hierarchy,
    pub labels: Labels,
}

/// The k chosen at each level.
#[napi(object, object_from_js = false)]
pub struct ChosenK {
    pub k1: u32,
    /// Level-2 k per level-1 group (1 = not split; 0 = empty group).
    pub k2: Vec<u32>,
}

/// Options of `assign`.
#[napi(object)]
pub struct AssignOptions {
    #[napi(ts_type = "'l2' | 'inner_product' | 'ip' | 'cosine'")]
    pub metric: Option<String>,
    #[napi(ts_type = "Backend")]
    pub backend: Option<String>,
}

/// Optional per-row side information of `dedup`.
#[napi(object)]
pub struct DedupSide {
    /// Creation time (Unix seconds) per row; NaN = unknown.
    pub created: Option<Float64Array>,
    /// Burst id per row; negative = none. Rows of one burst share a group.
    pub burst: Option<Int32Array>,
    /// Representative preference per row; higher wins.
    pub priority: Option<Float32Array>,
}

// ------------------------------------------------------------------ tasks

pub struct TrainTask {
    data: Float32Array,
    d: usize,
    k: usize,
    config: Option<Value>,
}

impl Task for TrainTask {
    type Output = api::TrainOutput;
    type JsValue = TrainResult;

    fn compute(&mut self) -> Result<Self::Output> {
        api::train(&self.data, self.d, self.k, self.config.as_ref()).map_err(js_err)
    }

    fn resolve(&mut self, _env: Env, t: Self::Output) -> Result<Self::JsValue> {
        Ok(TrainResult {
            k: t.k as u32,
            d: t.d as u32,
            objective: t.objective,
            exact_objective: t.exact_objective,
            stats: t
                .stats
                .iter()
                .map(|s| EpochStats {
                    obj: s.obj,
                    time: s.time,
                    imbalance_factor: s.imbalance_factor,
                    nsplit: s.nsplit as u32,
                })
                .collect(),
            labels: t.labels.map(Int32Array::new),
            backend: t.backend,
            centroids: Float32Array::new(t.centroids),
        })
    }
}

/// Train e-means with `k` centroids on `data` (`n x d`, row-major).
#[napi(
    ts_args_type = "data: Float32Array, d: number, k: number, config?: TrainConfig",
    ts_return_type = "Promise<TrainResult>"
)]
pub fn train(data: Float32Array, d: u32, k: u32, config: Option<Value>) -> AsyncTask<TrainTask> {
    AsyncTask::new(TrainTask {
        data,
        d: d as usize,
        k: k as usize,
        config,
    })
}

pub struct AssignTask {
    data: Float32Array,
    centroids: Float32Array,
    d: usize,
    metric: String,
    backend: String,
}

impl Task for AssignTask {
    type Output = api::AssignOutput;
    type JsValue = Assignment;

    fn compute(&mut self) -> Result<Self::Output> {
        api::assign(
            &self.data,
            &self.centroids,
            self.d,
            &self.metric,
            &self.backend,
        )
        .map_err(js_err)
    }

    fn resolve(&mut self, _env: Env, a: Self::Output) -> Result<Self::JsValue> {
        Ok(Assignment {
            labels: Uint32Array::new(a.labels),
            scores: Float32Array::new(a.scores),
        })
    }
}

/// Nearest of `centroids` (`k x d`) for every row of `data` (`n x d`).
#[napi(ts_return_type = "Promise<Assignment>")]
pub fn assign(
    data: Float32Array,
    centroids: Float32Array,
    d: u32,
    options: Option<AssignOptions>,
) -> AsyncTask<AssignTask> {
    let (metric, backend) = options.map_or((None, None), |o| (o.metric, o.backend));
    AsyncTask::new(AssignTask {
        data,
        centroids,
        d: d as usize,
        metric: metric.unwrap_or_else(|| "l2".into()),
        backend: backend.unwrap_or_else(|| api::DEFAULT_BACKEND.into()),
    })
}

pub struct AutoTask {
    data: Float32Array,
    d: usize,
    config: Option<Value>,
}

impl Task for AutoTask {
    type Output = (emeans::Hierarchy, api::LabelsOutput);
    type JsValue = AutoResult;

    fn compute(&mut self) -> Result<Self::Output> {
        api::auto_cluster(&self.data, self.d, self.config.as_ref()).map_err(js_err)
    }

    fn resolve(&mut self, _env: Env, (h, l): Self::Output) -> Result<Self::JsValue> {
        Ok(AutoResult {
            hierarchy: Hierarchy::wrap(h),
            labels: l.into(),
        })
    }
}

/// Two-level clustering with automatic k of `data` (`n x d`, unit rows).
#[napi(
    ts_args_type = "data: Float32Array, d: number, config?: AutoConfig",
    ts_return_type = "Promise<AutoResult>"
)]
pub fn auto_cluster(data: Float32Array, d: u32, config: Option<Value>) -> AsyncTask<AutoTask> {
    AsyncTask::new(AutoTask {
        data,
        d: d as usize,
        config,
    })
}

pub struct DedupTask {
    data: Float32Array,
    d: usize,
    config: Option<Value>,
    created: Option<Float64Array>,
    burst: Option<Vec<i64>>,
    priority: Option<Float32Array>,
}

impl Task for DedupTask {
    type Output = emeans::DedupResult;
    type JsValue = DedupResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let side = api::DedupSide {
            created: self.created.as_deref(),
            burst: self.burst.as_deref(),
            priority: self.priority.as_deref(),
        };
        api::dedup(&self.data, self.d, self.config.as_ref(), side).map_err(js_err)
    }

    fn resolve(&mut self, _env: Env, g: Self::Output) -> Result<Self::JsValue> {
        Ok(DedupResult {
            n_groups: g.n_groups() as u32,
            linked_pairs: g.linked_pairs as u32,
            split_components: g.split_components as u32,
            group_of: Uint32Array::new(g.group_of),
            representatives: Uint32Array::new(g.representatives),
            anchors: Uint32Array::new(g.anchors),
            sizes: Uint32Array::new(g.sizes),
            nn_sim: Float32Array::new(g.nn_sim),
        })
    }
}

/// Near-duplicate groups of the unit rows of `data` (`n x d`).
#[napi(
    ts_args_type = "data: Float32Array, d: number, config?: DedupConfig, side?: DedupSide",
    ts_return_type = "Promise<DedupResult>"
)]
pub fn dedup(
    data: Float32Array,
    d: u32,
    config: Option<Value>,
    side: Option<DedupSide>,
) -> AsyncTask<DedupTask> {
    let (created, burst, priority) =
        side.map_or((None, None, None), |s| (s.created, s.burst, s.priority));
    AsyncTask::new(DedupTask {
        data,
        d: d as usize,
        config,
        created,
        burst: burst.map(|b| b.iter().map(|&v| i64::from(v)).collect()),
        priority,
    })
}

/// The default config of `kind` (`'train'`, `'auto'` or `'dedup'`). Any key
/// can be passed in the config object.
#[napi(
    ts_args_type = "kind: 'train' | 'auto' | 'dedup'",
    ts_return_type = "Record<string, unknown>"
)]
pub fn default_config(kind: String) -> Result<Value> {
    api::default_config(&kind).map_err(js_err)
}

/// The backends this build can use on this machine (`cpu` first).
#[napi]
pub fn backends() -> Vec<String> {
    api::available_backends()
}

// -------------------------------------------------------------- Hierarchy

/// A two-level clustering that labels new rows and updates itself.
/// Calls on one hierarchy run one at a time.
#[napi]
pub struct Hierarchy {
    inner: Arc<Mutex<emeans::Hierarchy>>,
    d: u32,
}

impl Hierarchy {
    fn wrap(h: emeans::Hierarchy) -> Self {
        Self {
            d: h.d as u32,
            inner: Arc::new(Mutex::new(h)),
        }
    }
}

fn lock(h: &Mutex<emeans::Hierarchy>) -> Result<MutexGuard<'_, emeans::Hierarchy>> {
    h.lock()
        .map_err(|_| Error::from_reason("hierarchy lock poisoned by an earlier failure"))
}

/// The JS thread never waits for a worker: a busy hierarchy is an error.
fn try_lock(h: &Mutex<emeans::Hierarchy>) -> Result<MutexGuard<'_, emeans::Hierarchy>> {
    h.try_lock().map_err(|e| match e {
        std::sync::TryLockError::WouldBlock => Error::new(
            Status::GenericFailure,
            "the hierarchy is busy: await its add() first",
        ),
        std::sync::TryLockError::Poisoned(_) => {
            Error::from_reason("hierarchy lock poisoned by an earlier failure")
        }
    })
}

pub struct AddTask {
    h: Arc<Mutex<emeans::Hierarchy>>,
    data: Float32Array,
}

impl Task for AddTask {
    type Output = api::AddOutput;
    type JsValue = AddReport;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut h = lock(&self.h)?;
        api::hierarchy_add(&mut h, &self.data).map_err(js_err)
    }

    fn resolve(&mut self, _env: Env, r: Self::Output) -> Result<Self::JsValue> {
        Ok(AddReport {
            action: r.action,
            labels: r.labels.into(),
            new_outliers: r.new_outliers as u32,
            pending: r.pending as u32,
            outlier_fraction: r.outlier_fraction,
            new_leaves: r
                .new_leaves
                .iter()
                .map(|&(l1, l2)| Leaf { l1, l2 })
                .collect(),
        })
    }
}

pub struct SaveTask {
    h: Arc<Mutex<emeans::Hierarchy>>,
    path: String,
}

impl Task for SaveTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        let h = lock(&self.h)?;
        api::hierarchy_save(&h, &self.path).map_err(js_err)
    }

    fn resolve(&mut self, _env: Env, _: ()) -> Result<()> {
        Ok(())
    }
}

pub struct LoadTask {
    path: String,
    backend: String,
}

impl Task for LoadTask {
    type Output = emeans::Hierarchy;
    type JsValue = Hierarchy;

    fn compute(&mut self) -> Result<Self::Output> {
        api::hierarchy_load(&self.path, &self.backend).map_err(js_err)
    }

    fn resolve(&mut self, _env: Env, h: Self::Output) -> Result<Self::JsValue> {
        Ok(Hierarchy::wrap(h))
    }
}

#[napi]
impl Hierarchy {
    /// Label the rows of `dataAll` after the first `nSeen` and update the
    /// model when its policy says so. Pass every row the model has seen (in
    /// order), then the new rows.
    #[napi(ts_return_type = "Promise<AddReport>")]
    pub fn add(&self, data_all: Float32Array) -> AsyncTask<AddTask> {
        AsyncTask::new(AddTask {
            h: self.inner.clone(),
            data: data_all,
        })
    }

    /// Save to directory `path` (replaced as a whole).
    #[napi(ts_return_type = "Promise<void>")]
    pub fn save(&self, path: String) -> AsyncTask<SaveTask> {
        AsyncTask::new(SaveTask {
            h: self.inner.clone(),
            path,
        })
    }

    /// Load a hierarchy saved by `save` onto `backend` (default `cpu`).
    #[napi(
        ts_args_type = "path: string, backend?: Backend",
        ts_return_type = "Promise<Hierarchy>"
    )]
    pub fn load(path: String, backend: Option<String>) -> AsyncTask<LoadTask> {
        AsyncTask::new(LoadTask {
            path,
            backend: backend.unwrap_or_else(|| api::DEFAULT_BACKEND.into()),
        })
    }

    /// Columns of the rows.
    #[napi(getter)]
    pub fn d(&self) -> u32 {
        self.d
    }

    /// Rows the model has seen (trained on or added). Throws while an `add` runs.
    #[napi(getter)]
    pub fn n_seen(&self) -> Result<u32> {
        Ok(try_lock(&self.inner)?.n_seen() as u32)
    }

    /// The k chosen at each level. Throws while an `add` runs.
    #[napi(getter)]
    pub fn chosen_k(&self) -> Result<ChosenK> {
        let k = try_lock(&self.inner)?.chosen_k();
        Ok(ChosenK {
            k1: k.k1 as u32,
            k2: k.k2.iter().map(|&v| v as u32).collect(),
        })
    }
}

/// The address the binding reads `data` at (tests check that no copy is made).
#[napi(js_name = "_bufferAddress")]
pub fn buffer_address(data: Float32Array) -> f64 {
    data.as_ptr() as usize as f64
}
