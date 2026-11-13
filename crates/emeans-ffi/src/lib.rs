//! The public C API of emeans. `include/emeans.h` is generated from this file
//! by cbindgen (`mise run bind:go:header`); the doc comments here are the
//! header's documentation. The Go binding is its reference user; see
//! `.okf/recipes/adding-a-language-binding.md` for writing another one.
//!
//! The conventions every function follows are in the header's top comment
//! (the `autogen_warning` in `cbindgen.toml`): status codes, thread-local
//! errors, ownership, shapes, configs, thread safety and ABI stability.

#![allow(clippy::missing_safety_doc)] // the header states the contract once

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use anyhow::{Context, Result, bail};
use emeans_bindings_common as api;
use emeans_bindings_common::emeans;
use serde_json::Value;

// ------------------------------------------------------------- constants

/// The ABI version of this header. It changes only with a major version of
/// the library, when a function, struct or constant changes in a way that
/// breaks a compiled caller. Compare it with `emeans_abi_version()` at start-up.
pub const EMEANS_ABI_VERSION: u32 = 1;

/// Status: success.
pub const EMEANS_OK: i32 = 0;
/// Status: an argument was invalid before the library ran: a NULL pointer
/// with a non-zero length, a size that overflows, a misaligned array, text
/// that is not UTF-8, or a config that is not valid JSON.
pub const EMEANS_INVALID_ARGUMENT: i32 = 1;
/// Status: the library refused or failed the call: a bad shape or config
/// value, an unknown config key or backend, a backend this build or machine
/// lacks, an I/O error, corrupt model bytes. `emeans_last_error()` says why.
pub const EMEANS_ERROR: i32 = 2;
/// Status: a bug in emeans (a Rust panic, caught at the boundary). The
/// handles passed to the call should not be used again; free them.
pub const EMEANS_PANIC: i32 = 3;

/// Element type of an `EmeansArray`: `float` (IEEE 754 binary32).
pub const EMEANS_F32: i32 = 0;
/// Element type of an `EmeansArray`: `uint32_t`.
pub const EMEANS_U32: i32 = 1;
/// Element type of an `EmeansArray`: `int32_t`.
pub const EMEANS_I32: i32 = 2;
/// Element type of an `EmeansArray`: `uint8_t`.
pub const EMEANS_U8: i32 = 3;

// ----------------------------------------------------------------- types

/// A borrowed view of one result array. It points into the `EmeansResult`
/// it came from and is valid until that result is freed. Do not write to it
/// or free `data`.
#[repr(C)]
pub struct EmeansArray {
    /// The first element; NULL when `len` is 0.
    pub data: *const c_void,
    /// Number of elements (not bytes).
    pub len: usize,
    /// `EMEANS_F32`, `EMEANS_U32`, `EMEANS_I32` or `EMEANS_U8`.
    pub dtype: i32,
}

enum Data {
    F32(Vec<f32>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U8(Vec<u8>),
}

/// The outputs of one call: named arrays (`emeans_result_array`) and a JSON
/// object of scalars (`emeans_result_json`). Owned by the caller; free it
/// with `emeans_result_free`. Read-only, so several threads may read one.
pub struct EmeansResult {
    arrays: Vec<(&'static str, Data)>,
    json: CString,
}

/// A trained e-means model: centroids plus the state to save and reload
/// it. Owned by the caller; free it with `emeans_model_free`. Its functions
/// only read it, so several threads may use one model at once (assign
/// calls on one model run one at a time inside).
pub struct EmeansModel(emeans::Emeans);

/// A two-level clustering that labels new rows and updates itself. Owned by
/// the caller; free it with `emeans_hierarchy_free`. `emeans_hierarchy_add`
/// changes it: never run it while another thread uses the same hierarchy.
/// The other hierarchy functions only read it.
pub struct EmeansHierarchy(emeans::Hierarchy);

// --------------------------------------------------------------- helpers

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: String) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// An argument error: a different status than a library error.
struct BadArg(String);

impl std::fmt::Display for BadArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::fmt::Debug for BadArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BadArg {}

fn bad(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(BadArg(msg.into()))
}

/// Run `f`, turn its error or panic into a status and the thread's message.
fn guard(f: impl FnOnce() -> Result<()>) -> i32 {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => EMEANS_OK,
        Ok(Err(e)) => {
            let code = if e.downcast_ref::<BadArg>().is_some() {
                EMEANS_INVALID_ARGUMENT
            } else {
                EMEANS_ERROR
            };
            set_error(format!("{e:#}"));
            code
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            set_error(format!("panic in emeans: {msg}"));
            EMEANS_PANIC
        }
    }
}

/// `len` elements at `p`; NULL is allowed only for an empty slice.
unsafe fn slice<'a, T>(p: *const T, len: usize, what: &str) -> Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(bad(format!("{what} is NULL")));
    }
    if !(p as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(bad(format!("{what} is not aligned")));
    }
    // SAFETY: the caller guarantees `len` readable elements at `p` (header contract).
    Ok(unsafe { std::slice::from_raw_parts(p, len) })
}

unsafe fn opt_slice<'a, T>(p: *const T, len: usize, what: &str) -> Result<Option<&'a [T]>> {
    if p.is_null() {
        Ok(None)
    } else {
        unsafe { slice(p, len, what) }.map(Some)
    }
}

fn mul(n: usize, d: usize) -> Result<usize> {
    n.checked_mul(d).ok_or_else(|| bad("n * d overflows"))
}

unsafe fn string<'a>(p: *const c_char, what: &str) -> Result<&'a str> {
    if p.is_null() {
        return Err(bad(format!("{what} is NULL")));
    }
    // SAFETY: the caller passes a NUL-terminated string.
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| bad(format!("{what} is not UTF-8")))
}

/// A backend name, or the default for NULL.
unsafe fn backend<'a>(p: *const c_char) -> Result<&'a str> {
    if p.is_null() {
        Ok(api::DEFAULT_BACKEND)
    } else {
        unsafe { string(p, "backend") }
    }
}

unsafe fn config(p: *const c_char) -> Result<Option<Value>> {
    if p.is_null() {
        return Ok(None);
    }
    let text = unsafe { string(p, "config") }?;
    api::parse_overrides(text).map_err(|e| bad(format!("{e:#}")))
}

/// Write `v` to the caller's out-pointer.
unsafe fn out<T>(p: *mut T, v: T, what: &str) -> Result<()> {
    if p.is_null() {
        return Err(bad(format!("{what} is NULL")));
    }
    // SAFETY: non-null out-pointer from the caller.
    unsafe { p.write(v) };
    Ok(())
}

/// Check every out-pointer before any work, so a failed call allocates nothing.
fn outs(ptrs: &[(*const c_void, &str)]) -> Result<()> {
    match ptrs.iter().find(|(p, _)| p.is_null()) {
        Some((_, what)) => Err(bad(format!("{what} is NULL"))),
        None => Ok(()),
    }
}

fn c_string(s: String) -> Result<CString> {
    CString::new(s).context("string has a NUL byte")
}

fn result(arrays: Vec<(&'static str, Data)>, json: &Value) -> Result<*mut EmeansResult> {
    Ok(Box::into_raw(Box::new(EmeansResult {
        arrays,
        json: c_string(json.to_string())?,
    })))
}

fn labels(l: api::LabelsOutput) -> Vec<(&'static str, Data)> {
    vec![
        ("l1", Data::U32(l.l1)),
        ("l2", Data::U32(l.l2)),
        ("sim1", Data::F32(l.sim1)),
        ("sim", Data::F32(l.sim)),
        ("outlier", Data::U8(l.outlier)),
    ]
}

fn train_result(t: api::TrainOutput) -> Result<*mut EmeansResult> {
    let meta = t.meta();
    let mut arrays = vec![("centroids", Data::F32(t.centroids))];
    if let Some(l) = t.labels {
        arrays.push(("labels", Data::I32(l)));
    }
    result(arrays, &meta)
}

fn assign_result(a: api::AssignOutput) -> Result<*mut EmeansResult> {
    let arrays = vec![
        ("labels", Data::U32(a.labels)),
        ("scores", Data::F32(a.scores)),
    ];
    result(arrays, &Value::Object(Default::default()))
}

fn bytes_result(b: Vec<u8>) -> Result<*mut EmeansResult> {
    let meta = serde_json::json!({ "len": b.len() });
    result(vec![("bytes", Data::U8(b))], &meta)
}

unsafe fn model<'a>(m: *const EmeansModel) -> Result<&'a emeans::Emeans> {
    // SAFETY: a live handle from this library, or NULL.
    Ok(&unsafe { m.as_ref() }.ok_or_else(|| bad("model is NULL"))?.0)
}

unsafe fn hierarchy<'a>(h: *const EmeansHierarchy) -> Result<&'a emeans::Hierarchy> {
    // SAFETY: a live handle from this library, or NULL.
    Ok(&unsafe { h.as_ref() }
        .ok_or_else(|| bad("hierarchy is NULL"))?
        .0)
}

// --------------------------------------------------------------- general

/// The ABI version of the loaded library. A caller compiled against this
/// header should check `emeans_abi_version() == EMEANS_ABI_VERSION`.
#[unsafe(no_mangle)]
pub extern "C" fn emeans_abi_version() -> u32 {
    EMEANS_ABI_VERSION
}

/// The library version, `"major.minor.patch"`. A static string: do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn emeans_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// The message of the calling thread's last failed emeans call, or NULL if
/// its last call succeeded. Every emeans call (except this one) clears it
/// first. Valid until the thread's next emeans call; do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn emeans_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()))
}

/// Free a string that emeans returned through a `char **`. NULL is a no-op.
/// Never pass a string from `emeans_last_error`, `emeans_version` or
/// `emeans_result_json`: those are not owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: `s` came from `CString::into_raw` in this library.
        drop(unsafe { CString::from_raw(s) });
    }
}

/// The default config of `kind` (`"train"`, `"auto"` or `"dedup"`) as a JSON
/// object, including `"backend": "cpu"`. Any of its keys may be overridden in
/// the `config_json` of the matching call. On success `*out_json` is a new
/// string: free it with `emeans_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_default_config(
    kind: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    guard(|| {
        outs(&[(out_json.cast_const().cast(), "out_json")])?;
        let kind = unsafe { string(kind, "kind") }?;
        let v = api::default_config(kind).map_err(|e| bad(format!("{e:#}")))?;
        unsafe { out(out_json, c_string(v.to_string())?.into_raw(), "out_json") }
    })
}

/// The backends this build can use on this machine, as a JSON array of
/// names, `"cpu"` first (for example `["cpu","cpu-fused",...,"metal"]`).
/// Free `*out_json` with `emeans_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_backends(out_json: *mut *mut c_char) -> i32 {
    guard(|| {
        outs(&[(out_json.cast_const().cast(), "out_json")])?;
        let v = serde_json::to_string(&api::available_backends())?;
        unsafe { out(out_json, c_string(v)?.into_raw(), "out_json") }
    })
}

// ------------------------------------------------------------ one-shot calls

/// Train e-means with `k` centroids on `data` (`n` rows of `d` floats,
/// row-major) and keep only the result. For a model to assign with, save or
/// load later, use `emeans_model_train`.
///
/// `config_json`: overrides of `emeans_default_config("train")`, or NULL.
///
/// Result arrays: `centroids` (`EMEANS_F32`, `k * d`), `labels`
/// (`EMEANS_I32`, `n`, the last training pass; -1 for rows outside the
/// training subset; absent when `capture_assignments` is false). Result
/// JSON: `k`, `d`, `objective`, `exact_objective`, `stats` (per epoch: `obj`,
/// `time` in seconds, `imbalance_factor`, `nsplit`), `backend`.
///
/// `objective` is free: the last epoch's moving objective as a mean, that
/// epoch's `obj` (the sum of the squared distance, or `2 - 2 cos` when
/// spherical, from each row it processed to its centroid when the row's
/// batch was assigned) divided by the rows it processed. It is a little above
/// the exact objective. `exact_objective` (the same mean over all rows against
/// the final centroids) costs one extra assign pass: it is `null` unless the
/// config sets `"exact_objective": true`. See also `emeans_model_objective`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_train(
    data: *const f32,
    n: usize,
    d: usize,
    k: usize,
    config_json: *const c_char,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_result.cast_const().cast(), "out_result")])?;
        let x = unsafe { slice(data, mul(n, d)?, "data") }?;
        let cfg = unsafe { config(config_json) }?;
        let t = api::train(x, d, k, cfg.as_ref())?;
        unsafe { out(out_result, train_result(t)?, "out_result") }
    })
}

/// The nearest of `k` centroids (`k * d` floats) for every row of `data`
/// (`n * d` floats). `metric`: `"l2"` (squared distance, smallest wins) or
/// `"inner_product"` (largest wins; cosine for unit rows). `backend`: a
/// backend name, or NULL for `"cpu"`.
///
/// Result arrays: `labels` (`EMEANS_U32`, `n`), `scores` (`EMEANS_F32`, `n`:
/// the squared distance or inner product to the chosen centroid). Ties keep
/// the lowest centroid id.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_assign(
    data: *const f32,
    n: usize,
    d: usize,
    centroids: *const f32,
    k: usize,
    metric: *const c_char,
    backend: *const c_char,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_result.cast_const().cast(), "out_result")])?;
        let x = unsafe { slice(data, mul(n, d)?, "data") }?;
        let c = unsafe { slice(centroids, mul(k, d)?, "centroids") }?;
        let metric = unsafe { string(metric, "metric") }?;
        let a = api::assign(x, c, d, metric, unsafe { self::backend(backend) }?)?;
        unsafe { out(out_result, assign_result(a)?, "out_result") }
    })
}

/// Near-duplicate groups of the unit rows of `data` (`n * d` floats).
/// `config_json`: overrides of `emeans_default_config("dedup")`, or NULL.
/// `created` (Unix seconds, NaN = unknown), `burst` (burst id, negative =
/// none; rows of one burst share a group) and `priority` (higher = preferred
/// representative) are optional per-row arrays of `n` values, or NULL.
///
/// Result arrays: `group_of` (`EMEANS_U32`, `n`; groups are numbered by their
/// lowest row), `representatives`, `anchors`, `sizes` (`EMEANS_U32`, one per
/// group), `nn_sim` (`EMEANS_F32`, `n`: cosine to the nearest other row).
/// Result JSON: `n_groups`, `linked_pairs`, `split_components`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_dedup(
    data: *const f32,
    n: usize,
    d: usize,
    config_json: *const c_char,
    created: *const f64,
    burst: *const i64,
    priority: *const f32,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_result.cast_const().cast(), "out_result")])?;
        let x = unsafe { slice(data, mul(n, d)?, "data") }?;
        let cfg = unsafe { config(config_json) }?;
        let side = api::DedupSide {
            created: unsafe { opt_slice(created, n, "created") }?,
            burst: unsafe { opt_slice(burst, n, "burst") }?,
            priority: unsafe { opt_slice(priority, n, "priority") }?,
        };
        let g = api::dedup(x, d, cfg.as_ref(), side)?;
        let meta = api::dedup_meta(&g);
        let arrays = vec![
            ("group_of", Data::U32(g.group_of)),
            ("representatives", Data::U32(g.representatives)),
            ("anchors", Data::U32(g.anchors)),
            ("sizes", Data::U32(g.sizes)),
            ("nn_sim", Data::F32(g.nn_sim)),
        ];
        unsafe { out(out_result, result(arrays, &meta)?, "out_result") }
    })
}

// ------------------------------------------------------------------ model

/// Train e-means with `k` centroids on `data` (`n * d` floats) and return
/// the trained model in `*out_model` and the training result (the same
/// arrays and JSON as `emeans_train`) in `*out_result`. Pass NULL as
/// `out_result` to skip the result. Free the model with `emeans_model_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_train(
    data: *const f32,
    n: usize,
    d: usize,
    k: usize,
    config_json: *const c_char,
    out_model: *mut *mut EmeansModel,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_model.cast_const().cast(), "out_model")])?;
        let x = unsafe { slice(data, mul(n, d)?, "data") }?;
        let cfg = unsafe { config(config_json) }?;
        let (m, t) = api::train_model(x, d, k, cfg.as_ref())?;
        if !out_result.is_null() {
            unsafe { out(out_result, train_result(t)?, "out_result") }?;
        }
        let m = Box::into_raw(Box::new(EmeansModel(m)));
        unsafe { out(out_model, m, "out_model") }
    })
}

/// The nearest centroid of the model for every row of `data` (`n * d`
/// floats, `d` = the model's). The metric is the model's: inner product
/// (cosine) when it was trained with `spherical`, else squared L2. Result
/// arrays as `emeans_assign`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_assign(
    m: *const EmeansModel,
    data: *const f32,
    n: usize,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_result.cast_const().cast(), "out_result")])?;
        let m = unsafe { model(m) }?;
        let x = unsafe { slice(data, mul(n, m.config().d)?, "data") }?;
        let a = api::model_assign(m, x)?;
        unsafe { out(out_result, assign_result(a)?, "out_result") }
    })
}

/// The exact objective of the model on `data` (`n * d` floats): the mean
/// squared distance, or mean `2 - 2 cos` when spherical, of the rows to their
/// nearest centroid. One assign pass. Writes `*out_objective`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_objective(
    m: *const EmeansModel,
    data: *const f32,
    n: usize,
    out_objective: *mut f64,
) -> i32 {
    guard(|| {
        outs(&[(out_objective.cast_const().cast(), "out_objective")])?;
        let m = unsafe { model(m) }?;
        let x = unsafe { slice(data, mul(n, m.config().d)?, "data") }?;
        let v = api::model_objective(m, x)?;
        unsafe { out(out_objective, v, "out_objective") }
    })
}

/// A borrowed view of the model's centroids (`EMEANS_F32`, `k * d`),
/// valid until the model is freed. Do not write to it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_centroids(
    m: *const EmeansModel,
    out_array: *mut EmeansArray,
) -> i32 {
    guard(|| {
        let c = api::model_centroids(unsafe { model(m) }?)?;
        let a = EmeansArray {
            data: c.as_ptr().cast(),
            len: c.len(),
            dtype: EMEANS_F32,
        };
        unsafe { out(out_array, a, "out_array") }
    })
}

/// Facts about the model as JSON: `d`, `k`, `metric`, `spherical`,
/// `backend`, `epochs`. Free `*out_json` with `emeans_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_info(
    m: *const EmeansModel,
    out_json: *mut *mut c_char,
) -> i32 {
    guard(|| {
        outs(&[(out_json.cast_const().cast(), "out_json")])?;
        let s = api::model_info(unsafe { model(m) }?).to_string();
        unsafe { out(out_json, c_string(s)?.into_raw(), "out_json") }
    })
}

/// Save the model to file `path` (UTF-8). It is written to `path` with the
/// extension `.tmp`, then renamed over `path`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_save(m: *const EmeansModel, path: *const c_char) -> i32 {
    guard(|| api::model_save(unsafe { model(m) }?, unsafe { string(path, "path") }?))
}

/// Load a model saved by `emeans_model_save` onto `backend` (NULL = `"cpu"`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_load(
    path: *const c_char,
    backend: *const c_char,
    out_model: *mut *mut EmeansModel,
) -> i32 {
    guard(|| {
        outs(&[(out_model.cast_const().cast(), "out_model")])?;
        let m = api::model_load(unsafe { string(path, "path") }?, unsafe {
            self::backend(backend)
        }?)?;
        unsafe {
            out(
                out_model,
                Box::into_raw(Box::new(EmeansModel(m))),
                "out_model",
            )
        }
    })
}

/// The model as bytes (the same layout as a model file), for a database or
/// a network. Result array: `bytes` (`EMEANS_U8`). Result JSON: `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_to_bytes(
    m: *const EmeansModel,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_result.cast_const().cast(), "out_result")])?;
        let b = api::model_to_bytes(unsafe { model(m) }?)?;
        unsafe { out(out_result, bytes_result(b)?, "out_result") }
    })
}

/// A model from `len` bytes written by `emeans_model_to_bytes` (or read from
/// a model file), on `backend` (NULL = `"cpu"`). The bytes are copied; the
/// caller may free them after the call. Corrupt or short bytes are
/// `EMEANS_ERROR`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_from_bytes(
    bytes: *const u8,
    len: usize,
    backend: *const c_char,
    out_model: *mut *mut EmeansModel,
) -> i32 {
    guard(|| {
        outs(&[(out_model.cast_const().cast(), "out_model")])?;
        let b = unsafe { slice(bytes, len, "bytes") }?;
        let m = api::model_from_bytes(b, unsafe { self::backend(backend) }?)?;
        unsafe {
            out(
                out_model,
                Box::into_raw(Box::new(EmeansModel(m))),
                "out_model",
            )
        }
    })
}

/// Free a model. NULL is a no-op. Freeing twice is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_model_free(m: *mut EmeansModel) {
    if !m.is_null() {
        // SAFETY: a handle from this library, freed once.
        drop(unsafe { Box::from_raw(m) });
    }
}

// -------------------------------------------------------------- hierarchy

/// Two-level clustering with automatic k of `data` (`n * d` floats, unit
/// rows). `config_json`: overrides of `emeans_default_config("auto")`, or
/// NULL. Returns the hierarchy in `*out_hierarchy` (free it with
/// `emeans_hierarchy_free`) and the labels of the rows in `*out_labels`.
///
/// Label arrays: `l1`, `l2` (`EMEANS_U32`, `n`: stable level-1 id and
/// level-2 id inside the group), `sim1`, `sim` (`EMEANS_F32`, `n`: cosine to
/// the level-1 and the leaf centroid), `outlier` (`EMEANS_U8`, `n`: 1 when
/// below the leaf's outlier floor). Label JSON: the hierarchy info.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_auto_cluster(
    data: *const f32,
    n: usize,
    d: usize,
    config_json: *const c_char,
    out_hierarchy: *mut *mut EmeansHierarchy,
    out_labels: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[
            (out_hierarchy.cast_const().cast(), "out_hierarchy"),
            (out_labels.cast_const().cast(), "out_labels"),
        ])?;
        let x = unsafe { slice(data, mul(n, d)?, "data") }?;
        let cfg = unsafe { config(config_json) }?;
        let (h, l) = api::auto_cluster(x, d, cfg.as_ref())?;
        let r = result(labels(l), &api::hierarchy_info(&h))?;
        unsafe {
            out(out_labels, r, "out_labels")?;
            out(
                out_hierarchy,
                Box::into_raw(Box::new(EmeansHierarchy(h))),
                "out_hierarchy",
            )
        }
    })
}

/// Label new rows and update the model when its policy says so. `data_all`
/// holds `n` rows of the hierarchy's `d` floats: every row the hierarchy has
/// seen (`n_seen` in its info, in the same order), then the new rows.
/// Labels of all `n` rows come back (old rows can move after an update),
/// with the arrays of `emeans_auto_cluster`. Result JSON: `action`
/// (`"assign_only"`, `"ema_update"` or `"full_retrain"`), `new_outliers`,
/// `pending`, `outlier_fraction`, `new_leaves` (`[[l1, l2], ...]`).
/// Changes the hierarchy: no other thread may use it during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_hierarchy_add(
    h: *mut EmeansHierarchy,
    data_all: *const f32,
    n: usize,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_result.cast_const().cast(), "out_result")])?;
        // SAFETY: a live handle, not used by another thread (header contract).
        let h = &mut unsafe { h.as_mut() }
            .ok_or_else(|| bad("hierarchy is NULL"))?
            .0;
        let x = unsafe { slice(data_all, mul(n, h.d)?, "data_all") }?;
        let r = api::hierarchy_add(h, x)?;
        let meta = r.meta();
        unsafe { out(out_result, result(labels(r.labels), &meta)?, "out_result") }
    })
}

/// Facts about the hierarchy as JSON: `d`, `n_seen` (rows trained on or
/// added), `k1`, `k2` (level-2 k per level-1 group; 1 = not split, 0 =
/// empty). Free `*out_json` with `emeans_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_hierarchy_info(
    h: *const EmeansHierarchy,
    out_json: *mut *mut c_char,
) -> i32 {
    guard(|| {
        outs(&[(out_json.cast_const().cast(), "out_json")])?;
        let s = api::hierarchy_info(unsafe { hierarchy(h) }?).to_string();
        unsafe { out(out_json, c_string(s)?.into_raw(), "out_json") }
    })
}

/// Save the hierarchy to directory `path` (UTF-8), replaced as a whole.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_hierarchy_save(
    h: *const EmeansHierarchy,
    path: *const c_char,
) -> i32 {
    guard(|| api::hierarchy_save(unsafe { hierarchy(h) }?, unsafe { string(path, "path") }?))
}

/// Load a hierarchy saved by `emeans_hierarchy_save` onto `backend` (NULL =
/// `"cpu"`). Free it with `emeans_hierarchy_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_hierarchy_load(
    path: *const c_char,
    backend: *const c_char,
    out_hierarchy: *mut *mut EmeansHierarchy,
) -> i32 {
    guard(|| {
        outs(&[(out_hierarchy.cast_const().cast(), "out_hierarchy")])?;
        let path = unsafe { string(path, "path") }?;
        let h = api::hierarchy_load(path, unsafe { self::backend(backend) }?)?;
        let h = Box::into_raw(Box::new(EmeansHierarchy(h)));
        unsafe { out(out_hierarchy, h, "out_hierarchy") }
    })
}

/// The hierarchy as one byte blob (its manifest and models), for a database
/// or a network. Result array: `bytes` (`EMEANS_U8`). Result JSON: `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_hierarchy_to_bytes(
    h: *const EmeansHierarchy,
    out_result: *mut *mut EmeansResult,
) -> i32 {
    guard(|| {
        outs(&[(out_result.cast_const().cast(), "out_result")])?;
        let b = api::hierarchy_to_bytes(unsafe { hierarchy(h) }?)?;
        unsafe { out(out_result, bytes_result(b)?, "out_result") }
    })
}

/// A hierarchy from `len` bytes written by `emeans_hierarchy_to_bytes`, on
/// `backend` (NULL = `"cpu"`). The bytes are copied; the caller may free
/// them after the call. Corrupt or short bytes are `EMEANS_ERROR`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_hierarchy_from_bytes(
    bytes: *const u8,
    len: usize,
    backend: *const c_char,
    out_hierarchy: *mut *mut EmeansHierarchy,
) -> i32 {
    guard(|| {
        outs(&[(out_hierarchy.cast_const().cast(), "out_hierarchy")])?;
        let b = unsafe { slice(bytes, len, "bytes") }?;
        let h = api::hierarchy_from_bytes(b, unsafe { self::backend(backend) }?)?;
        let h = Box::into_raw(Box::new(EmeansHierarchy(h)));
        unsafe { out(out_hierarchy, h, "out_hierarchy") }
    })
}

/// Free a hierarchy. NULL is a no-op. Freeing twice is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_hierarchy_free(h: *mut EmeansHierarchy) {
    if !h.is_null() {
        // SAFETY: a handle from this library, freed once.
        drop(unsafe { Box::from_raw(h) });
    }
}

// ---------------------------------------------------------------- results

/// The array `name` of a result (a borrowed view, valid until the result is
/// freed). `EMEANS_ERROR` when the result has no array of that name; the
/// message lists the names it has.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_result_array(
    r: *const EmeansResult,
    name: *const c_char,
    out_array: *mut EmeansArray,
) -> i32 {
    guard(|| {
        // SAFETY: a live result from this library, or NULL.
        let r = unsafe { r.as_ref() }.ok_or_else(|| bad("result is NULL"))?;
        let name = unsafe { string(name, "name") }?;
        let Some((_, data)) = r.arrays.iter().find(|(n, _)| *n == name) else {
            let names: Vec<&str> = r.arrays.iter().map(|(n, _)| *n).collect();
            bail!(
                "the result has no array {name:?} (has: {})",
                names.join(", ")
            );
        };
        let (data, len, dtype) = match data {
            Data::F32(v) => (v.as_ptr().cast(), v.len(), EMEANS_F32),
            Data::U32(v) => (v.as_ptr().cast(), v.len(), EMEANS_U32),
            Data::I32(v) => (v.as_ptr().cast(), v.len(), EMEANS_I32),
            Data::U8(v) => (v.as_ptr().cast(), v.len(), EMEANS_U8),
        };
        let data = if len == 0 { std::ptr::null() } else { data };
        unsafe { out(out_array, EmeansArray { data, len, dtype }, "out_array") }
    })
}

/// The scalar outputs of a result as a JSON object (`{}` when it has none).
/// Valid until the result is freed; do not free it. NULL for a NULL result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_result_json(r: *const EmeansResult) -> *const c_char {
    // SAFETY: a live result from this library, or NULL.
    unsafe { r.as_ref() }.map_or(std::ptr::null(), |r| r.json.as_ptr())
}

/// Free a result. NULL is a no-op. Freeing twice is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emeans_result_free(r: *mut EmeansResult) {
    if !r.is_null() {
        // SAFETY: a result from this library, freed once.
        drop(unsafe { Box::from_raw(r) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last_error() -> String {
        let p = emeans_last_error();
        assert!(!p.is_null());
        unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned()
    }

    fn data() -> Vec<f32> {
        (0..400).map(|i| ((i * 37) % 101) as f32 / 101.0).collect()
    }

    unsafe fn array(r: *const EmeansResult, name: &CStr) -> EmeansArray {
        let mut a = EmeansArray {
            data: std::ptr::null(),
            len: 0,
            dtype: -1,
        };
        assert_eq!(
            unsafe { emeans_result_array(r, name.as_ptr(), &mut a) },
            EMEANS_OK
        );
        a
    }

    unsafe fn f32s<'a>(a: &EmeansArray) -> &'a [f32] {
        assert_eq!(a.dtype, EMEANS_F32);
        unsafe { std::slice::from_raw_parts(a.data.cast(), a.len) }
    }

    #[test]
    fn errors_have_codes_and_messages() {
        let mut r: *mut EmeansResult = std::ptr::null_mut();
        let x = [0f32; 10];
        let s = unsafe { emeans_train(x.as_ptr(), 2, 5, 2, c"{\"sead\": 1}".as_ptr(), &mut r) };
        assert_eq!(s, EMEANS_ERROR);
        assert!(last_error().contains("sead"), "{}", last_error());
        assert!(r.is_null());
        let s = unsafe { emeans_train(std::ptr::null(), 2, 5, 2, std::ptr::null(), &mut r) };
        assert_eq!(s, EMEANS_INVALID_ARGUMENT);
        assert!(last_error().contains("data is NULL"));
        let s = unsafe { emeans_train(x.as_ptr(), usize::MAX, 5, 2, std::ptr::null(), &mut r) };
        assert_eq!(s, EMEANS_INVALID_ARGUMENT);
        let s = unsafe { emeans_train(x.as_ptr(), 2, 5, 2, c"{".as_ptr(), &mut r) };
        assert_eq!(s, EMEANS_INVALID_ARGUMENT);
        let s =
            unsafe { emeans_train(x.as_ptr(), 2, 5, 2, std::ptr::null(), std::ptr::null_mut()) };
        assert_eq!(s, EMEANS_INVALID_ARGUMENT);
        // A successful call clears the message.
        assert_eq!(emeans_abi_version(), EMEANS_ABI_VERSION);
        let mut j: *mut c_char = std::ptr::null_mut();
        assert_eq!(unsafe { emeans_backends(&mut j) }, EMEANS_OK);
        assert!(emeans_last_error().is_null());
        unsafe { emeans_string_free(j) };
    }

    #[test]
    fn train_then_read_the_result() {
        let x = data();
        let mut r: *mut EmeansResult = std::ptr::null_mut();
        let s = unsafe { emeans_train(x.as_ptr(), 100, 4, 3, c"{\"seed\": 3}".as_ptr(), &mut r) };
        assert_eq!(s, EMEANS_OK);
        let a = unsafe { array(r, c"centroids") };
        assert_eq!((a.len, a.dtype), (12, EMEANS_F32));
        let mut a = a;
        assert_eq!(
            unsafe { emeans_result_array(r, c"nope".as_ptr(), &mut a) },
            EMEANS_ERROR
        );
        assert!(last_error().contains("centroids"));
        let json = unsafe { CStr::from_ptr(emeans_result_json(r)) }
            .to_str()
            .unwrap();
        let v: Value = serde_json::from_str(json).unwrap();
        assert!(v["objective"].as_f64().unwrap() > 0.0);
        unsafe { emeans_result_free(r) };
    }

    /// Model handle: train, assign, bytes and file round trips give the same bits.
    #[test]
    fn model_round_trips() {
        let x = data();
        let (mut m, mut r) = (std::ptr::null_mut(), std::ptr::null_mut());
        let cfg = c"{\"seed\": 3}";
        assert_eq!(
            unsafe { emeans_model_train(x.as_ptr(), 100, 4, 3, cfg.as_ptr(), &mut m, &mut r) },
            EMEANS_OK
        );
        let trained = unsafe { f32s(&array(r, c"centroids")) }.to_vec();
        let mut c = EmeansArray {
            data: std::ptr::null(),
            len: 0,
            dtype: -1,
        };
        assert_eq!(unsafe { emeans_model_centroids(m, &mut c) }, EMEANS_OK);
        assert_eq!(unsafe { f32s(&c) }, &trained[..]);

        let mut a1 = std::ptr::null_mut();
        assert_eq!(
            unsafe { emeans_model_assign(m, x.as_ptr(), 100, &mut a1) },
            EMEANS_OK
        );
        let scores = unsafe { f32s(&array(a1, c"scores")) }.to_vec();
        let mut obj = 0f64;
        assert_eq!(
            unsafe { emeans_model_objective(m, x.as_ptr(), 100, &mut obj) },
            EMEANS_OK
        );
        let mean = scores.iter().map(|&s| s as f64).sum::<f64>() / 100.0; // L2: score = distance
        assert!((obj - mean).abs() < 1e-9 * mean.max(1.0), "{obj} {mean}");

        let mut b = std::ptr::null_mut();
        assert_eq!(unsafe { emeans_model_to_bytes(m, &mut b) }, EMEANS_OK);
        let bytes = unsafe { array(b, c"bytes") };
        let mut m2 = std::ptr::null_mut();
        let s = unsafe {
            emeans_model_from_bytes(bytes.data.cast(), bytes.len, std::ptr::null(), &mut m2)
        };
        assert_eq!(s, EMEANS_OK);
        let mut a2 = std::ptr::null_mut();
        assert_eq!(
            unsafe { emeans_model_assign(m2, x.as_ptr(), 100, &mut a2) },
            EMEANS_OK
        );
        assert_eq!(unsafe { f32s(&array(a2, c"scores")) }, &scores[..]);
        let s = unsafe {
            emeans_model_from_bytes(bytes.data.cast(), bytes.len / 2, std::ptr::null(), &mut m2)
        };
        assert_eq!(s, EMEANS_ERROR);

        let dir = std::env::temp_dir().join(format!("emeans-ffi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = CString::new(dir.join("m.emeans").to_str().unwrap()).unwrap();
        assert_eq!(unsafe { emeans_model_save(m, path.as_ptr()) }, EMEANS_OK);
        let mut m3 = std::ptr::null_mut();
        assert_eq!(
            unsafe { emeans_model_load(path.as_ptr(), c"cpu".as_ptr(), &mut m3) },
            EMEANS_OK
        );
        let mut c3 = EmeansArray {
            data: std::ptr::null(),
            len: 0,
            dtype: -1,
        };
        assert_eq!(unsafe { emeans_model_centroids(m3, &mut c3) }, EMEANS_OK);
        assert_eq!(unsafe { f32s(&c3) }, &trained[..]);
        std::fs::remove_dir_all(&dir).unwrap();

        // Rows of the wrong width for the model.
        let mut bad = std::ptr::null_mut();
        assert_eq!(
            unsafe { emeans_model_assign(m, std::ptr::null(), 1, &mut bad) },
            EMEANS_INVALID_ARGUMENT
        );
        unsafe {
            for r in [r, a1, a2, b] {
                emeans_result_free(r);
            }
            for m in [m, m2, m3] {
                emeans_model_free(m);
            }
        }
    }
}
