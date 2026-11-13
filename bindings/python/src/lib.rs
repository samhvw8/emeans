//! Python binding of emeans (PyO3). The Python package `emeans` re-exports
//! this module (`emeans._emeans`); the stubs are in `python/emeans/_emeans.pyi`.
//!
//! - Inputs go through the buffer protocol without a copy: any C-contiguous
//!   float32 buffer (NumPy, `array.array('f')`, `memoryview`) or raw bytes
//!   (`bytes`, `bytearray`, read as native float32).
//! - Outputs are [`Array`]s: read-only buffers that NumPy, `memoryview` and
//!   `array` read without a copy. `Array.numpy()` wraps one in NumPy.
//! - Compute runs with the GIL released.

use std::ffi::{CStr, c_int, c_void};

use emeans_bindings_common::{self as api, emeans};
use pyo3::buffer::{Element, PyBuffer, PyUntypedBuffer};
use pyo3::exceptions::{PyBufferError, PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use pyo3::{create_exception, ffi};
use serde_json::Value;

create_exception!(
    _emeans,
    EmeansError,
    PyValueError,
    "A call emeans refused or failed: bad shape, config, backend or file."
);

fn err(e: impl std::fmt::Display) -> PyErr {
    EmeansError::new_err(format!("{e:#}"))
}

// ------------------------------------------------------------------ input

/// A float32 view of a Python buffer, read in place.
struct F32In {
    buf: PyUntypedBuffer,
    len: usize,
    /// Columns when the buffer is a 2-D float32 array.
    cols: Option<usize>,
}

impl F32In {
    /// Accept a C-contiguous buffer of native float32, or of raw bytes.
    fn get(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Self> {
        let buf = PyUntypedBuffer::get(obj).map_err(|_| {
            PyTypeError::new_err(format!(
                "{what} must support the buffer protocol (NumPy array, array.array, memoryview, bytes)"
            ))
        })?;
        if !buf.is_c_contiguous() {
            return Err(PyBufferError::new_err(format!(
                "{what} must be C-contiguous"
            )));
        }
        let fmt = native_format(buf.format());
        let is_f32 = fmt == Some("f") && buf.item_size() == 4;
        let is_bytes = matches!(fmt, Some("B" | "b" | "c")) && buf.item_size() == 1;
        if !is_f32 && !is_bytes {
            return Err(PyTypeError::new_err(format!(
                "{what} must hold native float32 (format 'f') or raw bytes, got format {:?} \
                 with {} bytes per item",
                buf.format(),
                buf.item_size()
            )));
        }
        let cols = match (is_f32, buf.shape()) {
            (true, [_, c]) => Some(*c),
            (true, [_]) | (false, _) => None,
            (true, _) => {
                return Err(PyValueError::new_err(format!("{what} must be 1-D or 2-D")));
            }
        };
        let bytes = buf.len_bytes();
        if bytes % 4 != 0 {
            return Err(PyValueError::new_err(format!(
                "{what} has {bytes} bytes, not a whole number of float32"
            )));
        }
        if !(buf.buf_ptr() as usize).is_multiple_of(4) {
            return Err(PyBufferError::new_err(format!(
                "{what} is not 4-byte aligned"
            )));
        }
        Ok(Self {
            len: bytes / 4,
            buf,
            cols,
        })
    }

    fn slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: a C-contiguous, 4-byte aligned buffer of `len` float32, kept
        // exported (so alive and not resized) by `self.buf`. The caller must
        // not write to it during the call; NumPy has the same rule.
        unsafe { std::slice::from_raw_parts(self.buf.buf_ptr() as *const f32, self.len) }
    }

    /// Check that a 2-D buffer has `d` columns.
    fn check(&self, d: usize, what: &str) -> PyResult<()> {
        match self.cols {
            Some(c) if c != d => Err(PyValueError::new_err(format!(
                "d = {d} but {what} has {c} columns"
            ))),
            _ => Ok(()),
        }
    }
}

/// `d` from the argument, else from the first 2-D buffer; every buffer must agree.
fn dim(d: Option<usize>, bufs: &[(&F32In, &str)]) -> PyResult<usize> {
    let d = d
        .or_else(|| bufs.iter().find_map(|(b, _)| b.cols))
        .ok_or_else(|| PyValueError::new_err("the data is 1-D: pass d (the number of columns)"))?;
    for (b, what) in bufs {
        b.check(d, what)?;
    }
    Ok(d)
}

/// The format letter of a native-order buffer.
fn native_format(f: &CStr) -> Option<&str> {
    let s = f.to_str().ok()?;
    Some(match s.as_bytes().first() {
        Some(b'@' | b'=') => &s[1..],
        Some(b'<') if cfg!(target_endian = "little") => &s[1..],
        _ => s,
    })
}

/// An optional per-row side array: copied (it is small next to the data).
fn side<T: Element>(obj: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<Option<Vec<T>>>
where
    for<'a, 'py> Vec<T>: FromPyObject<'a, 'py>,
{
    let Some(o) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    if let Ok(b) = PyBuffer::<T>::get(o) {
        return Ok(Some(b.to_vec(o.py())?));
    }
    o.extract::<Vec<T>>().map(Some).map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be a buffer of {} or a sequence of numbers",
            std::any::type_name::<T>()
        ))
    })
}

/// The address the binding reads `data` at. Tests use it to check that no copy is made.
#[pyfunction]
fn _buffer_address(data: &Bound<'_, PyAny>) -> PyResult<usize> {
    Ok(F32In::get(data, "data")?.slice().as_ptr() as usize)
}

/// A Python config value as JSON.
fn to_json(v: &Bound<'_, PyAny>) -> PyResult<Value> {
    Ok(if v.is_none() {
        Value::Null
    } else if v.is_instance_of::<PyBool>() {
        Value::Bool(v.extract()?)
    } else if v.is_instance_of::<PyInt>() {
        match v.extract::<i64>() {
            Ok(i) => i.into(),
            Err(_) => v.extract::<u64>()?.into(),
        }
    } else if v.is_instance_of::<PyFloat>() {
        let f: f64 = v.extract()?;
        serde_json::Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err(format!("config value {f} is not finite")))?
    } else if let Ok(s) = v.cast::<PyString>() {
        Value::String(s.to_str()?.to_owned())
    } else if let Ok(d) = v.cast::<PyDict>() {
        dict_json(d)?
    } else if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() {
        Value::Array(
            v.try_iter()?
                .map(|i| to_json(&i?))
                .collect::<PyResult<_>>()?,
        )
    } else {
        return Err(PyTypeError::new_err(format!(
            "config values must be None, bool, int, float, str, list or dict, got {}",
            v.get_type().name()?
        )));
    })
}

fn dict_json(d: &Bound<'_, PyDict>) -> PyResult<Value> {
    let mut m = serde_json::Map::new();
    for (k, v) in d.iter() {
        let k: String = k
            .extract()
            .map_err(|_| PyTypeError::new_err("config keys must be strings"))?;
        m.insert(k, to_json(&v)?);
    }
    Ok(Value::Object(m))
}

fn kwargs(c: Option<&Bound<'_, PyDict>>) -> PyResult<Option<Value>> {
    c.map(dict_json).transpose()
}

/// JSON (a default config) as Python values.
fn from_json<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.into_pyobject(py)?.into_any(),
            (_, Some(u)) => u.into_pyobject(py)?.into_any(),
            _ => n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any(),
        },
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(a) => PyList::new(
            py,
            a.iter()
                .map(|x| from_json(py, x))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        Value::Object(o) => {
            let d = PyDict::new(py);
            for (k, x) in o {
                d.set_item(k, from_json(py, x)?)?;
            }
            d.into_any()
        }
    })
}

// ----------------------------------------------------------------- output

enum Data {
    F32(Vec<f32>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U8(Vec<u8>),
}

/// A read-only array of results. It exports the buffer protocol, so
/// `numpy.asarray(a)`, `memoryview(a)` and `array.array` read it without a
/// copy; `a.numpy()` does the first.
#[pyclass(frozen, module = "emeans")]
struct Array {
    data: Data,
    shape: Vec<isize>,
    strides: Vec<isize>,
}

impl Array {
    fn py(py: Python<'_>, data: Data, cols: Option<usize>) -> PyResult<Py<Array>> {
        let (len, item) = match &data {
            Data::F32(v) => (v.len(), 4),
            Data::U32(v) => (v.len(), 4),
            Data::I32(v) => (v.len(), 4),
            Data::U8(v) => (v.len(), 1),
        };
        let (shape, strides) = match cols {
            Some(c) if c > 0 => (
                vec![(len / c) as isize, c as isize],
                vec![(c * item) as isize, item as isize],
            ),
            _ => (vec![len as isize], vec![item as isize]),
        };
        Py::new(
            py,
            Array {
                data,
                shape,
                strides,
            },
        )
    }

    fn raw(&self) -> (*const c_void, usize, &'static CStr, &'static str) {
        match &self.data {
            Data::F32(v) => (v.as_ptr().cast(), v.len() * 4, c"f", "float32"),
            Data::U32(v) => (v.as_ptr().cast(), v.len() * 4, c"I", "uint32"),
            Data::I32(v) => (v.as_ptr().cast(), v.len() * 4, c"i", "int32"),
            Data::U8(v) => (v.as_ptr().cast(), v.len(), c"B", "uint8"),
        }
    }
}

#[pymethods]
impl Array {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        if flags & ffi::PyBUF_WRITABLE == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("emeans arrays are read-only"));
        }
        let a = slf.get();
        let (buf, len, fmt, _) = a.raw();
        let item = a.strides[a.strides.len() - 1];
        let shape = a.shape.as_ptr().cast_mut();
        let strides = a.strides.as_ptr().cast_mut();
        let ndim = a.shape.len() as c_int;
        // SAFETY: `view` is the caller's Py_buffer. Every pointer stored in it
        // points into `a` (frozen: never changed) or to static data, and
        // `view.obj` keeps `a` alive until the buffer is released.
        unsafe {
            (*view).buf = buf.cast_mut();
            (*view).len = len as isize;
            (*view).readonly = 1;
            (*view).itemsize = item;
            (*view).format = if flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT {
                fmt.as_ptr().cast_mut()
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = ndim;
            (*view).shape = if flags & ffi::PyBUF_ND == ffi::PyBUF_ND {
                shape
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
                strides
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            (*view).obj = slf.into_any().into_ptr();
        }
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.shape[0] as usize
    }

    /// `(rows,)` or `(rows, cols)`.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.iter().map(|&s| s as usize).collect()
    }

    /// NumPy name of the element type: `float32`, `uint32`, `int32` or `uint8`.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.raw().3
    }

    /// The values as a list (a list of rows when 2-D).
    fn tolist<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let mv = slf
            .py()
            .import("builtins")?
            .getattr("memoryview")?
            .call1((slf,))?;
        mv.call_method0("tolist")
    }

    /// The array as a read-only NumPy array, without a copy. Needs NumPy.
    fn numpy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let np = slf.py().import("numpy").map_err(|_| {
            PyImportError::new_err("Array.numpy() needs NumPy; memoryview(a) works without it")
        })?;
        np.call_method1("asarray", (slf,))
    }

    fn __repr__(&self) -> String {
        format!(
            "emeans.Array(shape={:?}, dtype={})",
            self.shape(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------- results

/// Statistics of one training epoch.
#[pyclass(frozen, get_all, module = "emeans")]
struct EpochStats {
    obj: f64,
    time: f64,
    imbalance_factor: f64,
    nsplit: usize,
}

#[pymethods]
impl EpochStats {
    fn __repr__(&self) -> String {
        format!(
            "EpochStats(obj={}, time={:.3}, imbalance_factor={:.3}, nsplit={})",
            self.obj, self.time, self.imbalance_factor, self.nsplit
        )
    }
}

/// Result of `train`.
#[pyclass(frozen, get_all, module = "emeans")]
struct TrainResult {
    centroids: Py<Array>,
    objective: f64,
    exact_objective: Option<f64>,
    stats: Vec<Py<EpochStats>>,
    labels: Option<Py<Array>>,
    backend: String,
}

/// Result of `assign`.
#[pyclass(frozen, get_all, module = "emeans")]
struct Assignment {
    labels: Py<Array>,
    scores: Py<Array>,
}

/// Labels of rows under a hierarchy.
#[pyclass(frozen, get_all, module = "emeans")]
struct Labels {
    l1: Py<Array>,
    l2: Py<Array>,
    sim1: Py<Array>,
    sim: Py<Array>,
    outlier: Py<Array>,
}

impl Labels {
    fn py(py: Python<'_>, l: api::LabelsOutput) -> PyResult<Py<Labels>> {
        Py::new(
            py,
            Labels {
                l1: Array::py(py, Data::U32(l.l1), None)?,
                l2: Array::py(py, Data::U32(l.l2), None)?,
                sim1: Array::py(py, Data::F32(l.sim1), None)?,
                sim: Array::py(py, Data::F32(l.sim), None)?,
                outlier: Array::py(py, Data::U8(l.outlier), None)?,
            },
        )
    }
}

/// Result of `Hierarchy.add`.
#[pyclass(frozen, get_all, module = "emeans")]
struct AddReport {
    action: String,
    labels: Py<Labels>,
    new_outliers: usize,
    pending: usize,
    outlier_fraction: f64,
    new_leaves: Vec<(u32, u32)>,
}

/// Result of `dedup`.
#[pyclass(frozen, get_all, module = "emeans")]
struct DedupResult {
    group_of: Py<Array>,
    representatives: Py<Array>,
    anchors: Py<Array>,
    sizes: Py<Array>,
    nn_sim: Py<Array>,
    n_groups: usize,
    linked_pairs: usize,
    split_components: usize,
}

// ------------------------------------------------------------- functions

/// Train e-means with `k` centroids on `data` (`n x d` float32).
#[pyfunction]
#[pyo3(signature = (data, k, *, d=None, **config))]
fn train(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    k: usize,
    d: Option<usize>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<TrainResult> {
    let x = F32In::get(data, "data")?;
    let d = dim(d, &[(&x, "data")])?;
    let cfg = kwargs(config)?;
    let xs = x.slice();
    let t = py
        .detach(|| api::train(xs, d, k, cfg.as_ref()))
        .map_err(err)?;
    let stats = t
        .stats
        .iter()
        .map(|s| {
            Py::new(
                py,
                EpochStats {
                    obj: s.obj,
                    time: s.time,
                    imbalance_factor: s.imbalance_factor,
                    nsplit: s.nsplit,
                },
            )
        })
        .collect::<PyResult<_>>()?;
    Ok(TrainResult {
        centroids: Array::py(py, Data::F32(t.centroids), Some(d))?,
        objective: t.objective,
        exact_objective: t.exact_objective,
        stats,
        labels: t
            .labels
            .map(|l| Array::py(py, Data::I32(l), None))
            .transpose()?,
        backend: t.backend,
    })
}

/// Nearest of `centroids` (`k x d`) for every row of `data` (`n x d`).
#[pyfunction]
#[pyo3(signature = (data, centroids, *, d=None, metric="l2", backend="cpu"))]
fn assign(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    centroids: &Bound<'_, PyAny>,
    d: Option<usize>,
    metric: &str,
    backend: &str,
) -> PyResult<Assignment> {
    let x = F32In::get(data, "data")?;
    let c = F32In::get(centroids, "centroids")?;
    let d = dim(d, &[(&c, "centroids"), (&x, "data")])?;
    let (xs, cs) = (x.slice(), c.slice());
    let a = py
        .detach(|| api::assign(xs, cs, d, metric, backend))
        .map_err(err)?;
    Ok(Assignment {
        labels: Array::py(py, Data::U32(a.labels), None)?,
        scores: Array::py(py, Data::F32(a.scores), None)?,
    })
}

/// Two-level clustering with automatic k of `data` (`n x d`, unit rows).
/// Returns `(Hierarchy, Labels)`.
#[pyfunction]
#[pyo3(signature = (data, *, d=None, **config))]
fn auto_cluster(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    d: Option<usize>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(Hierarchy, Py<Labels>)> {
    let x = F32In::get(data, "data")?;
    let d = dim(d, &[(&x, "data")])?;
    let cfg = kwargs(config)?;
    let xs = x.slice();
    let (h, l) = py
        .detach(|| api::auto_cluster(xs, d, cfg.as_ref()))
        .map_err(err)?;
    Ok((Hierarchy { inner: h }, Labels::py(py, l)?))
}

/// Near-duplicate groups of the unit rows of `data` (`n x d`).
#[pyfunction]
#[pyo3(signature = (data, *, d=None, created=None, burst=None, priority=None, **config))]
fn dedup(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    d: Option<usize>,
    created: Option<&Bound<'_, PyAny>>,
    burst: Option<&Bound<'_, PyAny>>,
    priority: Option<&Bound<'_, PyAny>>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<DedupResult> {
    let x = F32In::get(data, "data")?;
    let d = dim(d, &[(&x, "data")])?;
    let cfg = kwargs(config)?;
    let created = side::<f64>(created, "created")?;
    let burst = side::<i64>(burst, "burst")?;
    let priority = side::<f32>(priority, "priority")?;
    let xs = x.slice();
    let s = api::DedupSide {
        created: created.as_deref(),
        burst: burst.as_deref(),
        priority: priority.as_deref(),
    };
    let g = py
        .detach(|| api::dedup(xs, d, cfg.as_ref(), s))
        .map_err(err)?;
    let u = |v| Array::py(py, Data::U32(v), None);
    Ok(DedupResult {
        n_groups: g.n_groups(),
        linked_pairs: g.linked_pairs,
        split_components: g.split_components,
        group_of: u(g.group_of)?,
        representatives: u(g.representatives)?,
        anchors: u(g.anchors)?,
        sizes: u(g.sizes)?,
        nn_sim: Array::py(py, Data::F32(g.nn_sim), None)?,
    })
}

/// The default config of `kind` (`"train"`, `"auto"` or `"dedup"`) as a dict.
/// Any key of it can be passed as a keyword argument.
#[pyfunction]
fn default_config<'py>(py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyAny>> {
    from_json(py, &api::default_config(kind).map_err(err)?)
}

/// The backends this build can use on this machine (`cpu` first).
#[pyfunction]
fn backends(py: Python<'_>) -> Vec<String> {
    py.detach(api::available_backends)
}

// -------------------------------------------------------------- Hierarchy

/// A two-level clustering that labels new rows and updates itself.
#[pyclass(module = "emeans")]
struct Hierarchy {
    inner: emeans::Hierarchy,
}

#[pymethods]
impl Hierarchy {
    /// Label the rows of `data_all` after the first `n_seen` and update the
    /// model when its policy says so. Pass every row the model has seen
    /// (in order), then the new rows.
    fn add(&mut self, py: Python<'_>, data_all: &Bound<'_, PyAny>) -> PyResult<Py<AddReport>> {
        let x = F32In::get(data_all, "data_all")?;
        x.check(self.inner.d, "data_all")?;
        let xs = x.slice();
        let h = &mut self.inner;
        let r = py.detach(|| api::hierarchy_add(h, xs)).map_err(err)?;
        Py::new(
            py,
            AddReport {
                action: r.action,
                labels: Labels::py(py, r.labels)?,
                new_outliers: r.new_outliers,
                pending: r.pending,
                outlier_fraction: r.outlier_fraction,
                new_leaves: r.new_leaves,
            },
        )
    }

    /// Save to directory `path` (replaced as a whole).
    fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        let p = path.to_string_lossy().into_owned();
        let h = &self.inner;
        py.detach(|| api::hierarchy_save(h, &p)).map_err(err)
    }

    /// Load a hierarchy saved by `save` onto `backend`.
    #[staticmethod]
    #[pyo3(signature = (path, backend="cpu"))]
    fn load(py: Python<'_>, path: std::path::PathBuf, backend: &str) -> PyResult<Hierarchy> {
        let p = path.to_string_lossy().into_owned();
        let h = py
            .detach(|| api::hierarchy_load(&p, backend))
            .map_err(err)?;
        Ok(Hierarchy { inner: h })
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    /// Rows the model has seen (trained on or added).
    #[getter]
    fn n_seen(&self) -> usize {
        self.inner.n_seen()
    }

    /// `(k1, [k2 per level-1 group])`.
    #[getter]
    fn chosen_k(&self) -> (usize, Vec<usize>) {
        let k = self.inner.chosen_k();
        (k.k1, k.k2)
    }

    fn __repr__(&self) -> String {
        let k = self.inner.chosen_k();
        format!(
            "emeans.Hierarchy(d={}, n_seen={}, k1={}, leaves={})",
            self.inner.d,
            self.inner.n_seen(),
            k.k1,
            k.k2.iter().sum::<usize>()
        )
    }
}

#[pymodule]
fn _emeans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("EmeansError", m.py().get_type::<EmeansError>())?;
    m.add_class::<Array>()?;
    m.add_class::<EpochStats>()?;
    m.add_class::<TrainResult>()?;
    m.add_class::<Assignment>()?;
    m.add_class::<Labels>()?;
    m.add_class::<AddReport>()?;
    m.add_class::<DedupResult>()?;
    m.add_class::<Hierarchy>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(assign, m)?)?;
    m.add_function(wrap_pyfunction!(auto_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(dedup, m)?)?;
    m.add_function(wrap_pyfunction!(default_config, m)?)?;
    m.add_function(wrap_pyfunction!(backends, m)?)?;
    m.add_function(wrap_pyfunction!(_buffer_address, m)?)?;
    Ok(())
}
