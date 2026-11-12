//! e-means CUDA backend: [`CudaCore`], the layer-1 engine of `emeans-core`, on
//! `cudarc` (driver API and NVRTC; no candle).
//!
//! The crate builds without a CUDA toolkit: cudarc loads `libcuda` and
//! `libnvrtc` at run time, and [`CudaCore::new`] returns an error when no
//! driver or device is present.
//!
//! Kernels are CUDA C compiled at runtime with NVRTC (`cuda_src`), the same
//! pattern as the Metal backend's runtime-compiled shaders. Everything runs on
//! one stream; ops queue without a host wait, and `read_*` waits.
//!
//! - `top1`: fused FP32 score + argmax (faiss's Triton `_assign_kernel` idea in
//!   CUDA C: register tiles, running best per row, no `rows x k` matrix).
//! - Exact ops (scatter-add, EMA, norms) compile with `--fmad=false`, so they
//!   round like the CPU code.

use std::ops::Range;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use emeans_core::assign::{Assigner, Metric, Rows};
use emeans_core::core::{Device, EmaStep, Mat, Select, check_select};

mod cuda_src;
use cuda_src::{OPS_SOURCE, TOP1_BF16_SOURCE, TOP1_SOURCE};

/// A device buffer. `Arc` so the bound data set is shared without a copy;
/// writes need sole ownership.
pub struct CBuf<T> {
    s: Arc<CudaSlice<T>>,
    len: usize,
}

impl<T> Clone for CBuf<T> {
    fn clone(&self) -> Self {
        CBuf {
            s: self.s.clone(),
            len: self.len,
        }
    }
}

impl<T> CBuf<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Top1P {
    rows: u32,
    k: u32,
    d: u32,
    ip: u32,
    indexed: u32,
    r0: u32,
    xr0: u32,
    chunk: u32,
    parts: u32,
    list: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScatterP {
    lo: u32,
    m: u32,
    k: u32,
    d: u32,
    indexed: u32,
    acc: u32,
    segs: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmaP {
    k: u32,
    d: u32,
    alpha: f32,
    decay: f32,
    replace: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Bf16P {
    rows: u32,
    k: u32,
    d: u32,
    ip: u32,
    r0: u32,
    xr0: u32,
    indexed: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Bf16P2 {
    rows: u32,
    k: u32,
    d: u32,
    ip: u32,
    r0: u32,
    xr0: u32,
    indexed: u32,
    chunk: u32,
    parts: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScreenP {
    rows: u32,
    k: u32,
    d: u32,
    ip: u32,
    r0: u32,
    xr0: u32,
    indexed: u32,
    parts: u32,
    dense_norm: u32,
    eps: f32,
}

/// A pipelined BF16 kernel: (centroid tile, stages), function, resident
/// blocks per SM, dynamic shared memory in bytes.
type Bf16Kernel = ((usize, usize), CudaFunction, u32, usize);

/// BF16 kernels: the converter, the first kernel (`v1`), the pipelined
/// kernels and the one `EMEANS_CUDA_BF16` picks, and the exact screen
/// (`EMEANS_CUDA_SCREEN`).
#[derive(Clone)]
struct Bf16Kernels {
    conv: CudaFunction,
    v1: CudaFunction,
    v2: Vec<Bf16Kernel>,
    combine: CudaFunction,
    /// None: v1; Some((tn, stages)): v2, tn 0 = by k (see `auto_tile`).
    pick: Option<(usize, usize)>,
    screen: Vec<Bf16Kernel>,
    verify: CudaFunction,
    max_aux: CudaFunction,
    conv16: CudaFunction,
    /// The screen kernel the FP32 path uses (tn 0 = by k), None: off.
    screen_pick: Option<(usize, usize)>,
}

/// Centroid tile of the pipelined kernels when not forced: 256 once k fills
/// it (measured: 0.87 vs 1.09 ms per 16k x 1024 batch), else 128.
fn auto_tile((tn, st): (usize, usize), k: usize) -> (usize, usize) {
    match tn {
        0 if k >= 512 => (256, st),
        0 => (128, st),
        _ => (tn, st),
    }
}

/// Smallest k at which the FP32 path screens (below it the FP32 kernel alone
/// is cheap).
const SCREEN_MIN_K: usize = 128;

/// Relative bound on |screen dot - FP32 dot| per unit of |x| |c| for rows of
/// length d, screening in FP16 on power-of-two-scaled operands: operand
/// rounding (2u + u^2, u = 2^-11; elements below FP16's normal range are
/// zeroed and bounded apart, by their mass), tensor-core accumulation (taken
/// as d 2^-21, four times the truncation bound), the FP32 kernel's own fmaf
/// chain (gamma_d with 2^-24), times 1.05.
fn screen_eps(d: usize) -> f32 {
    let u = 2f64.powi(-11);
    let g = d as f64 * 2f64.powi(-24);
    (((2.0 * u + u * u) + d as f64 * 2f64.powi(-21) * 1.01 + g / (1.0 - g)) * 1.05) as f32
}

/// The BF16 copy of a data set `top1` reads more than once (the training data):
/// converted on the second call with the same buffer, then rows are read by id.
/// `src` is weak so the cache never keeps the data alive; it also keeps the
/// address from being reused while the entry exists.
struct Bf16Data {
    src: std::sync::Weak<CudaSlice<f32>>,
    len: usize,
    /// FP16 screen copy (aux rows) or BF16 copy (squared norms).
    f16: bool,
    rows: Option<(CBuf<u16>, CBuf<f32>)>,
}

/// BF16 operands of one dispatch (see `bf16_prepare`). `dense`: the rows were
/// converted for this call, so row i of `xb`/`xn` is position i.
struct Bf16Prep {
    xb: CBuf<u16>,
    xn: CBuf<f32>,
    indexed: u32,
    xr0: u32,
    dense: bool,
    cb: CBuf<u16>,
    cn: CBuf<f32>,
}

// SAFETY: repr(C) structs of u32/f32, the layout the kernels declare.
unsafe impl DeviceRepr for Top1P {}
unsafe impl DeviceRepr for Bf16P {}
unsafe impl DeviceRepr for Bf16P2 {}
unsafe impl DeviceRepr for ScreenP {}
unsafe impl DeviceRepr for ScatterP {}
unsafe impl DeviceRepr for EmaP {}

use emeans_core::knn::GPU_KMAX as TOPK_KMAX;

/// The CUDA engine: context, stream, kernels.
pub struct CudaCore {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _top1_mod: Arc<CudaModule>,
    _ops_mod: Arc<CudaModule>,
    /// (BM, BN, vec) -> (kernel, resident blocks per SM).
    top1: Vec<((usize, usize, bool), CudaFunction, usize)>,
    combine: CudaFunction,
    /// Partial bests when the centroids are split across blocks.
    pv: Option<CBuf<f32>>,
    pj: Option<CBuf<u32>>,
    pxn: Option<CBuf<f32>>,
    /// Forced centroid parts (`EMEANS_CUDA_PARTS`).
    parts: Option<usize>,
    gather: CudaFunction,
    row_norms_f: CudaFunction,
    normalize_f: CudaFunction,
    count: CudaFunction,
    scan: CudaFunction,
    place: CudaFunction,
    sum: CudaFunction,
    ema_pos: CudaFunction,
    ema_usage: CudaFunction,
    split_f: CudaFunction,
    topk_f: CudaFunction,
    sms: usize,
    /// Forced (BM, BN) from `EMEANS_CUDA_TILE=BM,BN`.
    tile: Option<(usize, usize)>,
    /// Gather indexed rows into a contiguous block before `top1` (`EMEANS_CUDA_GATHER`).
    gather_rows: bool,
    bound: Option<(usize, usize, CBuf<f32>)>,
    start: Option<CBuf<u32>>,
    order: Option<CBuf<u32>>,
    cseg: Option<CBuf<u32>>,
    gathered: Option<CBuf<f32>>,
    /// BF16 kernels (the opt-in BF16 mode, and the screen of the FP32 mode).
    bk: Bf16Kernels,
    /// Opt-in BF16 operands (`new_bf16`): `top1` is not exact.
    bf16_mode: bool,
    xb16: Option<CBuf<u16>>,
    cb16: Option<CBuf<u16>>,
    cn16: Option<CBuf<f32>>,
    xnorm: Option<CBuf<f32>>,
    bf16_data: Option<Bf16Data>,
    /// Screen scratch: candidate keys, ids, dropped bounds, the list of rows
    /// left to the FP32 kernel and its length, the largest centroid norm.
    cv: Option<CBuf<f32>>,
    cjv: Option<CBuf<u32>>,
    ch: Option<CBuf<f32>>,
    lst: Option<CBuf<u32>>,
    cnt: Option<CBuf<u32>>,
    cmax: Option<CBuf<f32>>,
    /// `EMEANS_CUDA_SCREEN_STATS=1`: count screened rows and fallback rows
    /// (one sync per call) and print them when the engine is dropped.
    screen_stats: Option<(u64, u64)>,
}

impl Drop for CudaCore {
    fn drop(&mut self) {
        if let Some((rows, fb)) = self.screen_stats.filter(|s| s.0 > 0) {
            eprintln!(
                "screen: {fb} of {rows} rows on the FP32 fallback ({:.3}%)",
                100.0 * fb as f64 / rows as f64
            );
        }
    }
}

impl CudaCore {
    /// FP32 operands, FP32 accumulation: the exact path (default).
    pub fn new() -> Result<Self> {
        Self::build(false)
    }

    /// Opt-in BF16 operands on tensor cores, FP32 accumulation. Not exact:
    /// labels can differ beyond near-ties; it does not pass `contract::check`.
    /// Rows need `d % 8 == 0`, otherwise `top1` uses the FP32 kernel.
    pub fn new_bf16() -> Result<Self> {
        Self::build(true)
    }

    fn build(bf16: bool) -> Result<Self> {
        // cudarc panics when it cannot load libcuda (dynamic loading); report
        // that as an error like a missing device.
        let ctx = std::panic::catch_unwind(|| CudaContext::new(0))
            .map_err(|_| {
                anyhow::anyhow!("no CUDA device: the CUDA driver library is not loadable")
            })?
            .context("no CUDA device")?;
        // One stream: cudarc's per-argument event tracking only costs time.
        // SAFETY: every buffer of this engine is used on `stream` only.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        let (maj, min) = ctx.compute_capability()?;
        let arch = format!("--gpu-architecture=compute_{maj}{min}");
        let compile = |src: &str, fmad: bool, name: &str| -> Result<Arc<CudaModule>> {
            let ptx = compile_ptx_with_opts(
                src,
                CompileOptions {
                    fmad: Some(fmad),
                    options: vec![arch.clone(), "-std=c++17".into()],
                    ..Default::default()
                },
            )
            .map_err(|e| anyhow::anyhow!("NVRTC {name}: {e:?}"))?;
            Ok(ctx.load_module(ptx)?)
        };
        let top1_mod = compile(TOP1_SOURCE, true, "top1")?;
        let ops_mod = compile(OPS_SOURCE, false, "ops")?;
        let f = |m: &Arc<CudaModule>, name: &str| -> Result<CudaFunction> {
            m.load_function(name)
                .with_context(|| format!("no kernel {name}"))
        };
        let mut top1 = Vec::new();
        for (bm, bn, v) in [
            (128, 128, true),
            (128, 64, true),
            (64, 128, true),
            (64, 64, true),
            (128, 128, false),
            (64, 64, false),
        ] {
            let name = format!("top1_{bm}_{bn}_{}", if v { "v4" } else { "s" });
            let func = f(&top1_mod, &name)?;
            let occ = func
                .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
                .unwrap_or(1)
                .max(1) as usize;
            if std::env::var("EMEANS_CUDA_VERBOSE").is_ok() {
                eprintln!(
                    "{name}: {} registers, {occ} blocks per SM",
                    func.num_regs().unwrap_or(-1)
                );
            }
            top1.push(((bm, bn, v), func, occ));
        }
        let sms = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .unwrap_or(28) as usize;
        let tile = match std::env::var("EMEANS_CUDA_TILE") {
            Ok(v) => {
                let t: Vec<usize> = v
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()
                    .context("EMEANS_CUDA_TILE=BM,BN")?;
                ensure!(t.len() == 2, "EMEANS_CUDA_TILE=BM,BN");
                Some((t[0], t[1]))
            }
            Err(_) => None,
        };
        let gather_rows = std::env::var("EMEANS_CUDA_GATHER").is_ok_and(|v| v == "1");
        let combine = f(&top1_mod, "top1_combine")?;
        let parts = match std::env::var("EMEANS_CUDA_PARTS") {
            Ok(v) => Some(v.parse::<usize>().context("EMEANS_CUDA_PARTS")?.max(1)),
            Err(_) => None,
        };
        let bk = {
            let m = compile(TOP1_BF16_SOURCE, true, "top1_bf16")?;
            let load = |name: &str, tn: usize, st: usize| -> Result<Bf16Kernel> {
                let func = f(&m, name)?;
                let smem = (128 + tn) * 32 * 2 * st;
                func.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem as i32,
                )?;
                let occ = func
                    .occupancy_max_active_blocks_per_multiprocessor(256, smem, None)
                    .unwrap_or(1)
                    .max(1);
                if std::env::var("EMEANS_CUDA_VERBOSE").is_ok() {
                    eprintln!(
                        "{name}: {} registers, {smem} B shared, {occ} blocks per SM",
                        func.num_regs().unwrap_or(-1)
                    );
                }
                Ok(((tn, st), func, occ, smem))
            };
            let mut v2 = Vec::new();
            for (tn, st) in [(128usize, 3usize), (128, 4), (256, 3)] {
                v2.push(load(&format!("top1_bf16_{tn}_{st}"), tn, st)?);
            }
            let mut screen = Vec::new();
            for (tn, st) in [(128usize, 3usize), (256, 3)] {
                screen.push(load(&format!("screen_{tn}_{st}"), tn, st)?);
            }
            let screen_pick = match std::env::var("EMEANS_CUDA_SCREEN").as_deref() {
                Ok("0") => None,
                Ok(v) => {
                    let t: Vec<usize> = v
                        .split('x')
                        .map(|s| s.trim().parse())
                        .collect::<Result<_, _>>()
                        .context("EMEANS_CUDA_SCREEN=0 or TNxSTAGES (128x3, 256x3)")?;
                    ensure!(t.len() == 2, "EMEANS_CUDA_SCREEN=TNxSTAGES");
                    ensure!(
                        screen.iter().any(|(k, ..)| *k == (t[0], t[1])),
                        "no screen kernel {v}"
                    );
                    Some((t[0], t[1]))
                }
                Err(_) => Some((0, 3)),
            };
            let pick = match std::env::var("EMEANS_CUDA_BF16").as_deref() {
                Ok("v1") => None,
                Ok(v) => {
                    let t: Vec<usize> = v
                        .split('x')
                        .map(|s| s.trim().parse())
                        .collect::<Result<_, _>>()
                        .context("EMEANS_CUDA_BF16=v1 or TNxSTAGES (128x3, 128x4, 256x3)")?;
                    ensure!(t.len() == 2, "EMEANS_CUDA_BF16=TNxSTAGES");
                    ensure!(
                        v2.iter().any(|(k, ..)| *k == (t[0], t[1])),
                        "no BF16 kernel {v}"
                    );
                    Some((t[0], t[1]))
                }
                Err(_) => Some((0, 3)),
            };
            Bf16Kernels {
                conv: f(&m, "gather_bf16")?,
                v1: f(&m, "top1_bf16")?,
                v2,
                combine: f(&m, "top1_bf16_combine")?,
                pick,
                screen,
                verify: f(&m, "screen_verify")?,
                max_aux: f(&m, "max_aux")?,
                conv16: f(&m, "gather_f16s")?,
                screen_pick,
            }
        };
        Ok(Self {
            gather: f(&ops_mod, "gather_rows")?,
            row_norms_f: f(&ops_mod, "row_norms")?,
            normalize_f: f(&ops_mod, "normalize_rows")?,
            count: f(&ops_mod, "scatter_count")?,
            scan: f(&ops_mod, "scan_counts")?,
            place: f(&ops_mod, "scatter_place")?,
            sum: f(&ops_mod, "scatter_sum")?,
            ema_pos: f(&ops_mod, "ema_positions")?,
            ema_usage: f(&ops_mod, "ema_usage")?,
            topk_f: f(&ops_mod, "topk_warp")?,
            split_f: f(&ops_mod, "split_pairs")?,
            _ctx: ctx,
            stream,
            _top1_mod: top1_mod,
            _ops_mod: ops_mod,
            top1,
            combine,
            pv: None,
            pj: None,
            pxn: None,
            parts,
            sms,
            tile,
            gather_rows,
            bound: None,
            start: None,
            order: None,
            cseg: None,
            gathered: None,
            bk,
            bf16_mode: bf16,
            xb16: None,
            cb16: None,
            cn16: None,
            xnorm: None,
            bf16_data: None,
            cv: None,
            cjv: None,
            ch: None,
            lst: None,
            cnt: None,
            cmax: None,
            screen_stats: std::env::var("EMEANS_CUDA_SCREEN_STATS")
                .is_ok_and(|v| v == "1")
                .then_some((0, 0)),
        })
    }

    /// Turn the BF16 screen of the exact FP32 `top1` on or off (on by
    /// default; `EMEANS_CUDA_SCREEN=0` turns it off at build). Off: the FP32
    /// kernel alone. For tests and benches.
    pub fn set_screen(&mut self, on: bool) {
        self.bk.screen_pick = on.then_some((0, 3));
    }

    /// Rows of the most recent screened `top1` that went to the FP32 kernel
    /// (waits for the device); 0 before any screened call.
    pub fn screen_fallback_rows(&mut self) -> Result<u32> {
        match self.cnt.clone() {
            Some(c) => Ok(self.read(&c, 0..1)?[0]),
            None => Ok(0),
        }
    }

    fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CBuf<T>> {
        let s = self.stream.alloc_zeros::<T>(len.max(1))?;
        Ok(CBuf {
            s: Arc::new(s),
            len,
        })
    }

    fn upload<T: DeviceRepr + ValidAsZeroBits>(&self, v: &[T]) -> Result<CBuf<T>> {
        if v.is_empty() {
            return self.alloc(0);
        }
        Ok(CBuf {
            s: Arc::new(self.stream.clone_htod(v)?),
            len: v.len(),
        })
    }

    fn write<T: DeviceRepr>(&self, b: &mut CBuf<T>, offset: usize, v: &[T]) -> Result<()> {
        ensure!(offset + v.len() <= b.len, "write past the buffer");
        if v.is_empty() {
            return Ok(());
        }
        let s = Arc::get_mut(&mut b.s).context("cannot write a shared device buffer")?;
        let mut view = s.slice_mut(offset..offset + v.len());
        self.stream.memcpy_htod(v, &mut view)?;
        Ok(())
    }

    fn read<T: DeviceRepr>(&self, b: &CBuf<T>, range: Range<usize>) -> Result<Vec<T>> {
        ensure!(
            range.start <= range.end && range.end <= b.len,
            "read past the buffer"
        );
        if range.is_empty() {
            return Ok(Vec::new());
        }
        let v = self.stream.clone_dtoh(&b.s.slice(range))?;
        self.stream.synchronize()?;
        Ok(v)
    }

    fn scratch<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        slot: &mut Option<CBuf<T>>,
        len: usize,
    ) -> Result<CBuf<T>> {
        if slot.as_ref().is_none_or(|b| b.len < len) {
            *slot = Some(self.alloc(len.max(1024))?);
        }
        Ok(slot.as_ref().unwrap().clone())
    }

    /// BF16 operands for one `top1`: the rows (the cached copy of the whole
    /// buffer, or this call's rows converted; see [`Bf16Data`]) with their
    /// squared norms, and the centroids with theirs.
    fn bf16_prepare(
        &mut self,
        x: &CBuf<f32>,
        sel: &Select<'_, CBuf<u32>>,
        centroids: &CBuf<f32>,
        k: usize,
        f16: bool,
        row_norms: bool,
    ) -> Result<Bf16Prep> {
        let d = (centroids.len / k) as u32;
        let n = (x.len / d as usize) as u32;
        let r = sel.rows.clone();
        let rows = r.len();
        // Per row: BF16 one squared norm, FP16 an aux row of four floats.
        let per = if f16 { 4 } else { 1 };
        let mut slot = self.cb16.take();
        let cb = self.scratch(&mut slot, k * d as usize)?;
        self.cb16 = slot;
        let mut slot = self.cn16.take();
        let cn = self.scratch(&mut slot, k * per)?;
        self.cn16 = slot;
        let r0 = r.start as u32;
        // Same data buffer (and format) as the last call: convert all of it
        // once and read rows by id from then on. Otherwise convert only this
        // call's rows.
        let same = self.bf16_data.as_ref().is_some_and(|c| {
            c.len == x.len && c.f16 == f16 && std::ptr::eq(c.src.as_ptr(), Arc::as_ptr(&x.s))
        });
        if !same {
            self.bf16_data = Some(Bf16Data {
                src: Arc::downgrade(&x.s),
                len: x.len,
                f16,
                rows: None,
            });
        } else if self.bf16_data.as_ref().is_some_and(|c| c.rows.is_none()) {
            let xb = self.alloc::<u16>(x.len)?;
            let xn = self.alloc::<f32>(n as usize * per)?;
            self.convert(x, None, &xb, &xn, 0, n, d, f16, true)?;
            self.bf16_data.as_mut().unwrap().rows = Some((xb, xn));
        }
        let cached = self.bf16_data.as_ref().and_then(|c| c.rows.clone());
        let prep = match cached {
            Some((xb, xn)) => Bf16Prep {
                xb,
                xn,
                indexed: sel.ids.is_some() as u32,
                xr0: r0,
                dense: false,
                cb,
                cn,
            },
            None => {
                let mut slot = self.xb16.take();
                let xb = self.scratch(&mut slot, rows * d as usize)?;
                self.xb16 = slot;
                let mut slot = self.xnorm.take();
                let xn = self.scratch(&mut slot, rows * per)?;
                self.xnorm = slot;
                self.convert(x, sel.ids, &xb, &xn, r0, rows as u32, d, f16, row_norms)?;
                Bf16Prep {
                    xb,
                    xn,
                    indexed: 0,
                    xr0: 0,
                    dense: true,
                    cb,
                    cn,
                }
            }
        };
        self.convert(
            centroids, None, &prep.cb, &prep.cn, 0, k as u32, d, f16, true,
        )?;
        Ok(prep)
    }

    /// Convert `rows` rows of `x` (row i = `ids[r0 + i]`, or `r0 + i`) to
    /// BF16 (`norms`: squared norms, when `want`) or to the FP16 screen copy
    /// (`norms`: aux rows, see `gather_f16s`).
    #[allow(clippy::too_many_arguments)]
    fn convert(
        &self,
        x: &CBuf<f32>,
        ids: Option<&CBuf<u32>>,
        out: &CBuf<u16>,
        norms: &CBuf<f32>,
        r0: u32,
        rows: u32,
        d: u32,
        f16: bool,
        want: bool,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let indexed = ids.is_some() as u32;
        let want = want as u32;
        let func = if f16 { &self.bk.conv16 } else { &self.bk.conv };
        let mut b = self.stream.launch_builder(func);
        b.arg(&*x.s);
        // Without ids the kernel never reads its idx argument.
        match ids {
            Some(i) => b.arg(&*i.s),
            None => b.arg(&*x.s),
        };
        b.arg(&*out.s)
            .arg(&*norms.s)
            .arg(&r0)
            .arg(&rows)
            .arg(&d)
            .arg(&indexed);
        if !f16 {
            b.arg(&want);
        }
        // SAFETY: rows x d floats in (by id or position), rows x d halves and
        // rows norms (or aux rows of four) out.
        unsafe { b.launch(cfg(((rows as usize).div_ceil(8), 1), (256, 1)))? };
        Ok(())
    }

    /// Opt-in BF16 top-1 (not exact).
    #[allow(clippy::too_many_arguments)]
    fn top1_bf16(
        &mut self,
        x: &CBuf<f32>,
        sel: Select<'_, CBuf<u32>>,
        centroids: &CBuf<f32>,
        cnorm: Option<&CBuf<f32>>,
        k: usize,
        metric: Metric,
        labels: &CBuf<u32>,
        scores: &CBuf<f32>,
    ) -> Result<()> {
        let kern = self.bk.clone();
        let d = (centroids.len / k) as u32;
        let r = sel.rows.clone();
        let rows = r.len();
        let l2 = (metric == Metric::L2) as u32;
        let (ku, ru, r0) = (k as u32, rows as u32, r.start as u32);
        let prep = self.bf16_prepare(x, &sel, centroids, k, false, l2 == 1)?;
        let ids: &CudaSlice<u32> = sel.ids.map(|i| &*i.s).unwrap_or(&*labels.s);
        let cn = cnorm.unwrap_or(centroids);
        let Some((tn, st)) = kern.pick.map(|t| auto_tile(t, k)) else {
            let p = Bf16P {
                rows: ru,
                k: ku,
                d,
                ip: 1 - l2,
                r0,
                xr0: prep.xr0,
                indexed: prep.indexed,
            };
            let mut b = self.stream.launch_builder(&kern.v1);
            b.arg(&*prep.xb.s)
                .arg(ids)
                .arg(&*prep.cb.s)
                .arg(&*prep.xn.s)
                .arg(&*cn.s)
                .arg(&*labels.s)
                .arg(&*scores.s)
                .arg(&p);
            // SAFETY: rows read by id or position; outputs at positions r.
            unsafe { b.launch(cfg((rows.div_ceil(128), 1), (256, 1)))? };
            return Ok(());
        };
        let (_, func, occ, smem) = kern
            .v2
            .iter()
            .find(|(t, ..)| *t == (tn, st))
            .cloned()
            .context("no BF16 kernel")?;
        let (chunk, parts) = self.split_parts(rows, 128, k, tn, occ as usize);
        let p = Bf16P2 {
            rows: ru,
            k: ku,
            d,
            ip: 1 - l2,
            r0,
            xr0: prep.xr0,
            indexed: prep.indexed,
            chunk: u32::try_from(chunk)?,
            parts: parts as u32,
        };
        let (pv, pj, pxn) = self.partials(parts, rows, labels, scores)?;
        let mut b = self.stream.launch_builder(&func);
        b.arg(&*prep.xb.s)
            .arg(ids)
            .arg(&*prep.cb.s)
            .arg(&*prep.xn.s)
            .arg(&*cn.s)
            .arg(&*labels.s)
            .arg(&*scores.s)
            .arg(&*pv.s)
            .arg(&*pj.s)
            .arg(&*pxn.s)
            .arg(&p);
        let mut lc = cfg((rows.div_ceil(128), parts), (256, 1));
        lc.shared_mem_bytes = smem as u32;
        // SAFETY: rows read by id or position; outputs at positions r, or parts x rows partials.
        unsafe { b.launch(lc)? };
        if parts > 1 {
            let pu = parts as u32;
            let mut b = self.stream.launch_builder(&kern.combine);
            b.arg(&*pv.s)
                .arg(&*pj.s)
                .arg(&*pxn.s)
                .arg(&*labels.s)
                .arg(&*scores.s)
                .arg(&ru)
                .arg(&pu)
                .arg(&p.ip)
                .arg(&r0);
            // SAFETY: parts x rows partials; outputs at positions r.
            unsafe { b.launch(cfg((rows.div_ceil(256), 1), (256, 1)))? };
        }
        Ok(())
    }

    /// Exact FP32 top-1 through a BF16 screen: the screen keeps each row's
    /// candidates (every centroid that can be the FP32 best, by a rigorous
    /// error bound), `screen_verify` scores them with the FP32 kernel's
    /// arithmetic, and the rows whose candidates may be incomplete go to the
    /// FP32 kernel. Labels and scores are bit-equal to the FP32 kernel's.
    #[allow(clippy::too_many_arguments)]
    fn top1_screen(
        &mut self,
        x: &CBuf<f32>,
        sel: Select<'_, CBuf<u32>>,
        centroids: &CBuf<f32>,
        cnorm: Option<&CBuf<f32>>,
        k: usize,
        metric: Metric,
        labels: &CBuf<u32>,
        scores: &CBuf<f32>,
        fp32: Top1P,
        pick: (usize, usize),
    ) -> Result<()> {
        let kern = self.bk.clone();
        let d = (centroids.len / k) as u32;
        let r = sel.rows.clone();
        let rows = r.len();
        let l2 = (metric == Metric::L2) as u32;
        let (ku, ru, r0) = (k as u32, rows as u32, r.start as u32);
        let prep = self.bf16_prepare(x, &sel, centroids, k, true, true)?;
        let ids: &CudaSlice<u32> = sel.ids.map(|i| &*i.s).unwrap_or(&*labels.s);
        let cn = cnorm.unwrap_or(centroids);
        let (_, func, occ, smem) = kern
            .screen
            .iter()
            .find(|(t, ..)| *t == auto_tile(pick, k))
            .cloned()
            .context("no screen kernel")?;
        let (chunk, parts) = self.split_parts(rows, 128, k, auto_tile(pick, k).0, occ as usize);
        let mut slot = self.cv.take();
        let cv = self.scratch(&mut slot, parts * rows * 4)?;
        self.cv = slot;
        let mut slot = self.cjv.take();
        let cjv = self.scratch(&mut slot, parts * rows * 4)?;
        self.cjv = slot;
        let mut slot = self.ch.take();
        let ch = self.scratch(&mut slot, parts * rows)?;
        self.ch = slot;
        let mut slot = self.lst.take();
        let lst = self.scratch(&mut slot, rows)?;
        self.lst = slot;
        let mut slot = self.cnt.take();
        let cnt = self.scratch(&mut slot, 1)?;
        self.cnt = slot;
        let mut slot = self.cmax.take();
        let cmax = self.scratch(&mut slot, 3)?;
        self.cmax = slot;
        let p = Bf16P2 {
            rows: ru,
            k: ku,
            d,
            ip: 1 - l2,
            r0,
            xr0: prep.xr0,
            indexed: prep.indexed,
            chunk: u32::try_from(chunk)?,
            parts: parts as u32,
        };
        let sp = ScreenP {
            rows: ru,
            k: ku,
            d,
            ip: 1 - l2,
            r0,
            xr0: r0,
            indexed: sel.ids.is_some() as u32,
            parts: parts as u32,
            dense_norm: prep.dense as u32,
            eps: screen_eps(d as usize),
        };
        // SAFETY (all three): shapes checked by `top1`; aux rows are 4 floats;
        // the screen writes parts x rows x 4 candidates and parts x rows bounds
        // and zeroes cnt; verify writes positions r or appends to lst.
        unsafe {
            let mut b = self.stream.launch_builder(&kern.max_aux);
            b.arg(&*prep.cn.s).arg(&*cmax.s).arg(&ku);
            b.launch(cfg((1, 1), (256, 1)))?;
            let mut b = self.stream.launch_builder(&func);
            b.arg(&*prep.xb.s)
                .arg(ids)
                .arg(&*prep.cb.s)
                .arg(&*cn.s)
                .arg(&*prep.xn.s)
                .arg(&*prep.cn.s)
                .arg(&*cv.s)
                .arg(&*cjv.s)
                .arg(&*ch.s)
                .arg(&*cnt.s)
                .arg(&p);
            let mut lc = cfg((rows.div_ceil(128), parts), (256, 1));
            lc.shared_mem_bytes = smem as u32;
            b.launch(lc)?;
            let mut b = self.stream.launch_builder(&kern.verify);
            b.arg(&*x.s)
                .arg(ids)
                .arg(&*centroids.s)
                .arg(&*cn.s)
                .arg(&*prep.xn.s)
                .arg(&*cmax.s)
                .arg(&*cv.s)
                .arg(&*cjv.s)
                .arg(&*ch.s)
                .arg(&*labels.s)
                .arg(&*scores.s)
                .arg(&*lst.s)
                .arg(&*cnt.s)
                .arg(&sp);
            b.launch(cfg((rows.div_ceil(128), 1), (128, 1)))?;
        }
        // The rows left over, on the FP32 kernel. Their number stays on the
        // device: blocks past it return at once. The centroids are split in up
        // to 8 parts so a few rows still fill the GPU.
        let (bm, bn, f, _) = self.pick(rows, k, d as usize)?;
        let tiles = k.div_ceil(bn);
        let fparts = tiles.min(8);
        let fchunk = tiles.div_ceil(fparts) * bn;
        let fparts = k.div_ceil(fchunk);
        let mut q = fp32;
        q.list = 1;
        q.parts = fparts as u32;
        q.chunk = u32::try_from(fchunk)?;
        let (pv, pj, pxn) = self.partials(fparts, rows, labels, scores)?;
        let mut b = self.stream.launch_builder(&f);
        b.arg(&*x.s)
            .arg(ids)
            .arg(&*centroids.s)
            .arg(&*cn.s)
            .arg(&*labels.s)
            .arg(&*scores.s)
            .arg(&*pv.s)
            .arg(&*pj.s)
            .arg(&*pxn.s)
            .arg(&*lst.s)
            .arg(&*cnt.s)
            .arg(&q);
        // SAFETY: as the plain FP32 dispatch, rows from lst.
        unsafe { b.launch(cfg((rows.div_ceil(bm), fparts), (256, 1)))? };
        if fparts > 1 {
            let (pu, one) = (fparts as u32, 1u32);
            let mut b = self.stream.launch_builder(&self.combine);
            b.arg(&*pv.s)
                .arg(&*pj.s)
                .arg(&*pxn.s)
                .arg(&*labels.s)
                .arg(&*scores.s)
                .arg(&ru)
                .arg(&pu)
                .arg(&q.ip)
                .arg(&r0)
                .arg(&*lst.s)
                .arg(&*cnt.s)
                .arg(&one);
            // SAFETY: fparts x rows partials of the first *cnt list rows.
            unsafe { b.launch(cfg((rows.div_ceil(256), 1), (256, 1)))? };
        }
        if let Some(st) = self.screen_stats {
            let fb = self.read(&cnt, 0..1)?[0] as u64;
            self.screen_stats = Some((st.0 + rows as u64, st.1 + fb));
        }
        Ok(())
    }

    /// Centroids per part and parts: split the centroids over `parts` blocks
    /// per row tile when that fills the last wave better.
    fn split_parts(
        &self,
        rows: usize,
        bm: usize,
        k: usize,
        bn: usize,
        occ: usize,
    ) -> (usize, usize) {
        let row_blocks = rows.div_ceil(bm);
        let tiles = k.div_ceil(bn);
        let resident = (occ * self.sms).max(1);
        let parts = self.parts.unwrap_or_else(|| {
            (1..=4usize)
                .filter(|&q| q == 1 || tiles / q >= 2)
                .min_by(|&a, &b| {
                    let cost = |q: usize| (row_blocks * q).div_ceil(resident) as f64 / q as f64;
                    cost(a).total_cmp(&cost(b)).then(a.cmp(&b))
                })
                .unwrap_or(1)
        });
        let parts = parts.clamp(1, tiles.max(1));
        let chunk = tiles.div_ceil(parts) * bn;
        (chunk, k.div_ceil(chunk))
    }

    /// Partial-best buffers for a split dispatch (the outputs when not split).
    fn partials(
        &mut self,
        parts: usize,
        rows: usize,
        labels: &CBuf<u32>,
        scores: &CBuf<f32>,
    ) -> Result<(CBuf<f32>, CBuf<u32>, CBuf<f32>)> {
        if parts <= 1 {
            return Ok((scores.clone(), labels.clone(), scores.clone()));
        }
        let mut a = self.pv.take();
        let pv = self.scratch(&mut a, parts * rows)?;
        self.pv = a;
        let mut a = self.pj.take();
        let pj = self.scratch(&mut a, parts * rows)?;
        self.pj = a;
        let mut a = self.pxn.take();
        let pxn = self.scratch(&mut a, rows)?;
        self.pxn = a;
        Ok((pv, pj, pxn))
    }

    /// Kernel tile (BM, BN) for a dispatch.
    fn pick(&self, rows: usize, k: usize, d: usize) -> Result<(usize, usize, CudaFunction, usize)> {
        let vec = d.is_multiple_of(4);
        let (bm, bn) = match (self.tile, vec) {
            (_, false) => {
                if k <= 64 {
                    (64, 64)
                } else {
                    (128, 128)
                }
            }
            (Some(t), true) => t,
            (None, true) => {
                let bn = if k <= 64 { 64 } else { 128 };
                // Prefer 128-row blocks when they still give two waves of blocks.
                let bm = if rows.div_ceil(128) >= 2 * self.sms {
                    128
                } else {
                    64
                };
                (bm, bn)
            }
        };
        let (f, occ) = self
            .top1
            .iter()
            .find(|(t, _, _)| *t == (bm, bn, vec))
            .map(|(_, f, o)| (f.clone(), *o))
            .with_context(|| format!("no top1 kernel for tile {bm}x{bn} vec={vec}"))?;
        Ok((bm, bn, f, occ))
    }
}

fn cfg(grid: (usize, usize), block: (usize, usize)) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0 as u32, grid.1 as u32, 1),
        block_dim: (block.0 as u32, block.1 as u32, 1),
        shared_mem_bytes: 0,
    }
}

impl Device for CudaCore {
    type F = CBuf<f32>;
    type U = CBuf<u32>;

    fn name(&self) -> String {
        if self.bf16_mode {
            "cuda-fused-bf16".into()
        } else {
            "cuda-fused-f32".into()
        }
    }

    fn bind(&mut self, x: &[f32]) -> Result<CBuf<f32>> {
        if let Some((p, l, b)) = &self.bound
            && *p == x.as_ptr() as usize
            && *l == x.len()
        {
            return Ok(b.clone());
        }
        self.upload(x)
    }

    fn upload_f32(&mut self, v: &[f32]) -> Result<CBuf<f32>> {
        self.upload(v)
    }

    fn upload_u32(&mut self, v: &[u32]) -> Result<CBuf<u32>> {
        self.upload(v)
    }

    fn zeros_f32(&mut self, n: usize) -> Result<CBuf<f32>> {
        self.alloc(n)
    }

    fn zeros_u32(&mut self, n: usize) -> Result<CBuf<u32>> {
        self.alloc(n)
    }

    fn write_f32(&mut self, b: &mut CBuf<f32>, offset: usize, v: &[f32]) -> Result<()> {
        // A write changes the data: drop its BF16 copy (and the weak handle
        // that would make the buffer look shared).
        if self
            .bf16_data
            .as_ref()
            .is_some_and(|c| std::ptr::eq(c.src.as_ptr(), Arc::as_ptr(&b.s)))
        {
            self.bf16_data = None;
        }
        self.write(b, offset, v)
    }

    fn write_u32(&mut self, b: &mut CBuf<u32>, offset: usize, v: &[u32]) -> Result<()> {
        self.write(b, offset, v)
    }

    fn read_f32(&mut self, b: &CBuf<f32>, range: Range<usize>) -> Result<Vec<f32>> {
        self.read(b, range)
    }

    fn read_u32(&mut self, b: &CBuf<u32>, range: Range<usize>) -> Result<Vec<u32>> {
        self.read(b, range)
    }

    fn top1(
        &mut self,
        x: Mat<'_, CBuf<f32>>,
        sel: Select<'_, CBuf<u32>>,
        centroids: &CBuf<f32>,
        cnorm: Option<&CBuf<f32>>,
        k: usize,
        metric: Metric,
        labels: &mut CBuf<u32>,
        scores: &mut CBuf<f32>,
    ) -> Result<()> {
        let d = x.d;
        check_select(&sel, sel.ids.map(|i| i.len), x.dev.len / d)?;
        let r = sel.rows.clone();
        ensure!(
            r.end <= labels.len && r.end <= scores.len,
            "outputs too short"
        );
        ensure!(centroids.len == k * d, "centroids must be k x d");
        if metric == Metric::L2 {
            ensure!(cnorm.is_some_and(|c| c.len >= k), "L2 needs centroid norms");
        }
        if r.is_empty() {
            return Ok(());
        }
        let mut p = Top1P {
            rows: u32::try_from(r.len())?,
            k: u32::try_from(k)?,
            d: u32::try_from(d)?,
            ip: (metric == Metric::InnerProduct) as u32,
            indexed: sel.ids.is_some() as u32,
            r0: u32::try_from(r.start)?,
            xr0: u32::try_from(r.start)?,
            chunk: 0,
            parts: 1,
            list: 0,
        };
        if d.is_multiple_of(8) {
            if self.bf16_mode {
                return self.top1_bf16(x.dev, sel, centroids, cnorm, k, metric, labels, scores);
            }
            if let (Some(pick), false, true) =
                (self.bk.screen_pick, self.gather_rows, k >= SCREEN_MIN_K)
            {
                return self.top1_screen(
                    x.dev, sel, centroids, cnorm, k, metric, labels, scores, p, pick,
                );
            }
        }
        let mut xs = x.dev.clone();
        if let (Some(ids), true) = (sel.ids, self.gather_rows) {
            let mut slot = self.gathered.take();
            let g = self.scratch(&mut slot, r.len() * d)?;
            self.gathered = slot;
            let mut b = self.stream.launch_builder(&self.gather);
            b.arg(&*x.dev.s)
                .arg(&*ids.s)
                .arg(&*g.s)
                .arg(&p.r0)
                .arg(&p.rows)
                .arg(&p.d);
            // SAFETY: the kernel reads x rows by id and writes rows x d floats of g.
            unsafe { b.launch(cfg((d.div_ceil(64), r.len().div_ceil(4)), (64, 4)))? };
            p.indexed = 0;
            p.xr0 = 0;
            xs = g;
        }
        // Without ids the kernel never reads its idx argument; bind any buffer.
        let ids: &CudaSlice<u32> = match sel.ids {
            Some(i) if p.indexed == 1 => &i.s,
            _ => &labels.s,
        };
        let (bm, bn, f, occ) = self.pick(r.len(), k, d)?;
        // Split the centroids across `parts` blocks per row tile when that
        // fills the last wave better (a 16k-row batch is 128 blocks of 128 rows).
        let (chunk, parts) = self.split_parts(r.len(), bm, k, bn, occ);
        let row_blocks = r.len().div_ceil(bm);
        p.chunk = u32::try_from(chunk)?;
        p.parts = parts as u32;
        let (pv, pj, pxn) = self.partials(parts, r.len(), labels, scores)?;
        let cn = cnorm.unwrap_or(centroids);
        let mut b = self.stream.launch_builder(&f);
        b.arg(&*xs.s)
            .arg(ids)
            .arg(&*centroids.s)
            .arg(&*cn.s)
            .arg(&*labels.s)
            .arg(&*scores.s)
            .arg(&*pv.s)
            .arg(&*pj.s)
            .arg(&*pxn.s)
            .arg(&*labels.s)
            .arg(&*labels.s)
            .arg(&p);
        // SAFETY: shapes checked above; the kernel writes positions r (or the partials) only.
        unsafe { b.launch(cfg((row_blocks, parts), (256, 1)))? };
        if parts > 1 {
            let (rows, pu) = (r.len() as u32, parts as u32);
            let mut b = self.stream.launch_builder(&self.combine);
            b.arg(&*pv.s)
                .arg(&*pj.s)
                .arg(&*pxn.s)
                .arg(&*labels.s)
                .arg(&*scores.s)
                .arg(&rows)
                .arg(&pu)
                .arg(&p.ip)
                .arg(&p.r0)
                .arg(&*labels.s)
                .arg(&*labels.s)
                .arg(&p.list);
            // SAFETY: parts x rows partials; outputs at positions r.
            unsafe { b.launch(cfg((r.len().div_ceil(256), 1), (256, 1)))? };
        }
        Ok(())
    }

    fn topk(
        &mut self,
        queries: &[f32],
        data: &[f32],
        d: usize,
        kk: usize,
        exclude_self: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        ensure!(d > 0 && queries.len().is_multiple_of(d) && data.len().is_multiple_of(d));
        ensure!((1..=TOPK_KMAX).contains(&kk), "kk must be 1..={TOPK_KMAX}");
        let (m, n) = (queries.len() / d, data.len() / d);
        ensure!(n > 0, "no data rows");
        ensure!(
            !exclude_self || m == n,
            "exclude_self needs queries == data"
        );
        let qb = self.upload(queries)?;
        let db = self.upload(data)?;
        let ids = self.alloc::<u32>(m * kk)?;
        let sims = self.alloc::<f32>(m * kk)?;
        let (mu, nu, du, ku, ex) = (m as u32, n as u32, d as u32, kk as u32, exclude_self as u32);
        let mut b = self.stream.launch_builder(&self.topk_f);
        b.arg(&*qb.s)
            .arg(&*db.s)
            .arg(&*ids.s)
            .arg(&*sims.s)
            .arg(&mu)
            .arg(&nu)
            .arg(&du)
            .arg(&ku)
            .arg(&ex);
        // SAFETY: m x kk outputs; the kernel bounds-checks queries.
        unsafe { b.launch(cfg((m.div_ceil(4), 1), (128, 1)))? };
        Ok((self.read(&ids, 0..m * kk)?, self.read(&sims, 0..m * kk)?))
    }

    fn scatter_add(
        &mut self,
        x: Mat<'_, CBuf<f32>>,
        sel: Select<'_, CBuf<u32>>,
        labels: &CBuf<u32>,
        k: usize,
        sums: &mut CBuf<f32>,
        counts: &mut CBuf<u32>,
        acc: Option<&mut CBuf<f32>>,
    ) -> Result<()> {
        let d = x.d;
        check_select(&sel, sel.ids.map(|i| i.len), x.dev.len / d)?;
        ensure!(sums.len == k * d && counts.len == k, "sums k x d, counts k");
        ensure!(sel.rows.end <= labels.len, "labels too short");
        if let Some(a) = &acc {
            ensure!(a.len == k * d, "acc must be k x d");
        }
        let m = sel.rows.len();
        // Positions per segment and clusters per warp: SEG and SG in OPS_SOURCE.
        let (seg, sg) = (2048usize, 8usize);
        let segs = m.div_ceil(seg).max(1);
        let mut slot = self.start.take();
        let start = self.scratch(&mut slot, k + 1)?;
        self.start = slot;
        let mut slot = self.order.take();
        let order = self.scratch(&mut slot, m.max(1))?;
        self.order = slot;
        let mut slot = self.cseg.take();
        let cseg = self.scratch(&mut slot, segs * k)?;
        self.cseg = slot;
        let p = ScatterP {
            lo: u32::try_from(sel.rows.start)?,
            m: u32::try_from(m)?,
            k: u32::try_from(k)?,
            d: u32::try_from(d)?,
            indexed: sel.ids.is_some() as u32,
            acc: acc.is_some() as u32,
            segs: u32::try_from(segs)?,
        };
        let ids = sel.ids.unwrap_or(&start);
        let accb = acc.map(|a| a.clone()).unwrap_or_else(|| sums.clone());
        // Two warps per block: k = 1024 gives 64 blocks per segment.
        let groups = cfg((k.div_ceil(sg).div_ceil(2), segs), (64, 1));
        // SAFETY (all four): buffers sized and checked above; cseg is segs x k.
        unsafe {
            let mut b = self.stream.launch_builder(&self.count);
            b.arg(&*labels.s).arg(&*cseg.s).arg(&p);
            b.launch(groups)?;
            let mut b = self.stream.launch_builder(&self.scan);
            b.arg(&*cseg.s).arg(&*counts.s).arg(&*start.s).arg(&p);
            b.launch(cfg((1, 1), (1024, 1)))?;
            let mut b = self.stream.launch_builder(&self.place);
            b.arg(&*labels.s)
                .arg(&*ids.s)
                .arg(&*cseg.s)
                .arg(&*order.s)
                .arg(&p);
            b.launch(groups)?;
            let mut b = self.stream.launch_builder(&self.sum);
            b.arg(&*x.dev.s)
                .arg(&*start.s)
                .arg(&*order.s)
                .arg(&*sums.s)
                .arg(&*accb.s)
                .arg(&p);
            b.launch(cfg((d.div_ceil(64), k.div_ceil(4)), (64, 4)))?;
        }
        Ok(())
    }

    fn row_norms(
        &mut self,
        m: &CBuf<f32>,
        rows: usize,
        d: usize,
        out: &mut CBuf<f32>,
    ) -> Result<()> {
        ensure!(m.len >= rows * d && out.len >= rows, "shapes");
        if rows == 0 {
            return Ok(());
        }
        let (r, dd) = (rows as u32, d as u32);
        let mut b = self.stream.launch_builder(&self.row_norms_f);
        b.arg(&*m.s).arg(&*out.s).arg(&r).arg(&dd);
        // SAFETY: rows x d input, rows outputs.
        unsafe { b.launch(cfg((rows, 1), (32, 1)))? };
        Ok(())
    }

    fn normalize_rows(
        &mut self,
        src: &CBuf<f32>,
        rows: usize,
        d: usize,
        dst: &mut CBuf<f32>,
    ) -> Result<()> {
        ensure!(src.len >= rows * d && dst.len >= rows * d, "shapes");
        if rows == 0 {
            return Ok(());
        }
        let (r, dd) = (rows as u32, d as u32);
        let mut b = self.stream.launch_builder(&self.normalize_f);
        b.arg(&*src.s).arg(&*dst.s).arg(&r).arg(&dd);
        // SAFETY: rows x d input and output.
        unsafe { b.launch(cfg((rows, 1), (32, 1)))? };
        Ok(())
    }

    fn split(
        &mut self,
        positions: &mut CBuf<f32>,
        usage: &mut CBuf<f32>,
        plan: &[(usize, usize)],
        d: usize,
    ) -> Result<()> {
        let k = usage.len;
        ensure!(positions.len == k * d, "positions must be k x d");
        ensure!(
            plan.iter().all(|&(s, t)| s < k && t < k),
            "split ids out of range"
        );
        if plan.is_empty() {
            return Ok(());
        }
        let flat: Vec<u32> = plan
            .iter()
            .flat_map(|&(s, t)| [s as u32, t as u32])
            .collect();
        let pairs = self.upload(&flat)?;
        let (du, eps) = (d as u32, emeans_core::recipe::SPLIT_EPSILON as f32);
        let mut b = self.stream.launch_builder(&self.split_f);
        b.arg(&*positions.s)
            .arg(&*usage.s)
            .arg(&*pairs.s)
            .arg(&du)
            .arg(&eps);
        // SAFETY: ids checked against k; k x d positions.
        unsafe { b.launch(cfg((plan.len(), 1), (256, 1)))? };
        Ok(())
    }

    fn ema_update(
        &mut self,
        positions: &mut CBuf<f32>,
        usage: &mut CBuf<f32>,
        sums: &CBuf<f32>,
        counts: &CBuf<u32>,
        k: usize,
        d: usize,
        step: EmaStep,
    ) -> Result<()> {
        ensure!(
            positions.len == k * d && usage.len == k && sums.len == k * d && counts.len == k,
            "shapes"
        );
        let p = EmaP {
            k: u32::try_from(k)?,
            d: u32::try_from(d)?,
            alpha: step.alpha,
            decay: step.decay,
            replace: step.replace as u32,
        };
        // SAFETY: k x d positions and sums, k masses and counts.
        unsafe {
            let mut b = self.stream.launch_builder(&self.ema_pos);
            b.arg(&*positions.s)
                .arg(&*usage.s)
                .arg(&*sums.s)
                .arg(&*counts.s)
                .arg(&p);
            b.launch(cfg((d.div_ceil(64), k.div_ceil(4)), (64, 4)))?;
            let mut b = self.stream.launch_builder(&self.ema_usage);
            b.arg(&*usage.s).arg(&*counts.s).arg(&p);
            b.launch(cfg((k.div_ceil(256), 1), (256, 1)))?;
        }
        Ok(())
    }
}

impl Assigner for CudaCore {
    fn name(&self) -> String {
        Device::name(self)
    }

    fn bind_data(&mut self, data: &[f32], _d: usize) -> Result<()> {
        // Always a fresh copy: the same pointer and length can hold new values
        // (written in place, or a new allocation at a freed address), so a
        // (pointer, length) match is no proof the bound copy is current.
        // `Device::bind` still serves this copy by pointer and length while it
        // is bound; the caller must not change the data until `unbind_data`.
        self.bound = None;
        let b = self.upload(data)?;
        self.stream.synchronize()?;
        self.bound = Some((data.as_ptr() as usize, data.len(), b));
        Ok(())
    }

    fn unbind_data(&mut self) {
        self.bound = None;
        self.bf16_data = None;
    }

    fn with_engine(
        &mut self,
        f: &mut dyn FnMut(&mut dyn emeans_core::core::Engine) -> Result<()>,
    ) -> Option<Result<()>> {
        Some(f(self))
    }

    fn assign(
        &mut self,
        rows: Rows<'_>,
        d: usize,
        centroids: &[f32],
        metric: Metric,
        labels: &mut [u32],
        scores: &mut [f32],
    ) -> Result<()> {
        let bound = self
            .bound
            .as_ref()
            .map(|(p, l, b)| (*p as *const f32, *l, b.clone()));
        let bref = bound.as_ref().map(|(p, l, b)| (*p, *l, b));
        emeans_core::core::assign_on(self, bref, rows, d, centroids, metric, labels, scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emeans_core::core::contract::mixture;

    /// The engine, or None (and a note) when this machine has no CUDA device.
    fn device(bf16: bool) -> Option<CudaCore> {
        match CudaCore::build(bf16) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping: {e:#}");
                None
            }
        }
    }

    /// Many segments (m > 2048), k not a multiple of the 8 clusters per warp,
    /// one cluster holding half the rows, empty clusters: sums must be the rows
    /// added in position order, bit for bit; counts exact; acc += sums.
    #[test]
    fn scatter_many_segments_is_bit_equal() {
        let Some(mut dev) = device(false) else { return };
        for (n, d, k) in [(20_000, 96, 301), (9_000, 768, 1024), (5_000, 13, 7)] {
            let x = mixture(n, d, 17, 0.4, false, n as u32);
            let labels: Vec<u32> = (0..n as u32)
                .map(|i| {
                    if i % 2 == 0 {
                        5 % k as u32
                    } else {
                        (i.wrapping_mul(2654435761) >> 9) % (k as u32 - k as u32 / 4)
                    }
                })
                .collect();
            let perm: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 3) % n as u32).collect();
            let (lo, hi) = (777, n - 123);
            let xb = dev.upload_f32(&x).unwrap();
            let ids = dev.upload_u32(&perm).unwrap();
            let lab = dev.upload_u32(&labels).unwrap();
            let mut sums = dev.upload_f32(&vec![9.0; k * d]).unwrap();
            let mut counts = dev.upload_u32(&vec![4; k]).unwrap();
            let acc0: Vec<f32> = (0..k * d).map(|i| (i % 11) as f32 * 0.5).collect();
            let mut acc = dev.upload_f32(&acc0).unwrap();
            dev.scatter_add(
                Mat {
                    host: &x,
                    dev: &xb,
                    d,
                },
                Select {
                    ids: Some(&ids),
                    rows: lo..hi,
                },
                &lab,
                k,
                &mut sums,
                &mut counts,
                Some(&mut acc),
            )
            .unwrap();
            let got_s = dev.read_f32(&sums, 0..k * d).unwrap();
            let got_c = dev.read_u32(&counts, 0..k).unwrap();
            let got_a = dev.read_f32(&acc, 0..k * d).unwrap();
            let mut want_s = vec![0f32; k * d];
            let mut want_c = vec![0u32; k];
            for p in lo..hi {
                let (l, r) = (labels[p] as usize, perm[p] as usize);
                want_c[l] += 1;
                for t in 0..d {
                    want_s[l * d + t] += x[r * d + t];
                }
            }
            assert_eq!(got_c, want_c, "counts n={n} d={d} k={k}");
            for i in 0..k * d {
                assert_eq!(got_s[i].to_bits(), want_s[i].to_bits(), "sum {i} d={d}");
                assert_eq!(
                    got_a[i].to_bits(),
                    (acc0[i] + want_s[i]).to_bits(),
                    "acc {i}"
                );
            }
        }
    }

    /// BF16 top-1: the first call converts its rows, the second converts the
    /// whole buffer, the third reads the cached copy. All three agree bit for
    /// bit, and a write to the buffer drops the cache.
    #[test]
    fn bf16_cached_rows_match_converted_rows() {
        let Some(mut dev) = device(true) else { return };
        let (n, d, k) = (5_000, 768, 300);
        let x = mixture(n, d, 40, 0.5, true, 3);
        let c: Vec<f32> = x[..k * d].iter().map(|v| v * 0.9).collect();
        let mut xb = dev.upload_f32(&x).unwrap();
        let cb = dev.upload_f32(&c).unwrap();
        let mut cn = dev.zeros_f32(k).unwrap();
        dev.row_norms(&cb, k, d, &mut cn).unwrap();
        let perm: Vec<u32> = (0..n as u32).rev().collect();
        let ids = dev.upload_u32(&perm).unwrap();
        for metric in [Metric::InnerProduct, Metric::L2] {
            dev.bf16_data = None;
            let mut runs = Vec::new();
            for _ in 0..3 {
                let mut lab = dev.zeros_u32(n).unwrap();
                let mut sco = dev.zeros_f32(n).unwrap();
                dev.top1(
                    Mat {
                        host: &x,
                        dev: &xb,
                        d,
                    },
                    Select {
                        ids: Some(&ids),
                        rows: 100..n - 50,
                    },
                    &cb,
                    Some(&cn),
                    k,
                    metric,
                    &mut lab,
                    &mut sco,
                )
                .unwrap();
                let l = dev.read_u32(&lab, 0..n).unwrap();
                let s: Vec<u32> = dev
                    .read_f32(&sco, 0..n)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                runs.push((l, s));
            }
            assert!(dev.bf16_data.as_ref().is_some_and(|c| c.rows.is_some()));
            assert_eq!(
                runs[0], runs[1],
                "{metric:?}: converted vs whole-buffer copy"
            );
            assert_eq!(runs[1], runs[2], "{metric:?}: cached copy");
        }
        dev.write_f32(&mut xb, 0, &[0.5; 8]).unwrap();
        assert!(dev.bf16_data.is_none(), "a write drops the BF16 copy");
    }

    /// The BF16 screen returns the FP32 kernel's labels and scores bit for
    /// bit: twin centroids (a split 1e-4 apart), exact duplicates (ties), d
    /// with the BK = 16 padding step, both metrics, indexed and dense rows, a
    /// centroid split, and a crowd of near-equal centroids that sends rows to
    /// the FP32 fallback.
    #[test]
    fn screen_is_bit_equal_to_fp32() {
        let Some(mut dev) = device(false) else { return };
        let pick = dev.bk.screen_pick.expect("screen on by default");
        for (n, d, k, crowd) in [
            (6000, 768, 1024, false),
            (3000, 64, 300, false),
            (2500, 72, 200, false),
            (1500, 768, 256, true),
        ] {
            for metric in [Metric::InnerProduct, Metric::L2] {
                let unit = metric == Metric::InnerProduct;
                let x = mixture(n, d, 60, 0.5, unit, (n + d + k) as u32);
                let mut c = Vec::with_capacity(k * d);
                for j in 0..k {
                    let base = &x[(j / 2 % n) * d..(j / 2 % n + 1) * d];
                    let f = match (crowd, j % 4) {
                        (true, _) => 1.0 + 1e-3 * (j as f32 / k as f32),
                        (false, 0) | (false, 3) => 1.0,
                        (false, 1) => 1.0 - 1e-4,
                        _ => 1.0 + 1e-4,
                    };
                    if crowd {
                        c.extend(x[..d].iter().map(|v| v * f));
                    } else {
                        c.extend(base.iter().map(|v| v * f));
                    }
                }
                let xb = dev.upload_f32(&x).unwrap();
                let cb = dev.upload_f32(&c).unwrap();
                let mut cn = dev.zeros_f32(k).unwrap();
                dev.row_norms(&cb, k, d, &mut cn).unwrap();
                let perm: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 11) % n as u32).collect();
                let ids = dev.upload_u32(&perm).unwrap();
                for (indexed, rows) in [(true, 0..n), (false, 37..n - 5), (true, 100..n.min(1900))]
                {
                    let mut outs = Vec::new();
                    for screen in [Some(pick), None] {
                        dev.bk.screen_pick = screen;
                        let mut lab = dev.zeros_u32(n).unwrap();
                        let mut sco = dev.zeros_f32(n).unwrap();
                        // Twice: the second call reads the cached BF16 copy.
                        for _ in 0..2 {
                            dev.top1(
                                Mat {
                                    host: &x,
                                    dev: &xb,
                                    d,
                                },
                                Select {
                                    ids: indexed.then_some(&ids),
                                    rows: rows.clone(),
                                },
                                &cb,
                                Some(&cn),
                                k,
                                metric,
                                &mut lab,
                                &mut sco,
                            )
                            .unwrap();
                        }
                        let fallback = match (screen, dev.cnt.clone()) {
                            (Some(_), Some(c)) => dev.read_u32(&c, 0..1).unwrap()[0],
                            _ => 0,
                        };
                        let l = dev.read_u32(&lab, rows.clone()).unwrap();
                        let s: Vec<u32> = dev
                            .read_f32(&sco, rows.clone())
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                        outs.push((l, s, fallback));
                    }
                    let tag = format!("n={n} d={d} k={k} {metric:?} indexed={indexed}");
                    eprintln!(
                        "{tag}: {} of {} rows on the FP32 fallback",
                        outs[0].2,
                        rows.len()
                    );
                    if crowd {
                        assert!(outs[0].2 > 0, "{tag}: the crowd must reach the fallback");
                    }
                    assert_eq!(outs[0].0, outs[1].0, "{tag}: labels");
                    assert_eq!(outs[0].1, outs[1].1, "{tag}: scores");
                }
            }
        }
    }
}
