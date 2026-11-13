//! Nearest-centroid assignment: the `Assigner` trait and the CPU kernels.
//!
//! An assigner returns, for each row, the id of the nearest centroid and a
//! score: the squared L2 distance (`Metric::L2`, smallest wins) or the inner
//! product (`Metric::InnerProduct`, largest wins). Ties keep the lowest id.

use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod fused;
pub use fused::FusedTiles;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Squared Euclidean distance; the nearest centroid has the smallest score.
    L2,
    /// Inner product (cosine for unit vectors); the nearest has the largest score.
    InnerProduct,
}

/// The rows to assign: a dense `n x d` block, or row ids into a larger matrix.
#[derive(Clone, Copy)]
pub enum Rows<'a> {
    Dense(&'a [f32]),
    Indexed { data: &'a [f32], ids: &'a [u32] },
}

impl Rows<'_> {
    pub fn len(&self, d: usize) -> usize {
        match self {
            Rows::Dense(x) => x.len() / d,
            Rows::Indexed { ids, .. } => ids.len(),
        }
    }

    pub fn is_empty(&self, d: usize) -> bool {
        self.len(d) == 0
    }

    /// Row `i` of this selection.
    #[inline]
    pub fn row(&self, i: usize, d: usize) -> &[f32] {
        match self {
            Rows::Dense(x) => &x[i * d..(i + 1) * d],
            Rows::Indexed { data, ids } => {
                let r = ids[i] as usize;
                &data[r * d..(r + 1) * d]
            }
        }
    }

    /// Rows `[begin, end)` of this selection.
    pub fn slice(&self, begin: usize, end: usize, d: usize) -> Rows<'_> {
        match self {
            Rows::Dense(x) => Rows::Dense(&x[begin * d..end * d]),
            Rows::Indexed { data, ids } => Rows::Indexed {
                data,
                ids: &ids[begin..end],
            },
        }
    }
}

/// An immutable `n x d` data set an engine may keep alive and read in place
/// (`Assigner::bind_shared`): `Arc::new(vec)` coerces to it without a copy.
pub type SharedData = std::sync::Arc<dyn AsRef<[f32]> + Send + Sync>;

/// A nearest-centroid search backend.
pub trait Assigner: Send {
    /// Short name for reports, for example `cpu-accelerate` or `metal-fused`.
    fn name(&self) -> String;

    /// Keep `data` (`n x d`) on the device until `unbind_data`, so `Rows::Indexed`
    /// calls on it send only row ids. The backend copies `data`; the caller must
    /// not change or free it while bound (`Rows::Indexed` calls match the copy
    /// by pointer and length, so they would see the old copy). Binding again,
    /// also with the same pointer and length, always takes a fresh copy.
    fn bind_data(&mut self, _data: &[f32], _d: usize) -> Result<()> {
        Ok(())
    }

    /// `bind_data` for data the backend may keep and read in place: the backend
    /// holds `data` (an `Arc`, so it stays alive and unchanged) until
    /// `unbind_data` or drop. The Metal engine reads it without a copy; the
    /// default copies through `bind_data`.
    fn bind_shared(&mut self, data: SharedData, d: usize) -> Result<()> {
        self.bind_data((*data).as_ref(), d)
    }

    /// Release data kept by `bind_data`.
    fn unbind_data(&mut self) {}

    /// Hand this backend's engine (layer 1, [`crate::core::Device`]) to `f`, so
    /// training runs the device loop on it. None: no engine, the caller falls
    /// back to the host loop.
    fn with_engine(
        &mut self,
        _f: &mut dyn FnMut(&mut dyn crate::core::Engine) -> Result<()>,
    ) -> Option<Result<()>> {
        None
    }

    /// Write the nearest centroid of every row to `labels` and its score to `scores`.
    ///
    /// `centroids` is `k x d`, row-major. `labels` and `scores` have one slot per row.
    fn assign(
        &mut self,
        rows: Rows<'_>,
        d: usize,
        centroids: &[f32],
        metric: Metric,
        labels: &mut [u32],
        scores: &mut [f32],
    ) -> Result<()>;
}

/// Validate the shapes of an [`Assigner::assign`] call; returns `k`. For backend crates.
#[doc(hidden)]
pub fn check_shapes(
    rows: &Rows<'_>,
    d: usize,
    centroids: &[f32],
    labels: &[u32],
    scores: &[f32],
) -> Result<usize> {
    ensure!(d > 0, "d must be positive");
    ensure!(
        !centroids.is_empty() && centroids.len().is_multiple_of(d),
        "centroids must be k x d"
    );
    if let Rows::Dense(x) = rows {
        ensure!(x.len().is_multiple_of(d), "rows must be n x d");
    }
    if let Rows::Indexed { data, ids } = rows {
        let n = data.len() / d;
        ensure!(ids.iter().all(|&i| (i as usize) < n), "row id out of range");
    }
    let n = rows.len(d);
    ensure!(
        labels.len() == n && scores.len() == n,
        "output length must equal row count"
    );
    ensure!(centroids.len() / d <= u32::MAX as usize, "k too large");
    Ok(centroids.len() / d)
}

/// Squared norms of the rows of `m` (`r x d`).
pub fn row_norms(m: &[f32], d: usize) -> Vec<f32> {
    m.par_chunks(d)
        .with_min_len((16384 / d.max(1)).max(1))
        .map(|r| r.iter().map(|v| v * v).sum())
        .collect()
}

/// The rayon pool of the CPU backend's small per-batch ops and of the
/// `cpu-fused` top-1: one thread per physical core on a CPU with SMT (Linux;
/// 32 of 64 threads on the 2x Xeon box), else rayon's global pool.
/// `RAYON_NUM_THREADS` overrides it. Hyperthreads share one core's FMA units
/// and caches, and every extra thread is one more to wake for each small op;
/// see `.okf/results/cpu-kernels.md`.
pub(crate) fn cpu_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        if std::env::var_os("RAYON_NUM_THREADS").is_some() {
            return None;
        }
        let cores = physical_cores()?;
        let logical = std::thread::available_parallelism().ok()?.get();
        if cores >= logical {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(cores)
            .thread_name(|i| format!("emeans-cpu-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// Run `f` on the CPU backend's pool (see [`cpu_pool`]).
pub(crate) fn on_cpu_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match cpu_pool() {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// Physical cores: distinct `core_cpus_list` entries in sysfs (Linux only).
fn physical_cores() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let mut cores = std::collections::HashSet::new();
        for e in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
            let e = e.ok()?;
            let name = e.file_name();
            let name = name.to_string_lossy();
            if !(name.starts_with("cpu")
                && name[3..].chars().all(|c| c.is_ascii_digit())
                && name.len() > 3)
            {
                continue;
            }
            let topo = e.path().join("topology");
            let list = std::fs::read_to_string(topo.join("core_cpus_list"))
                .or_else(|_| std::fs::read_to_string(topo.join("thread_siblings_list")));
            if let Ok(list) = list {
                cores.insert(list.trim().to_string());
            }
        }
        (!cores.is_empty()).then_some(cores.len())
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Which CPU kernel computes the row-centroid dot products.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuKernel {
    /// Cache-blocked 4x4 register tiles with NEON FMA (portable fallback elsewhere), rayon.
    Blocked,
    /// The `gemm` crate per row block (the call candle's CPU matmul makes), then a
    /// rayon argmax. Not on macOS (use `Accelerate` there).
    #[serde(alias = "candle_gemm")]
    Gemm,
    /// Apple Accelerate `cblas_sgemm` (AMX) per row block, then a rayon argmax. macOS only.
    Accelerate,
    /// Packed GEMM with the argmax fused per tile (AVX2+FMA microkernel on x86-64,
    /// portable `mul_add` elsewhere, same bits), rayon over row tiles.
    Fused,
}

impl CpuKernel {
    /// Backend name: `cpu-blocked`, `cpu-gemm`, `cpu-accelerate` or `cpu-fused`.
    pub fn name(self) -> &'static str {
        match self {
            CpuKernel::Blocked => "cpu-blocked",
            CpuKernel::Gemm => "cpu-gemm",
            CpuKernel::Accelerate => "cpu-accelerate",
            CpuKernel::Fused => "cpu-fused",
        }
    }

    /// The fastest kernel measured on this machine (see `.okf/design/emeans-algorithm.md`).
    /// On WebAssembly: `Gemm` (simd128; 12x faster than `Fused`, whose `mul_add` is a
    /// software call there, see `.okf/design/web.md`); `Fused` gives native bits.
    pub fn fastest() -> CpuKernel {
        if cfg!(target_os = "macos") {
            CpuKernel::Accelerate
        } else if cfg!(target_family = "wasm") {
            CpuKernel::Gemm
        } else if cfg!(target_arch = "x86_64") {
            CpuKernel::Fused
        } else {
            CpuKernel::Blocked
        }
    }
}

/// CPU assigner. Dot products come from `kernel`; the argmin/argmax runs on rayon.
pub struct CpuAssigner {
    pub kernel: CpuKernel,
    gather: Vec<f32>,
    dots: Vec<f32>,
    packed: Vec<f32>,
    tiles: FusedTiles,
    isa: fused::Isa,
}

impl Default for CpuAssigner {
    fn default() -> Self {
        Self::new(CpuKernel::fastest())
    }
}

impl CpuAssigner {
    pub fn new(kernel: CpuKernel) -> Self {
        Self {
            kernel,
            gather: Vec::new(),
            dots: Vec::new(),
            packed: Vec::new(),
            tiles: FusedTiles::from_env(),
            isa: fused::Isa::detect(),
        }
    }

    /// Rows per GEMM block: about 16 M floats of dot products (one 16k-row
    /// batch at k = 1024 is one GEMM call).
    fn block_rows(k: usize) -> usize {
        ((16 << 20) / k).clamp(256, 16384)
    }
}

impl Assigner for CpuAssigner {
    fn name(&self) -> String {
        self.kernel.name().into()
    }

    fn with_engine(
        &mut self,
        f: &mut dyn FnMut(&mut dyn crate::core::Engine) -> Result<()>,
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
        // Only the FMA-bound fused kernel runs on physical cores; the scalar
        // blocked kernel and the gemm crate gain from hyperthreads (blocked top-1
        // at 200k x 1024 on a 2x Xeon E5-2683 v4: 265-267 ms on 64 threads, 472-473 ms on 32).
        if self.kernel == CpuKernel::Fused {
            on_cpu_pool(|| self.assign_here(rows, d, centroids, metric, labels, scores))
        } else {
            self.assign_here(rows, d, centroids, metric, labels, scores)
        }
    }
}

impl CpuAssigner {
    /// [`Assigner::assign`] on the calling thread's rayon pool.
    fn assign_here(
        &mut self,
        rows: Rows<'_>,
        d: usize,
        centroids: &[f32],
        metric: Metric,
        labels: &mut [u32],
        scores: &mut [f32],
    ) -> Result<()> {
        let k = check_shapes(&rows, d, centroids, labels, scores)?;
        let n = rows.len(d);
        if n == 0 {
            return Ok(());
        }
        let cnorm = match metric {
            Metric::L2 => row_norms(centroids, d),
            Metric::InnerProduct => Vec::new(),
        };
        if self.kernel == CpuKernel::Blocked {
            blocked_assign(rows, d, centroids, k, metric, &cnorm, labels, scores);
            return Ok(());
        }
        if self.kernel == CpuKernel::Fused {
            fused::pack_centroids(centroids, k, d, &mut self.packed);
            fused::assign(
                rows,
                d,
                &self.packed,
                k,
                metric,
                &cnorm,
                labels,
                scores,
                self.tiles,
                self.isa,
            );
            return Ok(());
        }
        let block = Self::block_rows(k);
        for b0 in (0..n).step_by(block) {
            let b1 = (b0 + block).min(n);
            let r = b1 - b0;
            let sub = rows.slice(b0, b1, d);
            let x: &[f32] = match sub {
                Rows::Dense(x) => x,
                Rows::Indexed { .. } => {
                    self.gather.resize(r * d, 0.0);
                    self.gather
                        .par_chunks_mut(d)
                        .enumerate()
                        .for_each(|(i, dst)| dst.copy_from_slice(sub.row(i, d)));
                    &self.gather
                }
            };
            self.dots.resize(r * k, 0.0);
            match self.kernel {
                CpuKernel::Accelerate => sgemm_accelerate(x, centroids, r, k, d, &mut self.dots)?,
                CpuKernel::Gemm => sgemm_gemm(x, centroids, r, k, d, &mut self.dots)?,
                CpuKernel::Blocked | CpuKernel::Fused => unreachable!(),
            }
            let dots = &self.dots;
            labels[b0..b1]
                .par_iter_mut()
                .zip(scores[b0..b1].par_iter_mut())
                .enumerate()
                .for_each(|(i, (l, s))| {
                    let row = &dots[i * k..(i + 1) * k];
                    let (j, v) = select(row, metric, &cnorm);
                    *l = j as u32;
                    *s = finish_score(v, metric, || sub.row(i, d).iter().map(|a| a * a).sum());
                });
        }
        Ok(())
    }
}

/// Best centroid from one row of dot products: max dot (IP) or min ||c||^2 - 2 dot (L2).
#[inline]
fn select(dots: &[f32], metric: Metric, cnorm: &[f32]) -> (usize, f32) {
    match metric {
        Metric::InnerProduct => argmax_lanes(dots),
        Metric::L2 => argmin_l2_lanes(dots, cnorm),
    }
}

/// Lanes of the vectorizable argmax/argmin: lane `t` sees columns `t mod LANES`.
const LANES: usize = 8;

/// Index and value of the largest entry; ties keep the lowest index (the
/// result of a sequential strict `>` scan), in branch-free lanes that vectorize.
#[inline]
pub(crate) fn argmax_lanes(v: &[f32]) -> (usize, f32) {
    let full = v.len() / LANES * LANES;
    let mut best = [f32::NEG_INFINITY; LANES];
    let mut idx = [0u32; LANES];
    for (c, chunk) in v[..full].as_chunks::<LANES>().0.iter().enumerate() {
        let base = (c * LANES) as u32;
        for t in 0..LANES {
            let m = chunk[t] > best[t];
            best[t] = if m { chunk[t] } else { best[t] };
            idx[t] = if m { base + t as u32 } else { idx[t] };
        }
    }
    let (mut b, mut j) = reduce_lanes(&best, &idx, |x, y| x > y);
    for (t, &x) in v.iter().enumerate().skip(full) {
        if x > b {
            b = x;
            j = t as u32;
        }
    }
    (j as usize, b)
}

/// Index and value of the smallest `cnorm[j] - 2 dots[j]`; ties keep the lowest index.
#[inline]
pub(crate) fn argmin_l2_lanes(dots: &[f32], cnorm: &[f32]) -> (usize, f32) {
    let n = dots.len();
    let full = n / LANES * LANES;
    let mut best = [f32::INFINITY; LANES];
    let mut idx = [0u32; LANES];
    for (c, (dc, nc)) in dots[..full]
        .as_chunks::<LANES>()
        .0
        .iter()
        .zip(cnorm[..full].as_chunks::<LANES>().0)
        .enumerate()
    {
        let base = (c * LANES) as u32;
        for t in 0..LANES {
            let s = nc[t] - 2.0 * dc[t];
            let m = s < best[t];
            best[t] = if m { s } else { best[t] };
            idx[t] = if m { base + t as u32 } else { idx[t] };
        }
    }
    let (mut b, mut j) = reduce_lanes(&best, &idx, |x, y| x < y);
    for t in full..n {
        let s = cnorm[t] - 2.0 * dots[t];
        if s < b {
            b = s;
            j = t as u32;
        }
    }
    (j as usize, b)
}

/// The lane winners in column order: a better value wins, an equal one keeps the lower index.
#[inline]
fn reduce_lanes(
    best: &[f32; LANES],
    idx: &[u32; LANES],
    better: impl Fn(f32, f32) -> bool,
) -> (f32, u32) {
    let (mut b, mut j) = (best[0], idx[0]);
    for t in 1..LANES {
        if better(best[t], b) || (best[t] == b && idx[t] < j) {
            b = best[t];
            j = idx[t];
        }
    }
    (b, j)
}

/// Turn a selection value into the reported score.
#[inline]
fn finish_score(v: f32, metric: Metric, xnorm: impl FnOnce() -> f32) -> f32 {
    match metric {
        Metric::InnerProduct => v,
        Metric::L2 => (xnorm() + v).max(0.0),
    }
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// dots (r x k) = x (r x d) * centroids (k x d)^T with Accelerate.
fn sgemm_accelerate(
    x: &[f32],
    centroids: &[f32],
    r: usize,
    k: usize,
    d: usize,
    dots: &mut [f32],
) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        ensure!(r <= i32::MAX as usize && k <= i32::MAX as usize && d <= i32::MAX as usize);
        const ROW_MAJOR: i32 = 101;
        const NO_TRANS: i32 = 111;
        const TRANS: i32 = 112;
        // SAFETY: the slices hold r*d, k*d and r*k floats, matching the dimensions.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                r as i32,
                k as i32,
                d as i32,
                1.0,
                x.as_ptr(),
                d as i32,
                centroids.as_ptr(),
                d as i32,
                0.0,
                dots.as_mut_ptr(),
                k as i32,
            );
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (x, centroids, r, k, d, dots);
        anyhow::bail!("the Accelerate kernel needs macOS")
    }
}

/// dots (r x k) = x (r x d) * centroids (k x d)^T with the `gemm` crate, called
/// the way candle-core 0.11.1's CPU matmul calls it for `x.matmul(&c.t())`
/// (same strides, alpha/beta and rayon parallelism), so results match it bit
/// for bit.
#[cfg(not(target_os = "macos"))]
fn sgemm_gemm(
    x: &[f32],
    centroids: &[f32],
    r: usize,
    k: usize,
    d: usize,
    dots: &mut [f32],
) -> Result<()> {
    ensure!(x.len() == r * d && centroids.len() == k * d && dots.len() == r * k);
    let threads = rayon::current_num_threads();
    let parallelism = if threads > 1 {
        gemm::Parallelism::Rayon(threads)
    } else {
        gemm::Parallelism::None
    };
    // SAFETY: the pointers cover r x d, k x d and r x k f32s (checked above);
    // `dst` does not alias the inputs.
    unsafe {
        gemm::gemm(
            /* m */ r,
            /* n */ k,
            /* k */ d,
            /* dst */ dots.as_mut_ptr(),
            /* dst_cs */ 1,
            /* dst_rs */ k as isize,
            /* read_dst */ false,
            /* lhs */ x.as_ptr(),
            /* lhs_cs */ 1,
            /* lhs_rs */ d as isize,
            /* rhs: c^T, a d x k view of the k x d centroids */ centroids.as_ptr(),
            /* rhs_cs */ d as isize,
            /* rhs_rs */ 1,
            /* alpha */ 0.0,
            /* beta */ 1.0,
            false,
            false,
            false,
            parallelism,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn sgemm_gemm(
    x: &[f32],
    centroids: &[f32],
    r: usize,
    k: usize,
    d: usize,
    dots: &mut [f32],
) -> Result<()> {
    let _ = (x, centroids, r, k, d, dots);
    anyhow::bail!("the gemm kernel is not built on macOS; use cpu-accelerate")
}

/// Rows per rayon task and centroids per cache tile in the blocked kernel.
const BLOCK_ROWS: usize = 32;
const BLOCK_CENTROIDS: usize = 64;

#[allow(clippy::too_many_arguments)]
fn blocked_assign(
    rows: Rows<'_>,
    d: usize,
    centroids: &[f32],
    k: usize,
    metric: Metric,
    cnorm: &[f32],
    labels: &mut [u32],
    scores: &mut [f32],
) {
    labels
        .par_chunks_mut(BLOCK_ROWS)
        .zip(scores.par_chunks_mut(BLOCK_ROWS))
        .enumerate()
        .for_each(|(blk, (lab, sco))| {
            let r0 = blk * BLOCK_ROWS;
            let nr = lab.len();
            let xs: Vec<&[f32]> = (0..nr).map(|i| rows.row(r0 + i, d)).collect();
            let mut best = vec![
                match metric {
                    Metric::L2 => f32::INFINITY,
                    Metric::InnerProduct => f32::NEG_INFINITY,
                };
                nr
            ];
            let mut best_j = vec![0u32; nr];
            let mut tile = [0f32; BLOCK_ROWS * BLOCK_CENTROIDS];
            for c0 in (0..k).step_by(BLOCK_CENTROIDS) {
                let nc = BLOCK_CENTROIDS.min(k - c0);
                let cs: Vec<&[f32]> = (0..nc)
                    .map(|j| &centroids[(c0 + j) * d..(c0 + j + 1) * d])
                    .collect();
                let mut i = 0;
                while i < nr {
                    let ri = (nr - i).min(4);
                    let mut j = 0;
                    while j < nc {
                        let cj = (nc - j).min(4);
                        let out = dot_tile(&xs[i..i + ri], &cs[j..j + cj], d);
                        for a in 0..ri {
                            for b in 0..cj {
                                tile[(i + a) * BLOCK_CENTROIDS + j + b] = out[a][b];
                            }
                        }
                        j += 4;
                    }
                    i += 4;
                }
                for a in 0..nr {
                    for b in 0..nc {
                        let v = tile[a * BLOCK_CENTROIDS + b];
                        let s = match metric {
                            Metric::InnerProduct => v,
                            Metric::L2 => cnorm[c0 + b] - 2.0 * v,
                        };
                        let better = match metric {
                            Metric::InnerProduct => s > best[a],
                            Metric::L2 => s < best[a],
                        };
                        if better {
                            best[a] = s;
                            best_j[a] = (c0 + b) as u32;
                        }
                    }
                }
            }
            for a in 0..nr {
                lab[a] = best_j[a];
                sco[a] = finish_score(best[a], metric, || xs[a].iter().map(|v| v * v).sum());
            }
        });
}

/// Dot products of up to 4 rows with up to 4 centroids.
#[inline]
fn dot_tile(xs: &[&[f32]], cs: &[&[f32]], d: usize) -> [[f32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // Pad short tiles by repeating the last row or centroid, so every dot
        // product uses the same summation order wherever its row falls.
        let pad = |v: &[&'_ [f32]]| -> [*const f32; 4] {
            let mut p = [v[v.len() - 1].as_ptr(); 4];
            for (dst, s) in p.iter_mut().zip(v) {
                *dst = s.as_ptr();
            }
            p
        };
        // SAFETY: NEON is always present on aarch64; every pointer has d floats.
        unsafe { dot_tile_neon(pad(xs), pad(cs), d) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = d; // the slices carry their length off aarch64
        let mut out = [[0f32; 4]; 4];
        for (a, x) in xs.iter().enumerate() {
            for (b, c) in cs.iter().enumerate() {
                out[a][b] = x.iter().zip(c.iter()).map(|(p, q)| p * q).sum();
            }
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_tile_neon(xp: [*const f32; 4], cp: [*const f32; 4], d: usize) -> [[f32; 4]; 4] {
    use std::arch::aarch64::*;
    let z = vdupq_n_f32(0.0);
    let mut acc = [[z; 4]; 4];
    let full = d / 4 * 4;
    let mut o = 0;
    while o < full {
        // SAFETY: o + 4 <= d for every row and centroid.
        unsafe {
            let x = [
                vld1q_f32(xp[0].add(o)),
                vld1q_f32(xp[1].add(o)),
                vld1q_f32(xp[2].add(o)),
                vld1q_f32(xp[3].add(o)),
            ];
            let c = [
                vld1q_f32(cp[0].add(o)),
                vld1q_f32(cp[1].add(o)),
                vld1q_f32(cp[2].add(o)),
                vld1q_f32(cp[3].add(o)),
            ];
            for a in 0..4 {
                for b in 0..4 {
                    acc[a][b] = vfmaq_f32(acc[a][b], x[a], c[b]);
                }
            }
        }
        o += 4;
    }
    let mut out = [[0f32; 4]; 4];
    for a in 0..4 {
        for b in 0..4 {
            let mut s = vaddvq_f32(acc[a][b]);
            for t in full..d {
                // SAFETY: t < d.
                s += unsafe { *xp[a].add(t) * *cp[b].add(t) };
            }
            out[a][b] = s;
        }
    }
    out
}

/// Assign every row of `rows` against `centroids` in chunks of `chunk` rows.
///
/// Returns (labels, scores). Convenience for callers that do not train.
pub fn assign_all(
    assigner: &mut dyn Assigner,
    rows: Rows<'_>,
    d: usize,
    centroids: &[f32],
    metric: Metric,
    chunk: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let n = rows.len(d);
    let mut labels = vec![0u32; n];
    let mut scores = vec![0f32; n];
    let chunk = chunk.max(1);
    for b in (0..n).step_by(chunk) {
        let e = (b + chunk).min(n);
        assigner.assign(
            rows.slice(b, e, d),
            d,
            centroids,
            metric,
            &mut labels[b..e],
            &mut scores[b..e],
        )?;
    }
    Ok((labels, scores))
}

/// `out` (r x k) = `x` (r x d) times `y` (k x d) transposed, on the CPU:
/// Accelerate on macOS, the `gemm` crate elsewhere.
pub fn gemm_nt(x: &[f32], y: &[f32], d: usize, out: &mut [f32]) -> Result<()> {
    ensure!(d > 0 && x.len().is_multiple_of(d) && y.len().is_multiple_of(d));
    let (r, k) = (x.len() / d, y.len() / d);
    ensure!(out.len() == r * k, "output must be r x k");
    if r == 0 || k == 0 {
        return Ok(());
    }
    if cfg!(target_os = "macos") {
        sgemm_accelerate(x, y, r, k, d, out)
    } else {
        sgemm_gemm(x, y, r, k, d, out)
    }
}

/// One assigner shared by several models, so a sweep over many models compiles
/// the Metal shaders once. Models that share it must not train at the same time.
#[derive(Clone)]
pub struct SharedAssigner(std::sync::Arc<std::sync::Mutex<Box<dyn Assigner>>>);

impl SharedAssigner {
    pub fn new(inner: Box<dyn Assigner>) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(inner)))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Box<dyn Assigner>>> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("shared assigner lock poisoned"))
    }
}

impl Assigner for SharedAssigner {
    fn name(&self) -> String {
        self.lock().map(|a| a.name()).unwrap_or_default()
    }

    fn bind_data(&mut self, data: &[f32], d: usize) -> Result<()> {
        self.lock()?.bind_data(data, d)
    }

    fn bind_shared(&mut self, data: SharedData, d: usize) -> Result<()> {
        self.lock()?.bind_shared(data, d)
    }

    fn unbind_data(&mut self) {
        if let Ok(mut a) = self.lock() {
            a.unbind_data();
        }
    }

    fn with_engine(
        &mut self,
        f: &mut dyn FnMut(&mut dyn crate::core::Engine) -> Result<()>,
    ) -> Option<Result<()>> {
        match self.lock() {
            Ok(mut a) => a.with_engine(f),
            Err(e) => Some(Err(e)),
        }
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
        self.lock()?
            .assign(rows, d, centroids, metric, labels, scores)
    }
}
