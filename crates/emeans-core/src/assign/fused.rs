//! The `cpu-fused` kernel: a packed GEMM with the argmax fused per tile.
//!
//! The centroids are packed once per call into panels of `NR` centroids
//! (`[panel][d][NR]`, zero-padded). Each rayon task takes `mc` rows, gathers
//! them straight into packed micro-panels of `MR` rows (`[panel][d][MR]`; the
//! gather and the packing are one copy), and walks the centroids in blocks of
//! `nc`: for each `kc` slice of the dimensions it runs the `MR x NR`
//! microkernel over every (row panel, centroid panel) pair of the block,
//! accumulating into an `mc x nc` tile, then folds the tile into the running
//! best per row. The `rows x k` score matrix never exists.
//!
//! Float order: every dot product is one sequential chain of fused
//! multiply-adds over the dimensions (the accumulators are reloaded between
//! `kc` slices, never summed in parts), so the result does not depend on the
//! tile sizes, the row's place in its panel, or the instruction set: the
//! AVX2+FMA microkernel and the portable `mul_add` one give the same bits.

use rayon::prelude::*;

use super::{Metric, Rows};

/// Rows per microkernel tile.
pub(crate) const MR: usize = 6;
/// Centroids per microkernel tile (two AVX registers).
pub(crate) const NR: usize = 16;

/// Tile sizes of the fused kernel (rows per task, centroids per block, dimensions per slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedTiles {
    pub mc: usize,
    pub nc: usize,
    pub kc: usize,
}

impl Default for FusedTiles {
    fn default() -> Self {
        FusedTiles {
            mc: 96,
            nc: 128,
            kc: 256,
        }
    }
}

impl FusedTiles {
    /// Defaults, overridden by `EMEANS_CPU_TILES=mc,nc,kc` (for tile sweeps).
    pub fn from_env() -> Self {
        let mut t = Self::default();
        if let Ok(s) = std::env::var("EMEANS_CPU_TILES") {
            let v: Vec<usize> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if let [mc, nc, kc] = v[..] {
                t = FusedTiles { mc, nc, kc };
            }
        }
        t.mc = t.mc.max(MR) / MR * MR;
        t.nc = t.nc.max(NR) / NR * NR;
        t.kc = t.kc.max(1);
        t
    }
}

/// Which microkernel this CPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Isa {
    /// Plain `f32::mul_add` loops (any CPU; bit-equal to the SIMD kernels).
    Portable,
    /// AVX2 + FMA (x86-64).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Avx2Fma,
}

impl Isa {
    pub(crate) fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        if std::env::var_os("EMEANS_CPU_PORTABLE").is_none()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return Isa::Avx2Fma;
        }
        Isa::Portable
    }
}

/// Pack `c` (`k x d`) into `[panel][d][NR]`, zero-padded to whole panels.
pub(crate) fn pack_centroids(c: &[f32], k: usize, d: usize, out: &mut Vec<f32>) {
    let panels = k.div_ceil(NR);
    out.clear();
    out.resize(panels * d * NR, 0.0);
    out.par_chunks_mut(d * NR).enumerate().for_each(|(p, dst)| {
        for j in 0..NR.min(k - p * NR) {
            let src = &c[(p * NR + j) * d..(p * NR + j + 1) * d];
            for (t, &v) in src.iter().enumerate() {
                dst[t * NR + j] = v;
            }
        }
    });
}

/// Per-thread scratch: packed rows, the score tile, the running best.
#[derive(Default)]
pub(crate) struct Scratch {
    a: Vec<f32>,
    tile: Vec<f32>,
    best: Vec<f32>,
    best_j: Vec<u32>,
}

/// Fused top-1 of every row of `rows` against packed centroids `bp` (k real ones).
#[allow(clippy::too_many_arguments)]
pub(crate) fn assign(
    rows: Rows<'_>,
    d: usize,
    bp: &[f32],
    k: usize,
    metric: Metric,
    cnorm: &[f32],
    labels: &mut [u32],
    scores: &mut [f32],
    tiles: FusedTiles,
    isa: Isa,
) {
    let mc = tiles.mc;
    labels
        .par_chunks_mut(mc)
        .zip(scores.par_chunks_mut(mc))
        .enumerate()
        .for_each_init(Scratch::default, |s, (t, (lab, sco))| {
            task(
                rows,
                t * mc,
                d,
                bp,
                k,
                metric,
                cnorm,
                lab,
                sco,
                tiles,
                isa,
                s,
            )
        });
}

#[allow(clippy::too_many_arguments)]
fn task(
    rows: Rows<'_>,
    r0: usize,
    d: usize,
    bp: &[f32],
    k: usize,
    metric: Metric,
    cnorm: &[f32],
    lab: &mut [u32],
    sco: &mut [f32],
    tiles: FusedTiles,
    isa: Isa,
    s: &mut Scratch,
) {
    let nr = lab.len();
    let mp = nr.div_ceil(MR);
    let FusedTiles { nc, kc, .. } = tiles;
    // Gather and pack the rows: [panel][d][MR], short panels padded with zero rows.
    s.a.clear();
    s.a.resize(mp * d * MR, 0.0);
    for (m, dst) in s.a.chunks_mut(d * MR).enumerate() {
        for i in 0..MR.min(nr - m * MR) {
            for (t, &v) in rows.row(r0 + m * MR + i, d).iter().enumerate() {
                dst[t * MR + i] = v;
            }
        }
    }
    s.tile.resize(mp * MR * nc, 0.0);
    s.best.clear();
    s.best.resize(
        nr,
        match metric {
            Metric::InnerProduct => f32::NEG_INFINITY,
            Metric::L2 => f32::INFINITY,
        },
    );
    s.best_j.clear();
    s.best_j.resize(nr, 0);
    let panels = k.div_ceil(NR);
    let block = nc / NR;
    for jc in (0..panels).step_by(block) {
        let jn = block.min(panels - jc);
        for k0 in (0..d).step_by(kc) {
            let kl = kc.min(d - k0);
            for jp in 0..jn {
                let b = &bp[(jc + jp) * d * NR + k0 * NR..];
                for m in 0..mp {
                    let a = &s.a[m * d * MR + k0 * MR..];
                    let c = &mut s.tile[m * MR * nc + jp * NR..];
                    microkernel(isa, a, b, kl, c, nc, k0 == 0);
                }
            }
        }
        // Fold the tile into the running best, columns in increasing order.
        let c0 = jc * NR;
        let cols = (jn * NR).min(k - c0);
        for i in 0..nr {
            let row = &s.tile[i * nc..i * nc + cols];
            let (j, v) = match metric {
                Metric::InnerProduct => super::argmax_lanes(row),
                Metric::L2 => super::argmin_l2_lanes(row, &cnorm[c0..c0 + cols]),
            };
            let better = match metric {
                Metric::InnerProduct => v > s.best[i],
                Metric::L2 => v < s.best[i],
            };
            if better {
                s.best[i] = v;
                s.best_j[i] = (c0 + j) as u32;
            }
        }
    }
    for i in 0..nr {
        lab[i] = s.best_j[i];
        sco[i] = super::finish_score(s.best[i], metric, || {
            rows.row(r0 + i, d).iter().map(|v| v * v).sum()
        });
    }
}

/// `c[i][j] (+)= sum_t a[t][i] * b[t][j]` for an `MR x NR` tile over `kc` dimensions.
#[inline]
fn microkernel(isa: Isa, a: &[f32], b: &[f32], kc: usize, c: &mut [f32], ldc: usize, init: bool) {
    assert!(a.len() >= kc * MR && b.len() >= kc * NR && c.len() >= (MR - 1) * ldc + NR);
    match isa {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 and FMA were detected; the assert above bounds every access.
        Isa::Avx2Fma => unsafe {
            microkernel_avx2(a.as_ptr(), b.as_ptr(), kc, c.as_mut_ptr(), ldc, init)
        },
        _ => microkernel_portable(a, b, kc, c, ldc, init),
    }
}

fn microkernel_portable(a: &[f32], b: &[f32], kc: usize, c: &mut [f32], ldc: usize, init: bool) {
    let mut acc = [[0f32; NR]; MR];
    if !init {
        for (i, r) in acc.iter_mut().enumerate() {
            r.copy_from_slice(&c[i * ldc..i * ldc + NR]);
        }
    }
    for t in 0..kc {
        let bt = &b[t * NR..t * NR + NR];
        for (i, r) in acc.iter_mut().enumerate() {
            let ai = a[t * MR + i];
            for (x, &bv) in r.iter_mut().zip(bt) {
                *x = ai.mul_add(bv, *x);
            }
        }
    }
    for (i, r) in acc.iter().enumerate() {
        c[i * ldc..i * ldc + NR].copy_from_slice(r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_avx2(
    a: *const f32,
    b: *const f32,
    kc: usize,
    c: *mut f32,
    ldc: usize,
    init: bool,
) {
    use std::arch::x86_64::*;
    // SAFETY (whole body): the caller checked a has kc*MR, b kc*NR floats and c
    // MR rows of NR floats at stride ldc.
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); 2]; MR];
        if !init {
            for (i, r) in acc.iter_mut().enumerate() {
                r[0] = _mm256_loadu_ps(c.add(i * ldc));
                r[1] = _mm256_loadu_ps(c.add(i * ldc + 8));
            }
        }
        let mut ap = a;
        let mut bp = b;
        for _ in 0..kc {
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            for (i, r) in acc.iter_mut().enumerate() {
                let ai = _mm256_broadcast_ss(&*ap.add(i));
                r[0] = _mm256_fmadd_ps(ai, b0, r[0]);
                r[1] = _mm256_fmadd_ps(ai, b1, r[1]);
            }
            ap = ap.add(MR);
            bp = bp.add(NR);
        }
        for (i, r) in acc.iter().enumerate() {
            _mm256_storeu_ps(c.add(i * ldc), r[0]);
            _mm256_storeu_ps(c.add(i * ldc + 8), r[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, d: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).max(1);
        (0..n * d)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    /// Labels and scores from every ISA this CPU has, for several tile sizes.
    fn run_all(x: &[f32], c: &[f32], d: usize, metric: Metric) -> Vec<(Vec<u32>, Vec<f32>)> {
        let (n, k) = (x.len() / d, c.len() / d);
        let cnorm = super::super::row_norms(c, d);
        let mut bp = Vec::new();
        pack_centroids(c, k, d, &mut bp);
        let mut isas = vec![Isa::Portable];
        if Isa::detect() != Isa::Portable {
            isas.push(Isa::detect());
        }
        let mut out = Vec::new();
        for isa in isas {
            for tiles in [
                FusedTiles::default(),
                FusedTiles {
                    mc: 6,
                    nc: 16,
                    kc: 7,
                },
                FusedTiles {
                    mc: 30,
                    nc: 48,
                    kc: 1000,
                },
            ] {
                let (mut l, mut s) = (vec![0u32; n], vec![0f32; n]);
                assign(
                    Rows::Dense(x),
                    d,
                    &bp,
                    k,
                    metric,
                    &cnorm,
                    &mut l,
                    &mut s,
                    tiles,
                    isa,
                );
                out.push((l, s));
            }
        }
        out
    }

    #[test]
    fn same_bits_for_every_isa_and_tile() {
        for &(n, d, k) in &[(97, 13, 37), (200, 768, 70), (31, 5, 1), (64, 64, 16)] {
            let x = data(n, d, 1);
            let c = data(k, d, 2);
            for metric in [Metric::InnerProduct, Metric::L2] {
                let all = run_all(&x, &c, d, metric);
                for r in &all[1..] {
                    assert_eq!(r.0, all[0].0, "labels n={n} d={d} k={k} {metric:?}");
                    let same =
                        r.1.iter()
                            .zip(&all[0].1)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(same, "scores n={n} d={d} k={k} {metric:?}");
                }
            }
        }
    }

    #[test]
    fn dot_is_one_sequential_fma_chain() {
        let (n, d, k) = (13, 300, 21);
        let x = data(n, d, 3);
        let c = data(k, d, 4);
        let all = run_all(&x, &c, d, Metric::InnerProduct);
        for i in 0..n {
            let mut bj = 0;
            let mut b = f32::NEG_INFINITY;
            for j in 0..k {
                let mut acc = 0f32;
                for t in 0..d {
                    acc = x[i * d + t].mul_add(c[j * d + t], acc);
                }
                if acc > b {
                    b = acc;
                    bj = j as u32;
                }
            }
            assert_eq!(all[0].0[i], bj);
            assert_eq!(all[0].1[i].to_bits(), b.to_bits());
        }
    }

    #[test]
    fn ties_keep_the_lowest_id() {
        let d = 8;
        let x = vec![1f32; 3 * d];
        let mut c = vec![0f32; 40 * d];
        for j in [5usize, 17, 33] {
            c[j * d..(j + 1) * d].fill(1.0);
        }
        for r in run_all(&x, &c, d, Metric::InnerProduct) {
            assert_eq!(r.0, vec![5, 5, 5]);
        }
        for r in run_all(&x, &c, d, Metric::L2) {
            assert_eq!(r.0, vec![5, 5, 5]);
        }
    }
}
