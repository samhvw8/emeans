//! e-means Metal backend (macOS): [`MetalCore`], the layer-1 engine of
//! `emeans-core`, with runtime-compiled Metal shaders.
//!
//! - Top-1 assignment: the fused kernel computes the dot-product tile with
//!   `simdgroup_float8x8` and keeps a running argmin/argmax per row in
//!   registers, so the `n x k` matrix never exists. Training runs the device
//!   loop of `emeans-core` on it.
//! - Top-k search ([`MetalKnn`], `Device::topk`): a fused GEMM + top-k kernel
//!   ([`KNN_SOURCE`]) for `k <= KNN_KMAX`.
//! - The exact ops (scatter-add, EMA, norms, splits) compile without fast math,
//!   so they round like the CPU code (see `emeans_core::core::Device`).
//!
//! On other targets this crate is empty.
#![cfg(target_os = "macos")]

mod engine;

use std::sync::Mutex;

use anyhow::Result;
use emeans_core::core::Device;

pub use engine::{FUSED_TILES, MetalCore, TOP1_SOURCE as FUSED_SOURCE};

/// Neighbours a thread keeps per row in the top-k kernel (the largest `k` it accepts).
pub use emeans_core::knn::GPU_KMAX as KNN_KMAX;

/// Exact top-k inner-product search: the GEMM tiling of `FUSED_SOURCE`
/// (64 query rows x 32 data rows per tile, `simdgroup_float8x8`), but each
/// tile of dot products goes to threadgroup memory and two threads per query
/// row keep its best `k` (unsorted, with the slot of the current worst).
/// The two lists merge at the end and the row is written sorted, best first.
pub const KNN_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct KnnParams {
    uint rows;      // query rows in this dispatch
    uint n;         // data rows
    uint d;
    uint kk;        // neighbours per row, <= KMAX
    uint self_off;  // query row r is data row r + self_off (when exclude == 1)
    uint exclude;   // 1: skip the data row equal to the query row
};

constant constexpr uint KMAX = 16;

// Is (a, ja) better than (b, jb)? Larger score first, then lower id.
inline bool kbetter(float a, uint ja, float b, uint jb) {
    return a > b || (a == b && ja < jb);
}

inline uint worst_slot(thread const float* s, thread const uint* j, uint kk) {
    uint w = 0;
    for (uint t = 1; t < kk; ++t)
        if (kbetter(s[w], j[w], s[t], j[t])) w = t;
    return w;
}

template <uint BM, uint BN, uint BK, uint WM, uint WN, bool VEC>
inline void knn_topk(
    device const float* x, device const float* c, device uint* out,
    constant KnnParams& p, uint tg, ushort tid, ushort sg, ushort lane,
    threadgroup float* Xs, threadgroup float* Cs, threadgroup float* T, threadgroup uint* rsrc)
{
    constexpr uint NT = WM * WN * 32;
    static_assert(NT == 2 * BM, "two threads per query row");
    constexpr uint LDK = BK + 4;
    constexpr uint TMS = 8 * WM;
    constexpr uint TNS = 8 * WN;
    constexpr uint TM = BM / TMS;
    constexpr uint TN = BN / TNS;
    constexpr uint HALF = BN / 2;

    const uint row0 = tg * BM;
    const uint d = p.d, n = p.n, kk = p.kk;

    for (uint r = tid; r < BM; r += NT) {
        const uint gr = row0 + r;
        rsrc[r] = gr < p.rows ? gr : 0xffffffffu;
    }

    const ushort tm = 8 * (sg / WN);
    const ushort tn = 8 * (sg % WN);
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + (lane / 2) % 4;
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint a_off = (tm + fm) * LDK + fn;
    const uint b_off = (tn + fn) * LDK + fm;

    // This thread's query row and half of each tile's columns.
    const uint lr = tid >> 1;
    const uint h = tid & 1;
    const uint grow = row0 + lr;
    const uint self_j = (p.exclude != 0 && grow < p.rows) ? grow + p.self_off : 0xffffffffu;
    float bs[KMAX];
    uint bj[KMAX];
    for (uint t = 0; t < KMAX; ++t) { bs[t] = -INFINITY; bj[t] = 0xffffffffu; }
    uint wslot = 0;

    for (uint n0 = 0; n0 < n; n0 += BN) {
        simdgroup_float8x8 acc[TM * TN];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TM * TN; ++t)
            acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0; k0 < d; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (VEC) {
                for (uint e = tid; e < BM * BK / 4; e += NT) {
                    const uint r = e / (BK / 4);
                    const uint c4 = (e % (BK / 4)) * 4;
                    const uint gc = k0 + c4;
                    const uint sr = rsrc[r];
                    float4 v = 0.0f;
                    if (sr != 0xffffffffu && gc < d) v = *(device const float4*)(x + (ulong)sr * d + gc);
                    *(threadgroup float4*)(Xs + r * LDK + c4) = v;
                }
                for (uint e = tid; e < BN * BK / 4; e += NT) {
                    const uint r = e / (BK / 4);
                    const uint c4 = (e % (BK / 4)) * 4;
                    const uint gc = k0 + c4;
                    const uint gj = n0 + r;
                    float4 v = 0.0f;
                    if (gj < n && gc < d) v = *(device const float4*)(c + (ulong)gj * d + gc);
                    *(threadgroup float4*)(Cs + r * LDK + c4) = v;
                }
            } else {
                for (uint e = tid; e < BM * BK; e += NT) {
                    const uint r = e / BK, cc = e % BK, gc = k0 + cc;
                    const uint sr = rsrc[r];
                    Xs[r * LDK + cc] = (sr != 0xffffffffu && gc < d) ? x[(ulong)sr * d + gc] : 0.0f;
                }
                for (uint e = tid; e < BN * BK; e += NT) {
                    const uint r = e / BK, cc = e % BK, gc = k0 + cc, gj = n0 + r;
                    Cs[r * LDK + cc] = (gj < n && gc < d) ? c[(ulong)gj * d + gc] : 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup const float* As = Xs + a_off;
            threadgroup const float* Bs = Cs + b_off;
            #pragma clang loop unroll(full)
            for (uint kq = 0; kq < BK; kq += 8) {
                simdgroup_float8x8 a[TM];
                simdgroup_float8x8 b[TN];
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < TM; ++i) {
                    a[i].thread_elements()[0] = As[i * TMS * LDK + kq];
                    a[i].thread_elements()[1] = As[i * TMS * LDK + kq + 1];
                }
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; ++j) {
                    b[j].thread_elements()[0] = Bs[j * TNS * LDK + kq];
                    b[j].thread_elements()[1] = Bs[j * TNS * LDK + LDK + kq];
                }
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < TM; ++i) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < TN; ++j) {
                        const uint js = (i % 2) ? (TN - 1 - j) : j;
                        simdgroup_multiply_accumulate(acc[i * TN + js], a[i], b[js], acc[i * TN + js]);
                    }
                }
            }
        }
        // The dot-product tile, row-major BM x BN. The next tile's first
        // barrier keeps it intact until every thread has scanned it.
        #pragma clang loop unroll(full)
        for (uint i = 0; i < TM; ++i) {
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; ++j) {
                const thread auto& e = acc[i * TN + j].thread_elements();
                const uint r = tm + fm + i * TMS;
                const uint cc = tn + j * TNS + fn;
                T[r * BN + cc] = e[0];
                T[r * BN + cc + 1] = e[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (grow < p.rows) {
            for (uint q = 0; q < HALF; ++q) {
                const uint col = h * HALF + q;
                const uint j = n0 + col;
                const float v = T[lr * BN + col];
                // Columns arrive in increasing id, so ">" keeps the lower id on ties.
                if (j < n && j != self_j && v > bs[wslot]) {
                    bs[wslot] = v;
                    bj[wslot] = j;
                    wslot = worst_slot(bs, bj, kk);
                }
            }
        }
    }
    // Merge the two half lists of each row: half 1 publishes, half 0 merges.
    threadgroup float* mb = Xs;                 // BM * KMAX floats fit in Xs
    threadgroup uint* mj = (threadgroup uint*)T; // and BM * KMAX ids in T
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (h == 1) {
        for (uint t = 0; t < kk; ++t) { mb[lr * KMAX + t] = bs[t]; mj[lr * KMAX + t] = bj[t]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (h == 0 && grow < p.rows) {
        for (uint t = 0; t < kk; ++t) {
            const float v = mb[lr * KMAX + t];
            const uint j = mj[lr * KMAX + t];
            if (kbetter(v, j, bs[wslot], bj[wslot])) {
                bs[wslot] = v;
                bj[wslot] = j;
                wslot = worst_slot(bs, bj, kk);
            }
        }
        // Insertion sort, best first.
        for (uint a = 1; a < kk; ++a) {
            const float v = bs[a];
            const uint j = bj[a];
            uint b = a;
            while (b > 0 && kbetter(v, j, bs[b - 1], bj[b - 1])) { bs[b] = bs[b - 1]; bj[b] = bj[b - 1]; --b; }
            bs[b] = v;
            bj[b] = j;
        }
        for (uint t = 0; t < kk; ++t) {
            out[2 * ((ulong)grow * kk + t)] = bj[t];
            out[2 * ((ulong)grow * kk + t) + 1] = as_type<uint>(bs[t]);
        }
    }
}

#define KNN_KERNEL(NAME, VEC)                                                \
kernel void NAME(                                                            \
    device const float* x   [[buffer(0)]],                                   \
    device const float* c   [[buffer(1)]],                                   \
    device uint*        out [[buffer(2)]],                                   \
    constant KnnParams& p   [[buffer(3)]],                                   \
    uint tg     [[threadgroup_position_in_grid]],                            \
    ushort tid  [[thread_index_in_threadgroup]],                             \
    ushort sg   [[simdgroup_index_in_threadgroup]],                          \
    ushort lane [[thread_index_in_simdgroup]])                               \
{                                                                            \
    threadgroup float Xs[64 * (32 + 4)];                                     \
    threadgroup float Cs[32 * (32 + 4)];                                     \
    threadgroup float T[64 * 32];                                            \
    threadgroup uint rsrc[64];                                               \
    knn_topk<64, 32, 32, 2, 2, VEC>(x, c, out, p, tg, tid, sg, lane, Xs, Cs, T, rsrc); \
}

KNN_KERNEL(knn_topk_v4, true)
KNN_KERNEL(knn_topk_s, false)
"#;

/// Exact top-k inner-product search on Metal: `MetalCore::topk` (layer 1).
pub struct MetalKnn {
    core: Mutex<MetalCore>,
}

impl MetalKnn {
    pub fn new() -> Result<Self> {
        Ok(Self {
            core: Mutex::new(MetalCore::new()?),
        })
    }

    /// Best `k` data rows of every query row by inner product, best first.
    ///
    /// Returns (ids, scores), each `queries x k`. With `exclude_self`, query row
    /// `i` is data row `i` (the caller passes the same matrix) and is skipped.
    /// Rows with fewer than `k` candidates are padded with id `u32::MAX`, score `-inf`.
    pub fn search(
        &self,
        queries: &[f32],
        data: &[f32],
        d: usize,
        k: usize,
        exclude_self: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let mut c = self
            .core
            .lock()
            .map_err(|_| anyhow::anyhow!("knn lock poisoned"))?;
        Device::topk(&mut *c, queries, data, d, k, exclude_self)
    }
}
