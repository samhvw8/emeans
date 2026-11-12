//! The Metal backend of the op contract, on its own command queue.
//!
//! Buffers are shared-storage `MTLBuffer`s with Metal's hazard tracking, so
//! dispatches run in order without explicit barriers. Ops are encoded into the
//! current command buffer; `flush` commits it without waiting, and the
//! `read_*`/`write_*` calls wait for every committed buffer first. Training
//! therefore encodes batch `b + 1` while the GPU runs batch `b`.
//!
//! Two libraries, both compiled from source at runtime (`newLibraryWithSource`):
//! `TOP1_SOURCE` (the fused assignment kernel, fast math) and `OPS_SOURCE`
//! (scatter-add, EMA, norms: safe math, no FMA contraction, so they round like
//! the CPU code). Top-k uses [`crate::KNN_SOURCE`].

use std::ffi::c_void;
use std::ops::Range;
use std::ptr::NonNull;

use anyhow::{Context, Result, bail, ensure};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

use emeans_core::assign::{Assigner, Metric, Rows};
use emeans_core::core::{Device, EmaStep, Mat, Select, check_select};

type Pipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Cmd = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Enc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Top-1 tiles compiled into `TOP1_SOURCE` (`assign_top1`, d % 4 == 0):
/// (BM, BN, BK, WM, WN); each simdgroup owns a 32 x 32 block. The engine picks
/// one per call (`auto_top1`) unless a tile is forced. The tile sweep that
/// chose them is in `.okf/results/metal-kernels.md`.
pub const FUSED_TILES: &[(usize, usize, usize, usize, usize)] =
    &[(64, 128, 32, 2, 4), (64, 64, 16, 2, 2)];
/// Threadgroups a top-1 dispatch should reach before centroid parts stop
/// paying (16 GPU cores on the M1 Pro).
const TOP1_GROUPS: usize = 48;
/// Tile for d % 4 != 0 (scalar loads).
const SCALAR_TILE: (usize, usize, usize, usize, usize) = (64, 32, 32, 2, 2);

pub const TOP1_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Fragment layout and loading follow MLX steel GEMM (BlockMMA): lane l of an
// 8x8 simdgroup matrix holds (sm, sn) and (sm, sn + 1), with
// qid = l / 4, sm = (qid & 4) + (l / 2) % 4, sn = (qid & 2) * 2 + (l % 2) * 2.

struct Params {
    uint rows;     // rows in this dispatch: positions r0 .. r0 + rows
    uint k;
    uint d;
    uint metric;   // 0 = L2 (min ||c||^2 - 2 x.c), 1 = inner product (max x.c)
    uint indexed;  // 1: position r is x[idx[r]]; 0: position r is x[r - r0 + xr0]
    uint r0;       // first position (outputs, ids)
    uint xr0;      // dense: x row of the first position
    uint parts;    // centroid parts (assign_top1): grid y; 1 = write labels directly
    uint kp;       // centroids per part (a multiple of the tile's BN)
};

inline bool better(bool ip, float a, uint ja, float b, uint jb) {
    // Is (a, ja) better than (b, jb)? Ties keep the lower centroid id.
    return ip ? (a > b || (a == b && ja < jb)) : (a < b || (a == b && ja < jb));
}

template <uint BM, uint BN, uint BK, uint WM, uint WN, bool VEC>
inline void assign_fused(
    device const float* x, device const uint* idx, device const float* c,
    device const float* cn, device uint* labels, device float* scores, constant Params& p, uint tg, ushort tid, ushort sg, ushort lane,
    threadgroup float* Xs, threadgroup float* Cs, threadgroup uint* rsrc,
    threadgroup float* tb, threadgroup uint* tj)
{
    constexpr uint NT = WM * WN * 32;
    constexpr uint LDK = BK + 4;
    constexpr uint TMS = 8 * WM;      // row stride between a simdgroup's fragments
    constexpr uint TNS = 8 * WN;      // column stride
    constexpr uint TM = BM / TMS;
    constexpr uint TN = BN / TNS;

    const uint row0 = tg * BM;
    const uint d = p.d, k = p.k;
    const bool ip = p.metric == 1;

    for (uint r = tid; r < BM; r += NT) {
        const uint gr = row0 + r;
        rsrc[r] = gr < p.rows ? (p.indexed ? idx[p.r0 + gr] : p.xr0 + gr) : 0xffffffffu;
    }

    const ushort tm = 8 * (sg / WN);
    const ushort tn = 8 * (sg % WN);
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + (lane / 2) % 4;
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint a_off = (tm + fm) * LDK + fn;
    const uint b_off = (tn + fn) * LDK + fm;

    float best[TM];
    uint bj[TM];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i) { best[i] = ip ? -INFINITY : INFINITY; bj[i] = 0; }

    for (uint n0 = 0; n0 < k; n0 += BN) {
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
                    if (gj < k && gc < d) v = *(device const float4*)(c + (ulong)gj * d + gc);
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
                    Cs[r * LDK + cc] = (gj < k && gc < d) ? c[(ulong)gj * d + gc] : 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup const float* As = Xs + a_off;
            threadgroup const float* Bs = Cs + b_off;
            #pragma clang loop unroll(full)
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 a[TM];
                simdgroup_float8x8 b[TN];
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < TM; ++i) {
                    a[i].thread_elements()[0] = As[i * TMS * LDK + kk];
                    a[i].thread_elements()[1] = As[i * TMS * LDK + kk + 1];
                }
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; ++j) {
                    b[j].thread_elements()[0] = Bs[j * TNS * LDK + kk];
                    b[j].thread_elements()[1] = Bs[j * TNS * LDK + LDK + kk];
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
        // Running best over this lane's own elements, in increasing centroid id.
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j) {
            const uint c0 = n0 + tn + j * TNS + fn;
            float n2a = 0.0f, n2b = 0.0f;
            if (!ip) {
                n2a = c0 < k ? cn[c0] : 0.0f;
                n2b = c0 + 1 < k ? cn[c0 + 1] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; ++i) {
                const thread auto& e = acc[i * TN + j].thread_elements();
                const float va = ip ? e[0] : n2a - 2.0f * e[0];
                const float vb = ip ? e[1] : n2b - 2.0f * e[1];
                if (c0 < k && better(ip, va, c0, best[i], bj[i])) { best[i] = va; bj[i] = c0; }
                if (c0 + 1 < k && better(ip, vb, c0 + 1, best[i], bj[i])) { best[i] = vb; bj[i] = c0 + 1; }
            }
        }
    }
    // Combine the 4 lanes that share a row (lane bits 0 and 3), then the WN
    // simdgroups that share it through threadgroup memory.
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i) {
        float b0 = best[i];
        uint j0 = bj[i];
        float ob = simd_shuffle_xor(b0, (ushort)1);
        uint oj = simd_shuffle_xor(j0, (ushort)1);
        if (better(ip, ob, oj, b0, j0)) { b0 = ob; j0 = oj; }
        ob = simd_shuffle_xor(b0, (ushort)8);
        oj = simd_shuffle_xor(j0, (ushort)8);
        if (better(ip, ob, oj, b0, j0)) { b0 = ob; j0 = oj; }
        if ((lane & 9) == 0) {
            const uint r = tm + fm + i * TMS;
            tb[(sg % WN) * BM + r] = b0;
            tj[(sg % WN) * BM + r] = j0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Two threads per row: pick the winner and sum ||x||^2 halves for L2.
    for (uint t = tid; t < 2 * BM; t += NT) {
        const uint rr = t >> 1;
        const uint h = t & 1;
        const uint sr = rsrc[rr];
        float xn = 0.0f;
        if (!ip && sr != 0xffffffffu) {
            device const float* xr = x + (ulong)sr * d;
            for (uint q = h; q < d; q += 2) xn += xr[q] * xr[q];
        }
        xn += simd_shuffle_xor(xn, (ushort)1);
        if (h == 0 && sr != 0xffffffffu) {
            float b0 = tb[rr];
            uint j0 = tj[rr];
            for (uint w = 1; w < WN; ++w) {
                if (better(ip, tb[w * BM + rr], tj[w * BM + rr], b0, j0)) {
                    b0 = tb[w * BM + rr];
                    j0 = tj[w * BM + rr];
                }
            }
            const float score = ip ? b0 : max(xn + b0, 0.0f);
            labels[p.r0 + row0 + rr] = j0;
            scores[p.r0 + row0 + rr] = score;
        }
    }
}

#define FUSED_KERNEL(NAME, BM, BN, BK, WM, WN, VEC)                          \
kernel void NAME(                                                            \
    device const float* x   [[buffer(0)]],                                   \
    device const uint*  idx [[buffer(1)]],                                   \
    device const float* c   [[buffer(2)]],                                   \
    device const float* cn  [[buffer(3)]],                                   \
    device uint*     labels [[buffer(4)]],                                   \
    device float*    scores [[buffer(5)]],                                   \
    device float*    part_b [[buffer(6)]],                                   \
    device uint*     part_j [[buffer(7)]],                                   \
    constant Params&    p   [[buffer(8)]],                                   \
    uint tg     [[threadgroup_position_in_grid]],                            \
    ushort tid  [[thread_index_in_threadgroup]],                             \
    ushort sg   [[simdgroup_index_in_threadgroup]],                          \
    ushort lane [[thread_index_in_simdgroup]])                               \
{                                                                            \
    threadgroup float Xs[BM * (BK + 4)];                                     \
    threadgroup float Cs[BN * (BK + 4)];                                     \
    threadgroup uint rsrc[BM];                                               \
    threadgroup float tb[WN * BM];                                           \
    threadgroup uint tj[WN * BM];                                            \
    assign_fused<BM, BN, BK, WM, WN, VEC>(x, idx, c, cn, labels, scores, p, tg, tid, sg, lane, Xs, Cs, rsrc, tb, tj); \
}

FUSED_KERNEL(fused_64_32_32_2_2_s, 64, 32, 32, 2, 2, false)

// The main top-1 kernel (d % 4 == 0). Same fragment math as assign_fused, plus:
// - software pipelining: the (centroid tile, depth slice) steps are one flat
//   loop, and the next slice is loaded from device memory into registers while
//   the current one is multiplied from threadgroup memory;
// - centroid parts: threadgroup (x, y) takes row tile x and centroids
//   [y * kp, min(k, (y + 1) * kp)). With parts > 1 it writes its best (score,
//   id) per row to part_b / part_j and top1_combine picks the winner, so small
//   batches still fill the GPU.
template <uint BM, uint BN, uint BK, uint WM, uint WN>
inline void assign_top1(
    device const float* x, device const uint* idx, device const float* c,
    device const float* cn, device uint* labels, device float* scores,
    device float* part_b, device uint* part_j, constant Params& p,
    uint2 tg, ushort tid, ushort sg, ushort lane,
    threadgroup float* Xs, threadgroup float* Cs, threadgroup uint* rsrc,
    threadgroup float* tb, threadgroup uint* tj)
{
    constexpr uint NT = WM * WN * 32;
    constexpr uint LDK = BK + 4;
    constexpr uint TMS = 8 * WM;
    constexpr uint TNS = 8 * WN;
    constexpr uint TM = BM / TMS;
    constexpr uint TN = BN / TNS;
    constexpr uint KV = BK / 4;
    constexpr uint XV = BM * KV, CV = BN * KV;
    constexpr uint NXV = (XV + NT - 1) / NT, NCV = (CV + NT - 1) / NT;

    const uint row0 = tg.x * BM;
    const uint d = p.d;
    const uint cbeg = tg.y * p.kp;
    const uint k = min(p.k, cbeg + p.kp);
    const bool ip = p.metric == 1;
    const uint KS = (d + BK - 1) / BK;
    const uint S = cbeg < k ? KS * ((k - cbeg + BN - 1) / BN) : 0;

    for (uint r = tid; r < BM; r += NT) {
        const uint gr = row0 + r;
        rsrc[r] = gr < p.rows ? (p.indexed ? idx[p.r0 + gr] : p.xr0 + gr) : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort tm = 8 * (sg / WN);
    const ushort tn = 8 * (sg % WN);
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + (lane / 2) % 4;
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint a_off = (tm + fm) * LDK + fn;
    const uint b_off = (tn + fn) * LDK + fm;

    float best[TM];
    uint bj[TM];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i) { best[i] = ip ? -INFINITY : INFINITY; bj[i] = 0; }

    simdgroup_float8x8 acc[TM * TN];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TM * TN; ++t)
        acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    float4 rx[NXV];
    float4 rc[NCV];
    // Load step s of the flat loop into registers (zeros outside the data).
    #define TOP1_LOAD(S_)                                                            \
    {                                                                                \
        const uint ln0 = cbeg + ((S_) / KS) * BN, lk0 = ((S_) % KS) * BK;            \
        _Pragma("clang loop unroll(full)")                                           \
        for (uint i = 0; i < NXV; ++i) {                                             \
            const uint e = tid + i * NT;                                             \
            const uint r = e / KV, gc = lk0 + (e % KV) * 4;                          \
            float4 v = 0.0f;                                                         \
            if (e < XV) {                                                            \
                const uint sr = rsrc[r];                                             \
                if (sr != 0xffffffffu && gc < d) v = *(device const float4*)(x + (ulong)sr * d + gc); \
            }                                                                        \
            rx[i] = v;                                                               \
        }                                                                            \
        _Pragma("clang loop unroll(full)")                                           \
        for (uint i = 0; i < NCV; ++i) {                                             \
            const uint e = tid + i * NT;                                             \
            const uint gj = ln0 + e / KV, gc = lk0 + (e % KV) * 4;                   \
            float4 v = 0.0f;                                                         \
            if (e < CV && gj < k && gc < d) v = *(device const float4*)(c + (ulong)gj * d + gc); \
            rc[i] = v;                                                               \
        }                                                                            \
    }
    #define TOP1_STORE                                                               \
    {                                                                                \
        _Pragma("clang loop unroll(full)")                                           \
        for (uint i = 0; i < NXV; ++i) {                                             \
            const uint e = tid + i * NT;                                             \
            if (e < XV) *(threadgroup float4*)(Xs + (e / KV) * LDK + (e % KV) * 4) = rx[i]; \
        }                                                                            \
        _Pragma("clang loop unroll(full)")                                           \
        for (uint i = 0; i < NCV; ++i) {                                             \
            const uint e = tid + i * NT;                                             \
            if (e < CV) *(threadgroup float4*)(Cs + (e / KV) * LDK + (e % KV) * 4) = rc[i]; \
        }                                                                            \
    }
    if (S > 0) {
        TOP1_LOAD(0u)
        TOP1_STORE
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const float* As = Xs + a_off;
    threadgroup const float* Bs = Cs + b_off;
    for (uint s = 0; s < S; ++s) {
        const bool more = s + 1 < S;
        if (more) TOP1_LOAD(s + 1)
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[TM];
            simdgroup_float8x8 b[TN];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; ++i) {
                a[i].thread_elements()[0] = As[i * TMS * LDK + kk];
                a[i].thread_elements()[1] = As[i * TMS * LDK + kk + 1];
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; ++j) {
                b[j].thread_elements()[0] = Bs[j * TNS * LDK + kk];
                b[j].thread_elements()[1] = Bs[j * TNS * LDK + LDK + kk];
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; ++j) {
                    const uint js = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(acc[i * TN + js], a[i], b[js], acc[i * TN + js]);
                }
            }
        }
        if (s % KS == KS - 1) {
            // Running best over this lane's own elements, in increasing centroid id.
            const uint n0 = cbeg + (s / KS) * BN;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; ++j) {
                const uint c0 = n0 + tn + j * TNS + fn;
                float n2a = 0.0f, n2b = 0.0f;
                if (!ip) {
                    n2a = c0 < k ? cn[c0] : 0.0f;
                    n2b = c0 + 1 < k ? cn[c0 + 1] : 0.0f;
                }
                #pragma clang loop unroll(full)
                for (uint i = 0; i < TM; ++i) {
                    const thread auto& e = acc[i * TN + j].thread_elements();
                    const float va = ip ? e[0] : n2a - 2.0f * e[0];
                    const float vb = ip ? e[1] : n2b - 2.0f * e[1];
                    if (c0 < k && better(ip, va, c0, best[i], bj[i])) { best[i] = va; bj[i] = c0; }
                    if (c0 + 1 < k && better(ip, vb, c0 + 1, best[i], bj[i])) { best[i] = vb; bj[i] = c0 + 1; }
                }
            }
            #pragma clang loop unroll(full)
            for (uint t = 0; t < TM * TN; ++t)
                acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        if (more) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            TOP1_STORE
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #undef TOP1_LOAD
    #undef TOP1_STORE
    // Combine the 4 lanes that share a row (lane bits 0 and 3), then the WN
    // simdgroups that share it through threadgroup memory.
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i) {
        float b0 = best[i];
        uint j0 = bj[i];
        float ob = simd_shuffle_xor(b0, (ushort)1);
        uint oj = simd_shuffle_xor(j0, (ushort)1);
        if (better(ip, ob, oj, b0, j0)) { b0 = ob; j0 = oj; }
        ob = simd_shuffle_xor(b0, (ushort)8);
        oj = simd_shuffle_xor(j0, (ushort)8);
        if (better(ip, ob, oj, b0, j0)) { b0 = ob; j0 = oj; }
        if ((lane & 9) == 0) {
            const uint r = tm + fm + i * TMS;
            tb[(sg % WN) * BM + r] = b0;
            tj[(sg % WN) * BM + r] = j0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.parts > 1) {
        for (uint rr = tid; rr < BM; rr += NT) {
            if (row0 + rr >= p.rows) continue;
            float b0 = tb[rr];
            uint j0 = tj[rr];
            for (uint w = 1; w < WN; ++w) {
                if (better(ip, tb[w * BM + rr], tj[w * BM + rr], b0, j0)) {
                    b0 = tb[w * BM + rr];
                    j0 = tj[w * BM + rr];
                }
            }
            part_b[tg.y * p.rows + row0 + rr] = b0;
            part_j[tg.y * p.rows + row0 + rr] = j0;
        }
        return;
    }
    // Two threads per row: pick the winner and sum ||x||^2 halves for L2.
    for (uint t = tid; t < 2 * BM; t += NT) {
        const uint rr = t >> 1;
        const uint h = t & 1;
        const uint sr = rsrc[rr];
        float xn = 0.0f;
        if (!ip && sr != 0xffffffffu) {
            device const float* xr = x + (ulong)sr * d;
            for (uint q = h; q < d; q += 2) xn += xr[q] * xr[q];
        }
        xn += simd_shuffle_xor(xn, (ushort)1);
        if (h == 0 && sr != 0xffffffffu) {
            float b0 = tb[rr];
            uint j0 = tj[rr];
            for (uint w = 1; w < WN; ++w) {
                if (better(ip, tb[w * BM + rr], tj[w * BM + rr], b0, j0)) {
                    b0 = tb[w * BM + rr];
                    j0 = tj[w * BM + rr];
                }
            }
            const float score = ip ? b0 : max(xn + b0, 0.0f);
            labels[p.r0 + row0 + rr] = j0;
            scores[p.r0 + row0 + rr] = score;
        }
    }
}

#define TOP1_KERNEL(NAME, BM, BN, BK, WM, WN)                                \
kernel void NAME(                                                            \
    device const float* x   [[buffer(0)]],                                   \
    device const uint*  idx [[buffer(1)]],                                   \
    device const float* c   [[buffer(2)]],                                   \
    device const float* cn  [[buffer(3)]],                                   \
    device uint*     labels [[buffer(4)]],                                   \
    device float*    scores [[buffer(5)]],                                   \
    device float*    part_b [[buffer(6)]],                                   \
    device uint*     part_j [[buffer(7)]],                                   \
    constant Params&    p   [[buffer(8)]],                                   \
    uint2 tg    [[threadgroup_position_in_grid]],                            \
    ushort tid  [[thread_index_in_threadgroup]],                             \
    ushort sg   [[simdgroup_index_in_threadgroup]],                          \
    ushort lane [[thread_index_in_simdgroup]])                               \
{                                                                            \
    threadgroup float Xs[BM * (BK + 4)];                                     \
    threadgroup float Cs[BN * (BK + 4)];                                     \
    threadgroup uint rsrc[BM];                                               \
    threadgroup float tb[WN * BM];                                           \
    threadgroup uint tj[WN * BM];                                            \
    assign_top1<BM, BN, BK, WM, WN>(x, idx, c, cn, labels, scores, part_b, part_j, p, tg, tid, sg, lane, Xs, Cs, rsrc, tb, tj); \
}

// One thread per row: the best of the parts (in increasing centroid order;
// ties keep the lower id), then the score as the single-part kernel writes it.
kernel void top1_combine(
    device const float* x   [[buffer(0)]],
    device const uint*  idx [[buffer(1)]],
    device const float* c   [[buffer(2)]],
    device const float* cn  [[buffer(3)]],
    device uint*     labels [[buffer(4)]],
    device float*    scores [[buffer(5)]],
    device float*    part_b [[buffer(6)]],
    device uint*     part_j [[buffer(7)]],
    constant Params&    p   [[buffer(8)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= p.rows) return;
    const bool ip = p.metric == 1;
    float b0 = part_b[r];
    uint j0 = part_j[r];
    for (uint q = 1; q < p.parts; ++q) {
        const float b = part_b[q * p.rows + r];
        const uint j = part_j[q * p.rows + r];
        if (better(ip, b, j, b0, j0)) { b0 = b; j0 = j; }
    }
    float score = b0;
    if (!ip) {
        device const float* xr = x + (ulong)(p.indexed ? idx[p.r0 + r] : p.xr0 + r) * p.d;
        float e = 0.0f, o = 0.0f;
        for (uint q = 0; q < p.d; q += 2) e += xr[q] * xr[q];
        for (uint q = 1; q < p.d; q += 2) o += xr[q] * xr[q];
        score = max((e + o) + b0, 0.0f);
    }
    labels[p.r0 + r] = j0;
    scores[p.r0 + r] = score;
}

TOP1_KERNEL(top1_64_128_32_2_4, 64, 128, 32, 2, 4)
TOP1_KERNEL(top1_64_64_16_2_2, 64, 64, 16, 2, 2)
"#;

pub const OPS_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;

// Exact ops: compiled with MTLMathModeSafe and no FMA contraction, so each
// add, multiply and divide rounds like the CPU's f32 code.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

struct RowsP { uint rows; uint d; };

struct GatherP { uint r0; uint rows; uint d; };

// Thread (t, r): out[r, t] = x[idx[r0 + r], t].
kernel void gather_rows(
    device const float* x [[buffer(0)]], device const uint* idx [[buffer(1)]],
    device float* out [[buffer(2)]], constant GatherP& p [[buffer(3)]],
    uint2 g [[thread_position_in_grid]])
{
    const uint t = g.x, r = g.y;
    if (t >= p.d || r >= p.rows) return;
    out[(ulong)r * p.d + t] = x[(ulong)idx[p.r0 + r] * p.d + t];
}

// Row norms and normalization, bit-equal to the CPU's sequential f32 sum:
// one simdgroup per row (8 rows per threadgroup). Loads, the max and the
// divides run on all 32 lanes; the sum walks the row in column order, taking
// each lane's rounded square with simd_shuffle, so every lane adds the same
// values in the CPU's order. The old kernels (one thread per row, or lane 0
// adding squares staged in threadgroup memory) were memory-latency bound
// (.okf/results/metal-kernels.md).
kernel void row_norms(
    device const float* m [[buffer(0)]], device float* out [[buffer(1)]],
    constant RowsP& p [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg * 8 + sg;
    if (row >= p.rows) return;
    device const float* r = m + (ulong)row * p.d;
    float s = 0.0f;
    for (uint t0 = 0; t0 < p.d; t0 += 32) {
        const uint t = t0 + lane;
        const float v = t < p.d ? r[t] : 0.0f;
        const float q = v * v;
        const uint n = min(32u, p.d - t0);
        if (n == 32) {
            for (ushort l = 0; l < 32; ++l) s = s + simd_shuffle(q, l);
        } else {
            for (ushort l = 0; l < n; ++l) s = s + simd_shuffle(q, l);
        }
    }
    if (lane == 0) out[row] = s;
}

// faiss normalize_into, same order as the CPU: divide by the max magnitude,
// then by the norm of the scaled row.
kernel void normalize_rows(
    device const float* m [[buffer(0)]], device float* out [[buffer(1)]],
    constant RowsP& p [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg * 8 + sg;
    if (row >= p.rows) return;
    device const float* r = m + (ulong)row * p.d;
    device float* o = out + (ulong)row * p.d;
    float mx = 0.0f;
    for (uint t = lane; t < p.d; t += 32) mx = max(mx, fabs(r[t]));
    mx = simd_max(mx);
    const float s = mx > 0.0f ? mx : 1.0f;
    float ss = 0.0f;
    for (uint t0 = 0; t0 < p.d; t0 += 32) {
        const uint t = t0 + lane;
        const float v = t < p.d ? precise::divide(r[t], s) : 0.0f;
        const float q = v * v;
        const uint n = min(32u, p.d - t0);
        if (n == 32) {
            for (ushort l = 0; l < 32; ++l) ss = ss + simd_shuffle(q, l);
        } else {
            for (ushort l = 0; l < n; ++l) ss = ss + simd_shuffle(q, l);
        }
    }
    const float nrm = precise::sqrt(ss);
    const float s2 = nrm > 0.0f ? nrm : 1.0f;
    for (uint t = lane; t < p.d; t += 32) o[t] = precise::divide(precise::divide(r[t], s), s2);
}

struct SplitP { uint npairs; uint d; float eps; };

// One threadgroup per (donor, dead) pair: apply_splits on the device.
kernel void split_pairs(
    device float* pos [[buffer(0)]], device float* usage [[buffer(1)]],
    device const uint* pairs [[buffer(2)]], constant SplitP& p [[buffer(3)]],
    uint pi [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    threadgroup uint differ;
    if (tid == 0) differ = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong s = pairs[2 * pi], t = pairs[2 * pi + 1];
    for (uint i = tid; i < p.d; i += nt) {
        const float c = pos[s * p.d + i];
        const float off = (i % 2 == 0) ? p.eps : -p.eps;
        const float dn = c * (1.0f - off);
        const float ch = c * (1.0f + off);
        if (dn != ch) differ = 1;
        pos[s * p.d + i] = dn;
        pos[t * p.d + i] = ch;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        if (differ == 0) {
            const float c0 = pos[t * p.d];
            const float up = nextafter(c0, INFINITY);
            pos[t * p.d] = isfinite(up) ? up : nextafter(c0, -INFINITY);
        }
        const float hm = usage[s] / 2.0f;
        usage[s] = hm;
        usage[t] = hm;
    }
}

struct ScatterP {
    uint lo;       // first selection position
    uint m;        // positions lo .. lo + m
    uint k;
    uint d;
    uint indexed;  // position r is row idx[r] (else row r)
    uint acc;      // also acc += sums
};

// Stable counting sort of the batch's positions by label, in three passes
// whose work grows with m + k x blocks (the old count and place kernels each
// scanned all m labels once per cluster: 0.28 + 0.42 ms at m 16k, k 1024).
// Blocks of 256 positions. `bcnt` (blocks x k) is all zero between calls:
// scatter_block writes the counts present in a block, scatter_scan reads and
// clears them.

// One threadgroup per block: each position's rank among the earlier positions
// of its block with the same label; the last one of a label writes the count.
kernel void scatter_block(
    device const uint* labels [[buffer(0)]], device uint* rank [[buffer(1)]],
    device uint* bcnt [[buffer(2)]], constant ScatterP& p [[buffer(3)]],
    uint b [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup uint lab[256];
    const uint i = b * 256 + tid;
    const uint l = i < p.m ? labels[p.lo + i] : 0xffffffffu;
    lab[tid] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= p.m || l >= p.k) return;
    uint r = 0;
    for (uint j = 0; j < tid; ++j) r += lab[j] == l ? 1u : 0u;
    bool last = true;
    for (uint j = tid + 1; j < 256; ++j) {
        if (lab[j] == l) { last = false; break; }
    }
    rank[i] = r;
    if (last) bcnt[b * p.k + l] = r + 1;
}

// One threadgroup of 1024: counts[j] = rows of cluster j; start = exclusive
// prefix of counts (start[k] = m); boff[b, j] = start[j] + rows of j in the
// blocks before b. Clears bcnt.
kernel void scatter_scan(
    device uint* bcnt [[buffer(0)]], device uint* boff [[buffer(1)]],
    device uint* counts [[buffer(2)]], device uint* start [[buffer(3)]],
    constant ScatterP& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint part[32];
    const uint nb = (p.m + 255) / 256;
    const uint per = (p.k + nt - 1) / nt;
    const uint a = min(tid * per, p.k), e = min(a + per, p.k);
    uint local = 0;
    for (uint j = a; j < e; ++j) {
        uint c = 0;
        for (uint b = 0; b < nb; ++b) c += bcnt[b * p.k + j];
        counts[j] = c;
        local += c;
    }
    uint pre = simd_prefix_exclusive_sum(local);
    if (lane == 31) part[sg] = pre + local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint nsg = (nt + 31) / 32;
        uint v = lane < nsg ? part[lane] : 0u;
        uint ex = simd_prefix_exclusive_sum(v);
        if (lane < nsg) part[lane] = ex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint run = part[sg] + pre;
    for (uint j = a; j < e; ++j) {
        start[j] = run;
        for (uint b = 0; b < nb; ++b) {
            const uint c = bcnt[b * p.k + j];
            boff[b * p.k + j] = run;
            run += c;
            bcnt[b * p.k + j] = 0;
        }
    }
    if (tid == nt - 1) start[p.k] = run;
}

// One thread per position: its place in the cluster-sorted order.
kernel void scatter_place(
    device const uint* labels [[buffer(0)]], device const uint* rank [[buffer(1)]],
    device const uint* boff [[buffer(2)]], device uint* order [[buffer(3)]],
    constant ScatterP& p [[buffer(4)]], uint i [[thread_position_in_grid]])
{
    if (i >= p.m) return;
    const uint l = labels[p.lo + i];
    if (l >= p.k) return;
    order[boff[(i / 256) * p.k + l] + rank[i]] = i;
}

// Thread (t, j): sums[j, t] = rows of cluster j added in position order.
kernel void scatter_sum(
    device const float* x [[buffer(0)]], device const uint* idx [[buffer(1)]],
    device const uint* start [[buffer(2)]], device const uint* order [[buffer(3)]],
    device float* sums [[buffer(4)]], device float* acc [[buffer(5)]],
    constant ScatterP& p [[buffer(6)]],
    uint2 g [[thread_position_in_grid]])
{
    const uint t = g.x, j = g.y;
    if (t >= p.d || j >= p.k) return;
    float s = 0.0f;
    for (uint q = start[j]; q < start[j + 1]; ++q) {
        const uint pos = p.lo + order[q];
        const ulong row = p.indexed ? idx[pos] : pos;
        s += x[row * p.d + t];
    }
    sums[(ulong)j * p.d + t] = s;
    if (p.acc) acc[(ulong)j * p.d + t] += s;
}

struct EmaP { uint k; uint d; float alpha; float decay; uint replace; };

// One threadgroup per cluster j: the positions read the old mass, then thread
// 0 writes the new one (one dispatch instead of two).
kernel void ema_update(
    device float* pos [[buffer(0)]], device float* usage [[buffer(1)]],
    device const float* sums [[buffer(2)]], device const uint* counts [[buffer(3)]],
    constant EmaP& p [[buffer(4)]], uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]])
{
    if (j >= p.k) return;
    const uint cnt = counts[j];
    const float old = usage[j];
    if (cnt > 0) {
        const float c = float(cnt);
        const float retained = p.replace ? 0.0f : old * p.decay;
        const float step = p.alpha * c;
        const float mass = retained + step;
        const float w = precise::divide(step, mass);
        for (uint t = tid; t < p.d; t += nt) {
            const ulong i = (ulong)j * p.d + t;
            const float mean = precise::divide(sums[i], c);
            const float cur = pos[i];
            pos[i] = (w == 1.0f) ? mean : cur + w * (mean - cur);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        float u = old;
        if (!p.replace) u = u * p.decay;
        if (cnt > 0) u = (p.replace ? 0.0f : u) + p.alpha * float(cnt);
        usage[j] = u;
    }
}
"#;

/// A shared-storage Metal buffer of `len` 4-byte elements, starting `off`
/// bytes into `buf` (non-zero only for a no-copy view of host memory).
///
/// A no-copy view of shared data carries its owner (`keep`), and every clone
/// does too, so safe code cannot hold the view past the memory it reads.
pub struct MBuf {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    off: usize,
    /// The owner of the host memory this buffer views; None for a buffer
    /// that owns its storage.
    keep: Option<emeans_core::assign::SharedData>,
}

impl MBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Clone for MBuf {
    fn clone(&self) -> Self {
        MBuf {
            buf: self.buf.clone(),
            len: self.len,
            off: self.off,
            keep: self.keep.clone(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Top1P {
    rows: u32,
    k: u32,
    d: u32,
    metric: u32,
    indexed: u32,
    r0: u32,
    xr0: u32,
    parts: u32,
    kp: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RowsP {
    rows: u32,
    d: u32,
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
struct KnnP {
    rows: u32,
    n: u32,
    d: u32,
    kk: u32,
    self_off: u32,
    exclude: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GatherP {
    r0: u32,
    rows: u32,
    d: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SplitP {
    npairs: u32,
    d: u32,
    eps: f32,
}

struct Ops {
    split: Pipe,
    gather: Pipe,
    row_norms: Pipe,
    normalize: Pipe,
    block: Pipe,
    scan: Pipe,
    place: Pipe,
    sum: Pipe,
    ema: Pipe,
}

use crate::KNN_KMAX;

/// The Metal engine: device, queue, pipelines, the open command buffer.
pub struct MetalCore {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    top1_v4: Vec<Pipe>,
    top1_s: Pipe,
    top1_combine: Pipe,
    /// A forced entry of `FUSED_TILES`; None = `auto_top1` per call.
    tile: Option<usize>,
    /// A forced number of centroid parts (`EMEANS_TOP1_PARTS`); None = auto.
    parts: Option<usize>,
    part_b: Option<MBuf>,
    part_j: Option<MBuf>,
    knn_v4: Pipe,
    knn_s: Pipe,
    ops: Ops,
    cmd: Option<Cmd>,
    enc: Option<Enc>,
    /// Committed command buffers and the label of their work.
    inflight: Vec<(Cmd, &'static str)>,
    /// Label of the open command buffer (the last dispatch's op).
    label: &'static str,
    /// `EMEANS_METAL_PROFILE`: `op` commits after every dispatch, `batch` per
    /// flush; GPU start/end times are collected and printed on drop.
    profile: Option<Profile>,
    /// Data kept by `Assigner::bind_data`/`bind_shared`: (pointer, length, buffer).
    bound: Option<(usize, usize, MBuf)>,
    /// Owners of the host memory that queued work reads through a no-copy
    /// view: held until `sync` has waited for that work, so dropping the view
    /// (and the caller's handle) cannot free memory the GPU still reads.
    owners: Vec<emeans_core::assign::SharedData>,
    start: Option<MBuf>,
    order: Option<MBuf>,
    /// Counting-sort scratch of `scatter_add`: ranks (m), block counts (all
    /// zero between calls) and block offsets (blocks x k).
    rank: Option<MBuf>,
    bcnt: Option<MBuf>,
    boff: Option<MBuf>,
    gathered: Option<MBuf>,
    /// Query rows per top-k dispatch.
    pub knn_chunk: usize,
}

// SAFETY: Metal device, queue, pipeline and buffer objects are thread-safe;
// the command buffer and encoder are only touched through `&mut self`.
unsafe impl Send for MetalCore {}

fn ns_err(e: Retained<objc2_foundation::NSError>) -> anyhow::Error {
    anyhow::anyhow!("{}", e.localizedDescription())
}

fn size(w: usize, h: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: 1,
    }
}

impl MetalCore {
    /// Tiles chosen per call, or `EMEANS_FUSED_TILE=BM,BN,BK,WM,WN` (an entry
    /// of `FUSED_TILES`); `EMEANS_TOP1_PARTS` forces the centroid parts.
    pub fn new() -> Result<Self> {
        let tile = match std::env::var("EMEANS_FUSED_TILE") {
            Ok(v) => {
                let t: Vec<usize> = v
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()
                    .context("EMEANS_FUSED_TILE=BM,BN,BK,WM,WN")?;
                FUSED_TILES
                    .iter()
                    .position(|f| [f.0, f.1, f.2, f.3, f.4] == t[..])
                    .with_context(|| format!("EMEANS_FUSED_TILE {v} is not in {FUSED_TILES:?}"))?
            }
            Err(_) => return Self::build(None),
        };
        Self::with_tile(tile)
    }

    /// Always use top-1 tile `tile` (an index of `FUSED_TILES`).
    pub fn with_tile(tile: usize) -> Result<Self> {
        ensure!(tile < FUSED_TILES.len(), "tile index out of range");
        Self::build(Some(tile))
    }

    fn build(tile: Option<usize>) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("no Metal command queue")?;
        let compile = |src: &str, safe: bool| -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
            let opts = MTLCompileOptions::new();
            if safe {
                opts.setMathMode(MTLMathMode::Safe);
            }
            device
                .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
                .map_err(ns_err)
        };
        let fast = compile(TOP1_SOURCE, false).context("compile the top-1 shader")?;
        let exact = compile(OPS_SOURCE, true).context("compile the ops shader")?;
        let knn = compile(crate::KNN_SOURCE, false).context("compile the top-k shader")?;
        let make =
            |lib: &ProtocolObject<dyn MTLLibrary>, name: &str, threads: usize| -> Result<Pipe> {
                let f = lib
                    .newFunctionWithName(&NSString::from_str(name))
                    .with_context(|| format!("no kernel {name}"))?;
                let p = device
                    .newComputePipelineStateWithFunction_error(&f)
                    .map_err(ns_err)?;
                ensure!(
                    p.maxTotalThreadsPerThreadgroup() >= threads,
                    "{name} needs {threads} threads per threadgroup"
                );
                Ok(p)
            };
        let name = |t: &(usize, usize, usize, usize, usize)| {
            format!("{}_{}_{}_{}_{}", t.0, t.1, t.2, t.3, t.4)
        };
        let top1_v4 = FUSED_TILES
            .iter()
            .map(|t| make(&fast, &format!("top1_{}", name(t)), t.3 * t.4 * 32))
            .collect::<Result<Vec<_>>>()?;
        let st = SCALAR_TILE;
        let top1_s = make(&fast, &format!("fused_{}_s", name(&st)), st.3 * st.4 * 32)?;
        let parts = match std::env::var("EMEANS_TOP1_PARTS") {
            Ok(v) => Some(v.parse::<usize>().context("EMEANS_TOP1_PARTS")?.max(1)),
            Err(_) => None,
        };
        let ops = Ops {
            gather: make(&exact, "gather_rows", 256)?,
            row_norms: make(&exact, "row_norms", 256)?,
            normalize: make(&exact, "normalize_rows", 256)?,
            split: make(&exact, "split_pairs", 256)?,
            block: make(&exact, "scatter_block", 256)?,
            scan: make(&exact, "scatter_scan", 1024)?,
            place: make(&exact, "scatter_place", 256)?,
            sum: make(&exact, "scatter_sum", 256)?,
            ema: make(&exact, "ema_update", 256)?,
        };
        Ok(Self {
            top1_combine: make(&fast, "top1_combine", 256)?,
            parts,
            part_b: None,
            part_j: None,
            knn_v4: make(&knn, "knn_topk_v4", 128)?,
            knn_s: make(&knn, "knn_topk_s", 128)?,
            device,
            queue,
            top1_v4,
            top1_s,
            tile,
            ops,
            cmd: None,
            enc: None,
            inflight: Vec::new(),
            label: "",
            profile: Profile::from_env(),
            bound: None,
            owners: Vec::new(),
            start: None,
            order: None,
            rank: None,
            bcnt: None,
            boff: None,
            gathered: None,
            knn_chunk: 4096,
        })
    }

    /// Force the top-1 centroid parts (None = auto).
    pub fn set_top1_parts(&mut self, parts: Option<usize>) {
        self.parts = parts.map(|p| p.max(1));
    }

    /// The forced top-1 tile, if any.
    pub fn tile(&self) -> Option<(usize, usize, usize, usize, usize)> {
        self.tile.map(|t| FUSED_TILES[t])
    }

    /// Top-1 tile and centroid parts for `rows` rows and `k` centroids,
    /// measured on the M1 Pro (`bench/src/bin/top1_bench.rs`; `.okf/results/metal-kernels.md`).
    fn auto_top1(&self, rows: usize, k: usize) -> (usize, usize) {
        // 64 x 128 x 32 is the fastest tile once k fills it and the batch has
        // enough row tiles; 64 x 64 x 16 wastes less for small k and small batches.
        let small = k <= 64 || (rows < 4096 && k <= 512);
        let tile = self.tile.unwrap_or(if small { 1 } else { 0 });
        let (bm, bn) = (FUSED_TILES[tile].0, FUSED_TILES[tile].1);
        let ntiles = k.div_ceil(bn);
        let parts = self
            .parts
            .unwrap_or_else(|| TOP1_GROUPS.div_ceil(rows.div_ceil(bm)))
            .clamp(1, ntiles);
        (tile, parts)
    }

    fn new_buf(&self, len: usize) -> Result<MBuf> {
        let bytes = (len * 4).max(4);
        let buf = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .context("Metal buffer allocation failed")?;
        // SAFETY: a fresh shared buffer of `bytes` bytes, not yet used by the GPU.
        unsafe { std::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>(), 0, bytes) };
        Ok(MBuf {
            buf,
            len,
            off: 0,
            keep: None,
        })
    }

    fn buf_from<T: Copy>(&self, v: &[T]) -> Result<MBuf> {
        assert_eq!(std::mem::size_of::<T>(), 4);
        if v.is_empty() {
            return self.new_buf(0);
        }
        // SAFETY: `v` is a live slice of v.len() * 4 bytes; Metal copies it.
        let buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(v.as_ptr() as *mut c_void).unwrap(),
                v.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .context("Metal buffer allocation failed")?;
        Ok(MBuf {
            buf,
            len: v.len(),
            off: 0,
            keep: None,
        })
    }

    /// A buffer over the host memory of `v` itself (no copy), when Metal
    /// accepts it: the pages that hold `v` are wrapped with
    /// `newBufferWithBytesNoCopy` and the view starts at `v`'s offset in the
    /// first page (16-byte aligned, for the float4 loads). Saves the copy
    /// (0.26-0.44 s for 614 MB on the M1 Pro) and the second copy's memory.
    ///
    /// The GPU reads `v` until the queued work using the buffer completes, so
    /// only callers that keep `v` alive and unchanged until then use it:
    /// `bind_shared` (stores the `Arc` in the buffer's `keep`, and `dispatch`
    /// holds it in `owners` until `sync`) and `topk` (borrows `v` and waits
    /// before it returns).
    /// `bind_data` and `Device::bind` copy. Kernels never write it. A view
    /// freed early reads the allocator's free-list words (seen in
    /// `tests/metal.rs` when `bind` used it).
    fn buf_nocopy(&self, v: &[f32]) -> Option<MBuf> {
        unsafe extern "C" {
            fn getpagesize() -> i32;
        }
        // SAFETY: a libc call without arguments.
        let page = usize::try_from(unsafe { getpagesize() }).ok()?;
        let p = v.as_ptr() as usize;
        let base = p & !(page - 1);
        let off = p - base;
        if v.is_empty()
            || !off.is_multiple_of(16)
            || std::env::var_os("EMEANS_METAL_COPY").is_some()
        {
            return None;
        }
        // Whole pages: the first and last ones hold bytes of `v`, so they are mapped.
        let len = (off + v.len() * 4).next_multiple_of(page);
        // SAFETY: [base, base + len) covers the mapped pages of `v`; no
        // deallocator, so Metal never frees the memory; the GPU only reads it.
        let buf = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(base as *mut c_void)?,
                    len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }?;
        Some(MBuf {
            buf,
            len: v.len(),
            off,
            keep: None,
        })
    }

    fn encoder(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if self.cmd.is_none() {
            self.cmd = Some(self.queue.commandBuffer().context("no command buffer")?);
        }
        if self.enc.is_none() {
            let e = self
                .cmd
                .as_ref()
                .unwrap()
                .computeCommandEncoder()
                .context("no compute encoder")?;
            self.enc = Some(e);
        }
        Ok(self.enc.as_deref().unwrap())
    }

    /// Commit the open command buffer without waiting.
    fn commit(&mut self) {
        if let Some(e) = self.enc.take() {
            e.endEncoding();
        }
        if let Some(c) = self.cmd.take() {
            c.commit();
            self.inflight.push((c, self.label));
        }
    }

    /// `Device::topk`; may leave work queued on error (see `topk`).
    fn knn(
        &mut self,
        queries: &[f32],
        data: &[f32],
        d: usize,
        kk: usize,
        exclude_self: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        ensure!(d > 0 && queries.len().is_multiple_of(d) && data.len().is_multiple_of(d));
        ensure!((1..=KNN_KMAX).contains(&kk), "kk must be 1..={KNN_KMAX}");
        let (m, n) = (queries.len() / d, data.len() / d);
        ensure!(n > 0, "no data rows");
        ensure!(
            !exclude_self || m == n,
            "exclude_self needs queries == data"
        );
        // No copy: `read` below waits for the GPU before this call returns.
        let db = match self.buf_nocopy(data) {
            Some(b) => b,
            None => self.buf_from(data)?,
        };
        let qb = if std::ptr::eq(queries, data) {
            db.clone()
        } else {
            match self.buf_nocopy(queries) {
                Some(b) => b,
                None => self.buf_from(queries)?,
            }
        };
        let pipe = if d.is_multiple_of(4) {
            self.knn_v4.clone()
        } else {
            self.knn_s.clone()
        };
        let chunk = self.knn_chunk.max(64);
        let out = self.new_buf(m * kk * 2)?;
        for q0 in (0..m).step_by(chunk) {
            let q1 = (q0 + chunk).min(m);
            let p = KnnP {
                rows: u32::try_from(q1 - q0)?,
                n: u32::try_from(n)?,
                d: u32::try_from(d)?,
                kk: kk as u32,
                self_off: u32::try_from(q0)?,
                exclude: exclude_self as u32,
            };
            // The kernel addresses queries and output from its dispatch's first row.
            let enc = self.encoder()?;
            enc.setComputePipelineState(&pipe);
            // SAFETY: offsets stay inside the buffers; Metal retains them.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&qb.buf), qb.off + q0 * d * 4, 0);
                enc.setBuffer_offset_atIndex(Some(&db.buf), db.off, 1);
                enc.setBuffer_offset_atIndex(Some(&out.buf), q0 * kk * 2 * 4, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p as *const KnnP as *mut c_void).unwrap(),
                    std::mem::size_of::<KnnP>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                size((q1 - q0).div_ceil(64), 1),
                size(128, 1),
            );
            self.commit();
        }
        let v: Vec<u32> = self.read(&out, 0..m * kk * 2)?;
        let ids = v.iter().step_by(2).copied().collect();
        let sims = v
            .iter()
            .skip(1)
            .step_by(2)
            .map(|&b| f32::from_bits(b))
            .collect();
        Ok((ids, sims))
    }

    /// Map a freshly bound data buffer for the GPU now (one tiny dispatch that
    /// reads row 0), not in the first training batch: 25 ms for 614 MB, against
    /// 25-230 ms added to the first run (`.okf/results/metal-kernels.md`).
    fn make_resident(&mut self, b: &MBuf, d: usize, len: usize) -> Result<()> {
        if d == 0 || len < d {
            return Ok(());
        }
        let ids = self.buf_from(&[0u32])?;
        let g = self.new_buf(d)?;
        let gp = GatherP {
            r0: 0,
            rows: 1,
            d: u32::try_from(d)?,
        };
        let gpipe = self.ops.gather.clone();
        self.dispatch(
            "touch",
            &gpipe,
            &[b, &ids, &g],
            bytes(&gp),
            size(d.div_ceil(64), 1),
            size(64, 4),
        )?;
        self.sync()
    }

    /// GPU time of the work `f` encodes: commits it alone and waits; ms from
    /// the first command buffer's GPU start to the last one's GPU end.
    pub fn gpu_ms(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<f64> {
        self.sync()?;
        f(self)?;
        self.commit();
        let cmds: Vec<Cmd> = self.inflight.iter().map(|(c, _)| c.clone()).collect();
        self.sync()?;
        let a = cmds
            .iter()
            .map(|c| c.GPUStartTime())
            .fold(f64::INFINITY, f64::min);
        let b = cmds.iter().map(|c| c.GPUEndTime()).fold(0.0, f64::max);
        Ok(if cmds.is_empty() { 0.0 } else { (b - a) * 1e3 })
    }

    /// Wait for all committed work; fail if any command buffer failed.
    pub fn sync(&mut self) -> Result<()> {
        self.commit();
        if let Some((last, _)) = self.inflight.last() {
            last.waitUntilCompleted();
        }
        let mut failed = None;
        for (c, label) in self.inflight.drain(..) {
            if c.status() != MTLCommandBufferStatus::Completed {
                c.waitUntilCompleted();
            }
            if c.status() == MTLCommandBufferStatus::Error {
                let msg = c
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".into());
                failed.get_or_insert(msg);
                continue;
            }
            if let Some(p) = self.profile.as_mut() {
                p.spans.push((label, c.GPUStartTime(), c.GPUEndTime()));
            }
        }
        // Every command buffer has completed: nothing reads host memory now.
        self.owners.clear();
        if let Some(msg) = failed {
            bail!("Metal command buffer failed: {msg}");
        }
        Ok(())
    }

    fn dispatch(
        &mut self,
        label: &'static str,
        pipe: &Pipe,
        bufs: &[&MBuf],
        params: &[u8],
        groups: MTLSize,
        threads: MTLSize,
    ) -> Result<()> {
        let pipe = pipe.clone();
        // Keep the owner of every no-copy view this work reads until `sync`.
        for k in bufs.iter().filter_map(|b| b.keep.as_ref()) {
            if !self
                .owners
                .last()
                .is_some_and(|o| std::sync::Arc::ptr_eq(o, k))
            {
                self.owners.push(k.clone());
            }
        }
        let enc = self.encoder()?;
        enc.setComputePipelineState(&pipe);
        for (i, b) in bufs.iter().enumerate() {
            // SAFETY: the buffers stay alive until the command buffer completes
            // (Metal retains bound buffers); the kernels index within `len`.
            unsafe { enc.setBuffer_offset_atIndex(Some(&b.buf), b.off, i) };
        }
        // SAFETY: `params` is a live POD struct; Metal copies the bytes.
        unsafe {
            enc.setBytes_length_atIndex(
                NonNull::new(params.as_ptr() as *mut c_void).unwrap(),
                params.len(),
                bufs.len(),
            )
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        self.label = label;
        if self.profile.as_ref().is_some_and(|p| p.per_op) {
            self.commit();
        }
        Ok(())
    }

    fn scratch(slot: &mut Option<MBuf>, dev: &Self, len: usize) -> Result<MBuf> {
        if slot.as_ref().is_none_or(|b| b.len < len) {
            *slot = Some(dev.new_buf(len.max(1024))?);
        }
        Ok(slot.as_ref().unwrap().clone())
    }

    fn read<T: Copy + Default>(&mut self, b: &MBuf, range: Range<usize>) -> Result<Vec<T>> {
        ensure!(
            range.start <= range.end && range.end <= b.len,
            "read past the buffer"
        );
        self.sync()?;
        let mut out = vec![T::default(); range.len()];
        // SAFETY: the GPU is idle; the buffer holds `len` 4-byte elements.
        unsafe {
            let src = b
                .buf
                .contents()
                .as_ptr()
                .byte_add(b.off)
                .cast::<T>()
                .add(range.start);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), range.len());
        }
        Ok(out)
    }

    fn write<T: Copy>(&mut self, b: &MBuf, offset: usize, v: &[T]) -> Result<()> {
        ensure!(offset + v.len() <= b.len, "write past the buffer");
        self.sync()?;
        // SAFETY: the GPU is idle; the range is inside the buffer.
        unsafe {
            let dst = b
                .buf
                .contents()
                .as_ptr()
                .byte_add(b.off)
                .cast::<T>()
                .add(offset);
            std::ptr::copy_nonoverlapping(v.as_ptr(), dst, v.len());
        }
        Ok(())
    }
}

/// GPU times of committed command buffers (`EMEANS_METAL_PROFILE=op|batch`).
struct Profile {
    per_op: bool,
    spans: Vec<(&'static str, f64, f64)>,
}

impl Profile {
    fn from_env() -> Option<Self> {
        match std::env::var("EMEANS_METAL_PROFILE").as_deref() {
            Ok("op") => Some(Self {
                per_op: true,
                spans: Vec::new(),
            }),
            Ok("batch") => Some(Self {
                per_op: false,
                spans: Vec::new(),
            }),
            _ => None,
        }
    }

    /// Per label: buffers, GPU ms total and median; then busy time, the span
    /// from the first start to the last end, and the idle gaps between buffers.
    fn report(&self) -> String {
        let mut out = String::new();
        if self.spans.is_empty() {
            return out;
        }
        let mut labels: Vec<&str> = self.spans.iter().map(|s| s.0).collect();
        labels.sort_unstable();
        labels.dedup();
        out.push_str("metal profile (GPU ms): label count total median\n");
        for l in labels {
            let mut v: Vec<f64> = self
                .spans
                .iter()
                .filter(|s| s.0 == l)
                .map(|s| (s.2 - s.1) * 1e3)
                .collect();
            v.sort_by(f64::total_cmp);
            let total: f64 = v.iter().sum();
            out.push_str(&format!(
                "  {l:<10} {:6} {:10.3} {:8.4}\n",
                v.len(),
                total,
                v[v.len() / 2]
            ));
        }
        let mut s: Vec<(f64, f64)> = self.spans.iter().map(|s| (s.1, s.2)).collect();
        s.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut busy, mut idle, mut end) = (0.0, 0.0, s[0].0);
        for &(a, b) in &s {
            if a > end {
                idle += a - end;
            }
            let a2 = a.max(end);
            if b > a2 {
                busy += b - a2;
                end = b;
            }
        }
        out.push_str(&format!(
            "  busy {:.3} ms, idle gaps {:.3} ms, span {:.3} ms, {} buffers\n",
            busy * 1e3,
            idle * 1e3,
            (end - s[0].0) * 1e3,
            s.len()
        ));
        out
    }
}

impl Drop for MetalCore {
    fn drop(&mut self) {
        // Buffers may view host memory (`buf_nocopy`): finish all GPU work first.
        let r = self.sync();
        if let Some(p) = &self.profile {
            eprint!("{}", p.report());
        }
        if let Err(e) = r {
            eprintln!("metal: {e:#}");
        }
    }
}

fn bytes<T: Copy>(v: &T) -> &[u8] {
    // SAFETY: T is a repr(C) POD params struct.
    unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), std::mem::size_of::<T>()) }
}

impl Device for MetalCore {
    type F = MBuf;
    type U = MBuf;

    fn name(&self) -> String {
        match self.tile {
            Some(t) => {
                let t = FUSED_TILES[t];
                format!("metal-fused-{}x{}x{}-{}x{}", t.0, t.1, t.2, t.3, t.4)
            }
            None => "metal-fused".into(),
        }
    }

    fn bind(&mut self, x: &[f32]) -> Result<MBuf> {
        // Serve the bound buffer only when it views memory an `Arc` pins
        // (`bind_shared`): then the same pointer and length are the same,
        // unchanged data. A `bind_data` copy is not served by address: the
        // caller may have freed the data, and a new buffer of the same size
        // can land at the same address.
        if let Some((p, l, b)) = &self.bound
            && *p == x.as_ptr() as usize
            && *l == x.len()
            && b.keep.is_some()
        {
            return Ok(b.clone());
        }
        // A copy: callers may free `x` before the queued work completes
        // (`assign_on` drops its gathered rows before reading the labels).
        self.buf_from(x)
    }

    fn upload_f32(&mut self, v: &[f32]) -> Result<MBuf> {
        self.buf_from(v)
    }

    fn upload_u32(&mut self, v: &[u32]) -> Result<MBuf> {
        self.buf_from(v)
    }

    fn zeros_f32(&mut self, n: usize) -> Result<MBuf> {
        self.new_buf(n)
    }

    fn zeros_u32(&mut self, n: usize) -> Result<MBuf> {
        self.new_buf(n)
    }

    fn write_f32(&mut self, b: &mut MBuf, offset: usize, v: &[f32]) -> Result<()> {
        self.write(b, offset, v)
    }

    fn write_u32(&mut self, b: &mut MBuf, offset: usize, v: &[u32]) -> Result<()> {
        self.write(b, offset, v)
    }

    fn read_f32(&mut self, b: &MBuf, range: Range<usize>) -> Result<Vec<f32>> {
        self.read(b, range)
    }

    fn read_u32(&mut self, b: &MBuf, range: Range<usize>) -> Result<Vec<u32>> {
        self.read(b, range)
    }

    fn flush(&mut self) -> Result<()> {
        self.commit();
        Ok(())
    }

    fn top1(
        &mut self,
        x: Mat<'_, MBuf>,
        sel: Select<'_, MBuf>,
        centroids: &MBuf,
        cnorm: Option<&MBuf>,
        k: usize,
        metric: Metric,
        labels: &mut MBuf,
        scores: &mut MBuf,
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
        let (pipe, t, parts) = if d.is_multiple_of(4) {
            let (tile, parts) = self.auto_top1(r.len(), k);
            (self.top1_v4[tile].clone(), FUSED_TILES[tile], parts)
        } else {
            (self.top1_s.clone(), SCALAR_TILE, 1)
        };
        let kp = k.div_ceil(t.1).div_ceil(parts) * t.1;
        let parts = k.div_ceil(kp);
        let mut p = Top1P {
            rows: u32::try_from(r.len())?,
            k: u32::try_from(k)?,
            d: u32::try_from(d)?,
            metric: (metric == Metric::InnerProduct) as u32,
            indexed: 0,
            r0: u32::try_from(r.start)?,
            xr0: u32::try_from(r.start)?,
            parts: u32::try_from(parts)?,
            kp: u32::try_from(kp)?,
        };
        let cn = cnorm.unwrap_or(centroids);
        // Rows chosen by id: gather them into one contiguous block first. The
        // kernel re-reads its rows once per centroid tile, and scattered rows
        // cost 2.8x with the old 64 x 32 tile (39 ms vs 14 ms per 16k x 1024
        // x 768 batch on the M1 Pro); with 64 x 128 reading by id is still
        // slower (9.94 ms vs 8.81 + 0.64 ms gather).
        let xs = match sel.ids {
            Some(ids) => {
                let mut slot = self.gathered.take();
                let g = Self::scratch(&mut slot, self, r.len() * d)?;
                self.gathered = slot;
                let gp = GatherP {
                    r0: u32::try_from(r.start)?,
                    rows: u32::try_from(r.len())?,
                    d: u32::try_from(d)?,
                };
                let gpipe = self.ops.gather.clone();
                self.dispatch(
                    "gather",
                    &gpipe,
                    &[x.dev, ids, &g],
                    bytes(&gp),
                    size(d.div_ceil(64), r.len().div_ceil(4)),
                    size(64, 4),
                )?;
                p.xr0 = 0;
                g
            }
            None => x.dev.clone(),
        };
        // Partial results of the centroid parts (unused when parts == 1).
        let n_part = if parts > 1 { parts * r.len() } else { 1 };
        let mut slot = self.part_b.take();
        let pb = Self::scratch(&mut slot, self, n_part)?;
        self.part_b = slot;
        let mut slot = self.part_j.take();
        let pj = Self::scratch(&mut slot, self, n_part)?;
        self.part_j = slot;
        let bufs = [&xs, &xs, centroids, cn, &*labels, &*scores, &pb, &pj];
        self.dispatch(
            "top1",
            &pipe,
            &bufs,
            bytes(&p),
            size(r.len().div_ceil(t.0), parts),
            size(t.3 * t.4 * 32, 1),
        )?;
        if parts > 1 {
            let comb = self.top1_combine.clone();
            self.dispatch(
                "combine",
                &comb,
                &bufs,
                bytes(&p),
                size(r.len().div_ceil(256), 1),
                size(256, 1),
            )?;
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
        let r = self.knn(queries, data, d, kk, exclude_self);
        if r.is_err() {
            // Queued work may view `queries`/`data` in place: finish it before
            // the caller can free them.
            let _ = self.sync();
        }
        r
    }

    fn scatter_add(
        &mut self,
        x: Mat<'_, MBuf>,
        sel: Select<'_, MBuf>,
        labels: &MBuf,
        k: usize,
        sums: &mut MBuf,
        counts: &mut MBuf,
        acc: Option<&mut MBuf>,
    ) -> Result<()> {
        let d = x.d;
        check_select(&sel, sel.ids.map(|i| i.len), x.dev.len / d)?;
        ensure!(sums.len == k * d && counts.len == k, "sums k x d, counts k");
        ensure!(sel.rows.end <= labels.len, "labels too short");
        if let Some(a) = &acc {
            ensure!(a.len == k * d, "acc must be k x d");
        }
        let m = sel.rows.len();
        let mut start_slot = self.start.take();
        let start = Self::scratch(&mut start_slot, self, k + 1)?;
        self.start = start_slot;
        let mut order_slot = self.order.take();
        let order = Self::scratch(&mut order_slot, self, m.max(1))?;
        self.order = order_slot;
        let p = ScatterP {
            lo: u32::try_from(sel.rows.start)?,
            m: u32::try_from(m)?,
            k: u32::try_from(k)?,
            d: u32::try_from(d)?,
            indexed: sel.ids.is_some() as u32,
            acc: acc.is_some() as u32,
        };
        let ids = sel.ids.unwrap_or(x.dev);
        let accb = acc.map(|a| &*a).unwrap_or(sums).clone();
        let nb = m.div_ceil(256).max(1);
        let mut slot = self.rank.take();
        let rank = Self::scratch(&mut slot, self, m.max(1))?;
        self.rank = slot;
        let mut slot = self.bcnt.take();
        let bcnt = Self::scratch(&mut slot, self, nb * k)?;
        self.bcnt = slot;
        let mut slot = self.boff.take();
        let boff = Self::scratch(&mut slot, self, nb * k)?;
        self.boff = slot;
        let o = &self.ops;
        let (block, scan, place, sum) = (
            o.block.clone(),
            o.scan.clone(),
            o.place.clone(),
            o.sum.clone(),
        );
        let pb = bytes(&p);
        if m > 0 {
            self.dispatch(
                "block",
                &block,
                &[labels, &rank, &bcnt],
                pb,
                size(nb, 1),
                size(256, 1),
            )?;
        }
        self.dispatch(
            "scan",
            &scan,
            &[&bcnt, &boff, counts, &start],
            pb,
            size(1, 1),
            size(1024, 1),
        )?;
        if m > 0 {
            self.dispatch(
                "place",
                &place,
                &[labels, &rank, &boff, &order],
                pb,
                size(m.div_ceil(256), 1),
                size(256, 1),
            )?;
        }
        self.dispatch(
            "sum",
            &sum,
            &[x.dev, ids, &start, &order, sums, &accb],
            pb,
            size(d.div_ceil(64), k.div_ceil(4)),
            size(64, 4),
        )
    }

    fn row_norms(&mut self, m: &MBuf, rows: usize, d: usize, out: &mut MBuf) -> Result<()> {
        ensure!(m.len >= rows * d && out.len >= rows, "shapes");
        if rows == 0 {
            return Ok(());
        }
        let p = RowsP {
            rows: u32::try_from(rows)?,
            d: u32::try_from(d)?,
        };
        // One simdgroup per row, 8 rows per threadgroup (see OPS_SOURCE).
        let pipe = self.ops.row_norms.clone();
        self.dispatch(
            "norms",
            &pipe,
            &[m, out],
            bytes(&p),
            size(rows.div_ceil(8), 1),
            size(256, 1),
        )
    }

    fn normalize_rows(&mut self, src: &MBuf, rows: usize, d: usize, dst: &mut MBuf) -> Result<()> {
        ensure!(src.len >= rows * d && dst.len >= rows * d, "shapes");
        if rows == 0 {
            return Ok(());
        }
        let p = RowsP {
            rows: u32::try_from(rows)?,
            d: u32::try_from(d)?,
        };
        // One simdgroup per row, 8 rows per threadgroup (see OPS_SOURCE).
        let pipe = self.ops.normalize.clone();
        self.dispatch(
            "normalize",
            &pipe,
            &[src, dst],
            bytes(&p),
            size(rows.div_ceil(8), 1),
            size(256, 1),
        )
    }

    fn split(
        &mut self,
        positions: &mut MBuf,
        usage: &mut MBuf,
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
        let pairs = self.buf_from(&flat)?;
        let p = SplitP {
            npairs: plan.len() as u32,
            d: u32::try_from(d)?,
            eps: emeans_core::recipe::SPLIT_EPSILON as f32,
        };
        let pipe = self.ops.split.clone();
        self.dispatch(
            "split",
            &pipe,
            &[positions, usage, &pairs],
            bytes(&p),
            size(plan.len(), 1),
            size(256, 1),
        )
    }

    fn ema_update(
        &mut self,
        positions: &mut MBuf,
        usage: &mut MBuf,
        sums: &MBuf,
        counts: &MBuf,
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
        let pipe = self.ops.ema.clone();
        self.dispatch(
            "ema",
            &pipe,
            &[positions, usage, sums, counts],
            bytes(&p),
            size(k, 1),
            size(256, 1),
        )
    }
}

impl Assigner for MetalCore {
    fn name(&self) -> String {
        Device::name(self)
    }

    fn bind_data(&mut self, data: &[f32], d: usize) -> Result<()> {
        // Always a fresh copy, also for the pointer and length already bound:
        // nothing makes the caller keep `data` alive or unchanged
        // (`bind_shared` does), so the same address may hold other data now.
        self.unbind_data();
        let b = self.buf_from(data)?;
        self.make_resident(&b, d, data.len())?;
        self.bound = Some((data.as_ptr() as usize, data.len(), b));
        Ok(())
    }

    fn bind_shared(&mut self, data: emeans_core::assign::SharedData, d: usize) -> Result<()> {
        // One borrow of the data for the whole bind: the view, the key and the
        // owner kept below all refer to the same memory.
        let x: &[f32] = (*data).as_ref();
        let (p, l) = (x.as_ptr() as usize, x.len());
        if self.bound.as_ref().is_some_and(|(bp, bl, b)| {
            *bp == p
                && *bl == l
                && b.keep
                    .as_ref()
                    .is_some_and(|k| std::sync::Arc::ptr_eq(k, &data))
        }) {
            return Ok(());
        }
        self.unbind_data();
        // No copy: the buffer holds the Arc (`keep`), so the pages stay alive
        // and unchanged for as long as the buffer or a clone of it exists, and
        // `owners` holds it while queued work reads them. A copy when Metal
        // refuses the pages; the copy holds the Arc too, so the address stays
        // pinned and `Device::bind` may serve it.
        let mut b = match self.buf_nocopy(x) {
            Some(b) => b,
            None => self.buf_from(x)?,
        };
        b.keep = Some(data);
        self.make_resident(&b, d, l)?;
        self.bound = Some((p, l, b));
        Ok(())
    }

    fn unbind_data(&mut self) {
        // The bound buffer may view shared data: wait for the work that reads
        // it, then let the engine's handle go (a clone of the view keeps its own).
        if self.bound.is_some()
            && let Err(e) = self.sync()
        {
            eprintln!("metal: {e:#}");
        }
        self.bound = None;
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
