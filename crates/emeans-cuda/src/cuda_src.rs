//! CUDA C sources of the CUDA backend, compiled at runtime with NVRTC.

/// Fused score + argmax, FP32 operands with FP32 accumulation (exact path).
///
/// The idea of faiss's Triton `_assign_kernel`: a block owns `BM` rows, walks
/// the centroids in tiles of `BN`, accumulates the `BM x BN` dot products over
/// `d` in registers (tiles of `BK` staged through double-buffered shared memory),
/// keeps each row's running best in registers, and never writes the `rows x k`
/// matrix. 256 threads (16 x 16); each owns an `(BM/16) x (BN/16)` sub-tile
/// split in 4-wide strips 64 apart, so shared-memory float4 reads are
/// conflict-free. Ties keep the lower centroid id.
pub const TOP1_SOURCE: &str = r#"
#define FULLMASK 0xffffffffu
#define EM_INF __int_as_float(0x7f800000)
#define NOROW 0xffffffffu

__device__ __forceinline__ bool better(bool ip, float a, unsigned ja, float b, unsigned jb) {
    return ip ? (a > b || (a == b && ja < jb)) : (a < b || (a == b && ja < jb));
}

struct Top1P {
    unsigned rows, k, d, ip, indexed, r0, xr0;
    unsigned chunk;   // centroids per part (multiple of BN); part = blockIdx.y
    unsigned parts;   // 1: write labels/scores; > 1: write partial bests to pv/pj (+ pxn)
    unsigned list;    // 1: the rows are positions lst[0 .. *cnt) (relative to r0)
};

template <int BM, int BN, int BK, bool VEC>
__device__ __forceinline__ void top1_body(
    const float* __restrict__ x, const unsigned* __restrict__ idx,
    const float* __restrict__ c, const float* __restrict__ cn,
    unsigned* __restrict__ labels, float* __restrict__ scores,
    float* __restrict__ pv, unsigned* __restrict__ pj, float* __restrict__ pxn,
    const unsigned* __restrict__ lst, const unsigned* __restrict__ cnt, const Top1P p)
{
    constexpr int NI = BM / 16;           // rows per thread
    constexpr int NJ = BN / 16;           // columns per thread
    constexpr int XQ = BM * BK / 4 / 256; // float4 loads per thread, rows
    constexpr int CQ = BN * BK / 4 / 256; // float4 loads per thread, centroids
    static_assert(XQ >= 1 && CQ >= 1, "tile too small");
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];
    __shared__ unsigned rsrc[BM];
    __shared__ unsigned rpos[BM];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const unsigned row0 = blockIdx.x * BM;
    const unsigned d = p.d, k = p.k;
    const bool ip = p.ip != 0;
    const unsigned nrows = p.list ? *cnt : p.rows;
    if (row0 >= nrows) return;

    for (int r = tid; r < BM; r += 256) {
        const unsigned gr = row0 + r;
        const unsigned pos = gr < nrows ? (p.list ? lst[gr] : gr) : NOROW;
        rpos[r] = pos;
        rsrc[r] = pos != NOROW ? (p.indexed ? idx[p.r0 + pos] : p.xr0 + pos) : NOROW;
    }
    __syncthreads();

    float best[NI];
    unsigned bj[NI];
    #pragma unroll
    for (int i = 0; i < NI; ++i) { best[i] = ip ? -EM_INF : EM_INF; bj[i] = 0; }

    float4 xr[XQ], cr[CQ];
    const int ntile = (d + BK - 1) / BK;

    auto load = [&](unsigned n0, unsigned k0) {
        #pragma unroll
        for (int q = 0; q < XQ; ++q) {
            const int e = tid + 256 * q;
            const int lr = e / (BK / 4), lc = (e % (BK / 4)) * 4;
            const unsigned src = rsrc[lr];
            const unsigned col = k0 + lc;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (src != NOROW) {
                const float* px = x + (unsigned long long)src * d + col;
                if (VEC) { if (col < d) v = *reinterpret_cast<const float4*>(px); }
                else {
                    if (col < d) v.x = px[0];
                    if (col + 1 < d) v.y = px[1];
                    if (col + 2 < d) v.z = px[2];
                    if (col + 3 < d) v.w = px[3];
                }
            }
            xr[q] = v;
        }
        #pragma unroll
        for (int q = 0; q < CQ; ++q) {
            const int e = tid + 256 * q;
            const int lr = e / (BK / 4), lc = (e % (BK / 4)) * 4;
            const unsigned gj = n0 + lr;
            const unsigned col = k0 + lc;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gj < k) {
                const float* pc = c + (unsigned long long)gj * d + col;
                if (VEC) { if (col < d) v = *reinterpret_cast<const float4*>(pc); }
                else {
                    if (col < d) v.x = pc[0];
                    if (col + 1 < d) v.y = pc[1];
                    if (col + 2 < d) v.z = pc[2];
                    if (col + 3 < d) v.w = pc[3];
                }
            }
            cr[q] = v;
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int q = 0; q < XQ; ++q) {
            const int e = tid + 256 * q;
            const int lr = e / (BK / 4), lc = (e % (BK / 4)) * 4;
            As[buf][lc + 0][lr] = xr[q].x;
            As[buf][lc + 1][lr] = xr[q].y;
            As[buf][lc + 2][lr] = xr[q].z;
            As[buf][lc + 3][lr] = xr[q].w;
        }
        #pragma unroll
        for (int q = 0; q < CQ; ++q) {
            const int e = tid + 256 * q;
            const int lr = e / (BK / 4), lc = (e % (BK / 4)) * 4;
            Bs[buf][lc + 0][lr] = cr[q].x;
            Bs[buf][lc + 1][lr] = cr[q].y;
            Bs[buf][lc + 2][lr] = cr[q].z;
            Bs[buf][lc + 3][lr] = cr[q].w;
        }
    };

    const unsigned c_lo = blockIdx.y * p.chunk;
    const unsigned c_hi = (c_lo + p.chunk < k) ? c_lo + p.chunk : k;
    for (unsigned n0 = c_lo; n0 < c_hi; n0 += BN) {
        float acc[NI][NJ];
        #pragma unroll
        for (int i = 0; i < NI; ++i)
            #pragma unroll
            for (int j = 0; j < NJ; ++j) acc[i][j] = 0.f;

        load(n0, 0);
        store(0);
        __syncthreads();
        for (int t = 0; t < ntile; ++t) {
            const int buf = t & 1;
            if (t + 1 < ntile) load(n0, (t + 1) * BK);
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float a[NI], b[NJ];
                #pragma unroll
                for (int s = 0; s < NI / 4; ++s) {
                    const float4 v = *reinterpret_cast<const float4*>(&As[buf][kk][s * 64 + ty * 4]);
                    a[s * 4 + 0] = v.x; a[s * 4 + 1] = v.y; a[s * 4 + 2] = v.z; a[s * 4 + 3] = v.w;
                }
                #pragma unroll
                for (int s = 0; s < NJ / 4; ++s) {
                    const float4 v = *reinterpret_cast<const float4*>(&Bs[buf][kk][s * 64 + tx * 4]);
                    b[s * 4 + 0] = v.x; b[s * 4 + 1] = v.y; b[s * 4 + 2] = v.z; b[s * 4 + 3] = v.w;
                }
                #pragma unroll
                for (int i = 0; i < NI; ++i)
                    #pragma unroll
                    for (int j = 0; j < NJ; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
            if (t + 1 < ntile) store(buf ^ 1);
            __syncthreads();
        }
        // Running best over this thread's columns.
        float nrm[NJ];
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const unsigned cj = n0 + (j / 4) * 64 + tx * 4 + (j % 4);
            nrm[j] = (!ip && cj < k) ? cn[cj] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < NI; ++i)
            #pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const unsigned cj = n0 + (j / 4) * 64 + tx * 4 + (j % 4);
                const float v = ip ? acc[i][j] : fmaf(-2.0f, acc[i][j], nrm[j]);
                if (cj < k && better(ip, v, cj, best[i], bj[i])) { best[i] = v; bj[i] = cj; }
            }
    }
    // Combine the 16 threads (one half-warp) that share each row.
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        float v = best[i];
        unsigned j = bj[i];
        #pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            const float ov = __shfl_xor_sync(FULLMASK, v, off);
            const unsigned oj = __shfl_xor_sync(FULLMASK, j, off);
            if (better(ip, ov, oj, v, j)) { v = ov; j = oj; }
        }
        const int lr = (i / 4) * 64 + ty * 4 + (i % 4);
        const unsigned src = rsrc[lr];
        float xn = 0.f;
        if (!ip && src != NOROW) {
            const float* px = x + (unsigned long long)src * d;
            for (unsigned t = tx; t < d; t += 16) xn = fmaf(px[t], px[t], xn);
        }
        if (!ip) {
            #pragma unroll
            for (int off = 8; off >= 1; off >>= 1) xn += __shfl_xor_sync(FULLMASK, xn, off);
        }
        if (tx == 0 && src != NOROW) {
            if (p.parts == 1) {
                const unsigned pos = p.r0 + rpos[lr];
                labels[pos] = j;
                scores[pos] = ip ? v : fmaxf(xn + v, 0.f);
            } else {
                const unsigned long long q = (unsigned long long)blockIdx.y * p.rows + row0 + lr;
                pv[q] = v;
                pj[q] = j;
                if (blockIdx.y == 0) pxn[row0 + lr] = xn;
            }
        }
    }
}

#define TOP1(NAME, BM, BN, VEC) \
extern "C" __global__ void __launch_bounds__(256) NAME( \
    const float* __restrict__ x, const unsigned* __restrict__ idx, const float* __restrict__ c, \
    const float* __restrict__ cn, unsigned* __restrict__ labels, float* __restrict__ scores, \
    float* __restrict__ pv, unsigned* __restrict__ pj, float* __restrict__ pxn, \
    const unsigned* __restrict__ lst, const unsigned* __restrict__ cnt, \
    const Top1P p) { top1_body<BM, BN, 16, VEC>(x, idx, c, cn, labels, scores, pv, pj, pxn, lst, cnt, p); }

TOP1(top1_128_128_v4, 128, 128, true)
TOP1(top1_128_64_v4, 128, 64, true)
TOP1(top1_64_128_v4, 64, 128, true)
TOP1(top1_64_64_v4, 64, 64, true)
TOP1(top1_128_128_s, 128, 128, false)
TOP1(top1_64_64_s, 64, 64, false)

// Combine the parts' bests per row (lower id on ties, the order one pass uses).
// In list mode the rows are positions lst[0 .. *cnt) and `rows` is the
// partials' stride.
extern "C" __global__ void top1_combine(const float* __restrict__ pv, const unsigned* __restrict__ pj,
    const float* __restrict__ pxn, unsigned* __restrict__ labels, float* __restrict__ scores,
    unsigned rows, unsigned parts, unsigned ip, unsigned r0,
    const unsigned* __restrict__ lst, const unsigned* __restrict__ cnt, unsigned list)
{
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= (list ? *cnt : rows)) return;
    const unsigned pos = list ? lst[r] : r;
    float v = pv[r];
    unsigned j = pj[r];
    for (unsigned q = 1; q < parts; ++q) {
        const float ov = pv[(unsigned long long)q * rows + r];
        const unsigned oj = pj[(unsigned long long)q * rows + r];
        if (better(ip != 0, ov, oj, v, j)) { v = ov; j = oj; }
    }
    labels[r0 + pos] = j;
    scores[r0 + pos] = ip ? v : fmaxf(pxn[r] + v, 0.f);
}
"#;

/// Exact ops (compiled with `--fmad=false`, IEEE division and square root):
/// gather, norms, scatter-add, EMA, and a simple top-k.
pub const OPS_SOURCE: &str = r#"
#define FULLMASK 0xffffffffu
#define EM_INF __int_as_float(0x7f800000)
__device__ __forceinline__ unsigned umin(unsigned a, unsigned b) { return a < b ? a : b; }

extern "C" __global__ void gather_rows(const float* __restrict__ x, const unsigned* __restrict__ idx,
    float* __restrict__ out, unsigned r0, unsigned rows, unsigned d)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned r = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= d || r >= rows) return;
    out[(unsigned long long)r * d + t] = x[(unsigned long long)idx[r0 + r] * d + t];
}

// Row norms and normalization, bit-equal to the CPU. One warp (block of 32)
// per row. Squares and divides are elementwise, so the lanes compute them in
// parallel; only the sum must follow the CPU's order (a sequential f32 sum), so
// lane 0 adds the staged squares in column order, 1024 columns at a time. The
// max is exact in any order.
extern "C" __global__ void row_norms(const float* __restrict__ m, float* __restrict__ out,
    unsigned rows, unsigned d)
{
    __shared__ float sq[1024];
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    if (row >= rows) return;
    const float* r = m + (unsigned long long)row * d;
    float s = 0.f;
    for (unsigned t0 = 0; t0 < d; t0 += 1024) {
        const unsigned n = umin(1024u, d - t0);
        for (unsigned i = lane; i < n; i += 32) sq[i] = r[t0 + i] * r[t0 + i];
        __syncwarp();
        if (lane == 0) for (unsigned i = 0; i < n; ++i) s = s + sq[i];
        __syncwarp();
    }
    if (lane == 0) out[row] = s;
}

// faiss normalize_into, same order as the CPU: divide by the max magnitude,
// then by the norm of the scaled row.
extern "C" __global__ void normalize_rows(const float* __restrict__ m, float* __restrict__ out,
    unsigned rows, unsigned d)
{
    __shared__ float sq[1024];
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    if (row >= rows) return;
    const float* r = m + (unsigned long long)row * d;
    float* o = out + (unsigned long long)row * d;
    float mx = 0.f;
    for (unsigned t = lane; t < d; t += 32) mx = fmaxf(mx, fabsf(r[t]));
    for (int off = 16; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, off));
    const float s = mx > 0.f ? mx : 1.f;
    float ss = 0.f;
    for (unsigned t0 = 0; t0 < d; t0 += 1024) {
        const unsigned n = umin(1024u, d - t0);
        for (unsigned i = lane; i < n; i += 32) { const float v = r[t0 + i] / s; sq[i] = v * v; }
        __syncwarp();
        if (lane == 0) for (unsigned i = 0; i < n; ++i) ss = ss + sq[i];
        __syncwarp();
    }
    const float nrm = __shfl_sync(FULLMASK, sqrtf(ss), 0);
    const float s2 = nrm > 0.f ? nrm : 1.f;
    for (unsigned t = lane; t < d; t += 32) o[t] = (r[t] / s) / s2;
}

// One block per (donor, dead) pair: apply_splits on the device.
extern "C" __global__ void split_pairs(float* __restrict__ pos, float* __restrict__ usage,
    const unsigned* __restrict__ pairs, unsigned d, float eps)
{
    __shared__ int differ;
    if (threadIdx.x == 0) differ = 0;
    __syncthreads();
    const unsigned long long s = pairs[2 * blockIdx.x], t = pairs[2 * blockIdx.x + 1];
    for (unsigned i = threadIdx.x; i < d; i += blockDim.x) {
        const float c = pos[s * d + i];
        const float off = (i % 2 == 0) ? eps : -eps;
        const float dn = c * (1.f - off);
        const float ch = c * (1.f + off);
        if (dn != ch) differ = 1;
        pos[s * d + i] = dn;
        pos[t * d + i] = ch;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (!differ) {
            const float c0 = pos[t * d];
            const float up = nextafterf(c0, EM_INF);
            pos[t * d] = (fabsf(up) <= 3.402823466e38f) ? up : nextafterf(c0, -EM_INF);
        }
        const float half = usage[s] / 2.f;
        usage[s] = half;
        usage[t] = half;
    }
}

struct ScatterP { unsigned lo, m, k, d, indexed, acc, segs; };

// Stable counting sort of the batch's positions by label, in three passes over
// `segs` segments of SEG positions. A warp owns SG consecutive clusters and one
// segment; lane g < SG holds cluster j0 + g's register. Labels are read 32 x 4
// at a time (four loads in flight), and the per-cluster ballots run only for
// the chunks that hit one of the warp's clusters.
#define SG 8u
#define SEG 2048u

// cseg[s * k + j] = positions of cluster j in segment s.
extern "C" __global__ void scatter_count(const unsigned* __restrict__ labels,
    unsigned* __restrict__ cseg, const ScatterP p)
{
    const unsigned w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const unsigned lane = threadIdx.x & 31;
    const unsigned j0 = w * SG, s = blockIdx.y;
    if (j0 >= p.k) return;
    const unsigned* lab = labels + p.lo;
    const unsigned e = umin(p.m, (s + 1) * SEG);
    unsigned c = 0;
    for (unsigned b = s * SEG; b < e; b += 128) {
        unsigned l[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const unsigned i = b + 32 * u + lane;
            l[u] = i < e ? lab[i] - j0 : SG;
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (__ballot_sync(FULLMASK, l[u] < SG) == 0) continue;
            #pragma unroll
            for (unsigned g = 0; g < SG; ++g) {
                const unsigned hits = __popc(__ballot_sync(FULLMASK, l[u] == g));
                if (lane == g) c += hits;
            }
        }
    }
    if (lane < SG && j0 + lane < p.k) cseg[(unsigned long long)s * p.k + j0 + lane] = c;
}

// counts[j] = total of cluster j; start[j] = exclusive prefix over clusters;
// cseg[s * k + j] becomes the first output slot of cluster j in segment s.
// One block of 1024 threads.
extern "C" __global__ void scan_counts(unsigned* __restrict__ cseg, unsigned* __restrict__ counts,
    unsigned* __restrict__ start, const ScatterP p)
{
    __shared__ unsigned part[1024];
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    const unsigned per = (p.k + nt - 1) / nt;
    const unsigned a = umin(tid * per, p.k), b = umin(a + per, p.k);
    unsigned local = 0;
    for (unsigned j = a; j < b; ++j) {
        unsigned t = 0;
        for (unsigned s = 0; s < p.segs; ++s) t += cseg[(unsigned long long)s * p.k + j];
        counts[j] = t;
        local += t;
    }
    part[tid] = local;
    __syncthreads();
    // Hillis-Steele inclusive scan over the thread totals.
    for (unsigned off = 1; off < nt; off <<= 1) {
        const unsigned v = tid >= off ? part[tid - off] : 0u;
        __syncthreads();
        part[tid] += v;
        __syncthreads();
    }
    unsigned run = part[tid] - local;
    for (unsigned j = a; j < b; ++j) {
        start[j] = run;
        for (unsigned s = 0; s < p.segs; ++s) {
            const unsigned long long q = (unsigned long long)s * p.k + j;
            const unsigned c = cseg[q];
            cseg[q] = run;
            run += c;
        }
    }
    if (tid == nt - 1) start[p.k] = run;
}

// order[slot] = the data row of each position, grouped by cluster, positions in
// order within a cluster.
extern "C" __global__ void scatter_place(const unsigned* __restrict__ labels,
    const unsigned* __restrict__ idx, const unsigned* __restrict__ cseg,
    unsigned* __restrict__ order, const ScatterP p)
{
    const unsigned w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const unsigned lane = threadIdx.x & 31;
    const unsigned j0 = w * SG, s = blockIdx.y;
    if (j0 >= p.k) return;
    const unsigned* lab = labels + p.lo;
    const unsigned e = umin(p.m, (s + 1) * SEG);
    unsigned pos = (lane < SG && j0 + lane < p.k) ? cseg[(unsigned long long)s * p.k + j0 + lane] : 0u;
    const unsigned lt = (1u << lane) - 1u;
    for (unsigned b = s * SEG; b < e; b += 128) {
        unsigned l[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const unsigned i = b + 32 * u + lane;
            l[u] = i < e ? lab[i] - j0 : SG;
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (__ballot_sync(FULLMASK, l[u] < SG) == 0) continue;
            const unsigned i = b + 32 * u + lane;
            #pragma unroll
            for (unsigned g = 0; g < SG; ++g) {
                const unsigned mask = __ballot_sync(FULLMASK, l[u] == g);
                const unsigned base = __shfl_sync(FULLMASK, pos, g);
                if (l[u] == g)
                    order[base + __popc(mask & lt)] = p.indexed ? idx[p.lo + i] : p.lo + i;
                if (lane == g) pos += __popc(mask);
            }
        }
    }
}

// Thread (t, j): sums[j, t] = rows of cluster j added in position order. Eight
// row loads are issued before the eight adds, which stay in order.
extern "C" __global__ void scatter_sum(const float* __restrict__ x,
    const unsigned* __restrict__ start, const unsigned* __restrict__ order,
    float* __restrict__ sums, float* __restrict__ acc, const ScatterP p)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= p.d || j >= p.k) return;
    const unsigned e = start[j + 1];
    unsigned q = start[j];
    float s = 0.f;
    for (; q + 8 <= e; q += 8) {
        float v[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = x[(unsigned long long)order[q + u] * p.d + t];
        #pragma unroll
        for (int u = 0; u < 8; ++u) s += v[u];
    }
    for (; q < e; ++q) s += x[(unsigned long long)order[q] * p.d + t];
    sums[(unsigned long long)j * p.d + t] = s;
    if (p.acc) acc[(unsigned long long)j * p.d + t] += s;
}

struct EmaP { unsigned k, d; float alpha, decay; unsigned replace; };

// Positions first (they read the old masses), then masses.
extern "C" __global__ void ema_positions(float* __restrict__ pos, const float* __restrict__ usage,
    const float* __restrict__ sums, const unsigned* __restrict__ counts, const EmaP p)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= p.d || j >= p.k) return;
    const unsigned cnt = counts[j];
    if (cnt == 0) return;
    const float c = (float)cnt;
    const float retained = p.replace ? 0.f : usage[j] * p.decay;
    const float step = p.alpha * c;
    const float mass = retained + step;
    const float w = step / mass;
    const unsigned long long i = (unsigned long long)j * p.d + t;
    const float mean = sums[i] / c;
    const float cur = pos[i];
    pos[i] = (w == 1.f) ? mean : cur + w * (mean - cur);
}

extern "C" __global__ void ema_usage(float* __restrict__ usage, const unsigned* __restrict__ counts,
    const EmaP p)
{
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p.k) return;
    float u = usage[j];
    if (!p.replace) u = u * p.decay;
    const unsigned cnt = counts[j];
    if (cnt > 0) u = (p.replace ? 0.f : u) + p.alpha * (float)cnt;
    usage[j] = u;
}

// Top-k by inner product, one warp per query: each lane keeps the best kk of
// the data rows it scans (rows lane, lane + 32, ...), then lane 0 merges the
// 32 lists. Correct, not tuned (dedup on CUDA is not a hot path).
#define KMAX 16
__device__ __forceinline__ bool kbetter(float a, unsigned ja, float b, unsigned jb) {
    return a > b || (a == b && ja < jb);
}
extern "C" __global__ void topk_warp(const float* __restrict__ q, const float* __restrict__ data,
    unsigned* __restrict__ out_ids, float* __restrict__ out_sims,
    unsigned m, unsigned n, unsigned d, unsigned kk, unsigned exclude)
{
    __shared__ float ss[4][32 * KMAX];
    __shared__ unsigned sj[4][32 * KMAX];
    const unsigned w = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned qi = blockIdx.x * 4 + w;
    float bs[KMAX];
    unsigned bj[KMAX];
    for (int t = 0; t < KMAX; ++t) { bs[t] = -EM_INF; bj[t] = 0xffffffffu; }
    if (qi < m) {
        const float* qr = q + (unsigned long long)qi * d;
        for (unsigned r = lane; r < n; r += 32) {
            if (exclude && r == qi) continue;
            const float* dr = data + (unsigned long long)r * d;
            float s = 0.f;
            for (unsigned t = 0; t < d; ++t) s += qr[t] * dr[t];
            unsigned wslot = 0;
            for (unsigned t = 1; t < kk; ++t) if (kbetter(bs[wslot], bj[wslot], bs[t], bj[t])) wslot = t;
            if (kbetter(s, r, bs[wslot], bj[wslot])) { bs[wslot] = s; bj[wslot] = r; }
        }
    }
    for (unsigned t = 0; t < kk; ++t) { ss[w][lane * KMAX + t] = bs[t]; sj[w][lane * KMAX + t] = bj[t]; }
    __syncwarp();
    if (lane == 0 && qi < m) {
        for (unsigned t = 0; t < kk; ++t) {
            unsigned bi = 0xffffffffu;
            for (unsigned e = 0; e < 32 * KMAX; ++e) {
                if ((e % KMAX) >= kk || sj[w][e] == 0xfffffffeu) continue;
                if (bi == 0xffffffffu || kbetter(ss[w][e], sj[w][e], ss[w][bi], sj[w][bi])) bi = e;
            }
            out_ids[(unsigned long long)qi * kk + t] = sj[w][bi];
            out_sims[(unsigned long long)qi * kk + t] = ss[w][bi];
            sj[w][bi] = 0xfffffffeu;
        }
    }
}
"#;

/// Opt-in BF16 top-1 (not exact: operands rounded to bfloat16, products
/// accumulated in FP32 on tensor cores with `mma.sync.m16n8k16`; faiss's
/// default CUDA path does the same). Inputs are pre-converted BF16 rows
/// (`to_bf16`, `gather_bf16`), `d % 8 == 0`. A block owns 128 rows and walks
/// the centroids in tiles of 128; 8 warps in a 2 x 4 grid each hold a
/// 64 x 32 accumulator tile (4 x 4 mma tiles), K steps of 32 staged through
/// double-buffered, padded shared memory.
pub const TOP1_BF16_SOURCE: &str = r#"
#define FULLMASK 0xffffffffu
#define EM_INF __int_as_float(0x7f800000)

__device__ __forceinline__ bool better(bool ip, float a, unsigned ja, float b, unsigned jb) {
    return ip ? (a > b || (a == b && ja < jb)) : (a < b || (a == b && ja < jb));
}

struct Bf16P { unsigned rows, k, d, ip, r0, xr0, indexed; };

#define BM 128
#define BN 128
#define BK 32
#define LDS (BK + 8)
#define NOROW 0xffffffffu

// x: bf16 rows, row of position r = idx[r0 + r] (indexed) or xr0 + r; c: k x d
// bf16; xn (per x row) / cn: fp32 squared norms (L2).
extern "C" __global__ void __launch_bounds__(256) top1_bf16(
    const unsigned short* __restrict__ x, const unsigned* __restrict__ idx,
    const unsigned short* __restrict__ c,
    const float* __restrict__ xn, const float* __restrict__ cn,
    unsigned* __restrict__ labels, float* __restrict__ scores, const Bf16P p)
{
    __shared__ __align__(16) unsigned short As[2][BM * LDS];
    __shared__ __align__(16) unsigned short Bs[2][BN * LDS];
    __shared__ float tb[4][BM];
    __shared__ unsigned tj[4][BM];
    __shared__ unsigned rsrc[BM];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wr = warp >> 2, wc = warp & 3;   // 2 x 4 warps
    const int g = lane >> 2, tig = lane & 3;
    const unsigned row0 = blockIdx.x * BM;
    const unsigned d = p.d, k = p.k;
    const bool ip = p.ip != 0;

    for (int r = tid; r < BM; r += 256) {
        const unsigned gr = row0 + r;
        rsrc[r] = gr < p.rows ? (p.indexed ? idx[p.r0 + gr] : p.xr0 + gr) : NOROW;
    }
    __syncthreads();

    float best[8];
    unsigned bj[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { best[i] = ip ? -EM_INF : EM_INF; bj[i] = 0; }

    // Each thread moves 2 x 16 bytes of x and of c per K step: 128 rows x 32 bf16 = 512 uint4.
    uint4 xr[2], cr[2];
    const int ntile = (d + BK - 1) / BK;
    auto load = [&](unsigned n0, unsigned k0) {
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int e = tid + 256 * q;
            const int lr = e >> 2, lc = (e & 3) * 8;
            const unsigned col = k0 + lc;
            uint4 v = make_uint4(0, 0, 0, 0);
            const unsigned src = rsrc[lr];
            if (src != NOROW && col < d) v = *reinterpret_cast<const uint4*>(x + (unsigned long long)src * d + col);
            xr[q] = v;
            uint4 w = make_uint4(0, 0, 0, 0);
            if (n0 + lr < k && col < d) w = *reinterpret_cast<const uint4*>(c + (unsigned long long)(n0 + lr) * d + col);
            cr[q] = w;
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int e = tid + 256 * q;
            const int lr = e >> 2, lc = (e & 3) * 8;
            *reinterpret_cast<uint4*>(&As[buf][lr * LDS + lc]) = xr[q];
            *reinterpret_cast<uint4*>(&Bs[buf][lr * LDS + lc]) = cr[q];
        }
    };

    for (unsigned n0 = 0; n0 < k; n0 += BN) {
        float acc[4][4][4];
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                #pragma unroll
                for (int e = 0; e < 4; ++e) acc[a][b][e] = 0.f;
        load(n0, 0);
        store(0);
        __syncthreads();
        for (int t = 0; t < ntile; ++t) {
            const int buf = t & 1;
            if (t + 1 < ntile) load(n0, (t + 1) * BK);
            #pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                unsigned af[4][4], bf[4][2];
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int r = wr * 64 + mi * 16 + g;
                    const unsigned short* a0 = &As[buf][r * LDS + kk + tig * 2];
                    af[mi][0] = *reinterpret_cast<const unsigned*>(a0);
                    af[mi][1] = *reinterpret_cast<const unsigned*>(a0 + 8 * LDS);
                    af[mi][2] = *reinterpret_cast<const unsigned*>(a0 + 8);
                    af[mi][3] = *reinterpret_cast<const unsigned*>(a0 + 8 * LDS + 8);
                }
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int cc = wc * 32 + ni * 8 + g;
                    const unsigned short* b0 = &Bs[buf][cc * LDS + kk + tig * 2];
                    bf[ni][0] = *reinterpret_cast<const unsigned*>(b0);
                    bf[ni][1] = *reinterpret_cast<const unsigned*>(b0 + 8);
                }
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi)
                    #pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        float* cacc = acc[mi][ni];
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(cacc[0]), "+f"(cacc[1]), "+f"(cacc[2]), "+f"(cacc[3])
                            : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]),
                              "r"(bf[ni][0]), "r"(bf[ni][1]));
                    }
            }
            if (t + 1 < ntile) store(buf ^ 1);
            __syncthreads();
        }
        // Rows of this thread: (mi, h) -> wr*64 + mi*16 + g + 8h; columns (ni, e) -> wc*32 + ni*8 + tig*2 + e.
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const unsigned cj = n0 + wc * 32 + ni * 8 + tig * 2 + e;
                const float nrm = (!ip && cj < k) ? cn[cj] : 0.f;
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float dot = acc[mi][ni][h * 2 + e];
                        const float v = ip ? dot : nrm - 2.0f * dot;
                        const int i = mi * 2 + h;
                        if (cj < k && better(ip, v, cj, best[i], bj[i])) { best[i] = v; bj[i] = cj; }
                    }
            }
    }
    // Combine the 4 lanes (tig) that share a row, then the 4 column warps.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = best[i];
        unsigned j = bj[i];
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            const float ov = __shfl_xor_sync(FULLMASK, v, off);
            const unsigned oj = __shfl_xor_sync(FULLMASK, j, off);
            if (better(ip, ov, oj, v, j)) { v = ov; j = oj; }
        }
        if (tig == 0) {
            const int r = wr * 64 + (i >> 1) * 16 + g + 8 * (i & 1);
            tb[wc][r] = v;
            tj[wc][r] = j;
        }
    }
    __syncthreads();
    if (tid < BM && row0 + tid < p.rows) {
        float v = tb[0][tid];
        unsigned j = tj[0][tid];
        for (int w = 1; w < 4; ++w)
            if (better(ip, tb[w][tid], tj[w][tid], v, j)) { v = tb[w][tid]; j = tj[w][tid]; }
        const unsigned pos = p.r0 + row0 + tid;
        labels[pos] = j;
        scores[pos] = ip ? v : fmaxf(xn[rsrc[tid]] + v, 0.f);
    }
}

// ---- v2: cp.async multi-stage pipeline, ldmatrix, swizzled shared memory ----
// A block owns BM = 128 rows (by id) and the centroids [chunk * y, + chunk);
// 8 warps in a 2 x 4 grid, warp tile 64 x TN/4. Shared rows are BK = 32 bf16
// (64 bytes, four 16-byte chunks); chunk c of row r is stored at c ^ ((r >> 1) & 3),
// so the eight rows an ldmatrix phase reads hit eight different bank groups.
// The (centroid tile, K step) loop is flattened so the pipeline runs across tiles.
struct Bf16P2 { unsigned rows, k, d, ip, r0, xr0, indexed, chunk, parts; };

__device__ __forceinline__ void cp16(unsigned short* dst, const unsigned short* src, bool pred) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldm4(unsigned* r, const unsigned short* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ int swz(int r, int c) { return r * 32 + ((c ^ ((r >> 1) & 3)) << 3); }

// SCREEN: instead of the best centroid, keep per row the four best by BF16 key
// (inner product, or 2 dot - |c|^2 for L2: larger is better) and a bound on
// every other key, for screen_verify. Each thread keeps per row slot its two
// best and the largest key it dropped; a block merges its 16 threads per row.
template <int TN, int STAGES, bool SCREEN>
__device__ __forceinline__ void top1_bf16_v2_body(
    const unsigned short* __restrict__ x, const unsigned* __restrict__ idx,
    const unsigned short* __restrict__ c, const float* __restrict__ xn, const float* __restrict__ cn,
    unsigned* __restrict__ labels, float* __restrict__ scores,
    float* __restrict__ pv, unsigned* __restrict__ pj, float* __restrict__ pxn, const Bf16P2 p,
    float* __restrict__ cv, unsigned* __restrict__ cjv, float* __restrict__ ch, unsigned* __restrict__ cnt,
    const float4* __restrict__ xa, const float4* __restrict__ ca)
{
    constexpr int BM2 = 128, BK2 = 32, WN = TN / 4, NI = WN / 8, MI = 4;
    constexpr int STAGE = (BM2 + TN) * BK2;   // halves per stage
    extern __shared__ __align__(16) unsigned short smem[];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wr = warp >> 2, wc = warp & 3;
    const int g = lane >> 2, tig = lane & 3;
    const unsigned row0 = blockIdx.x * BM2;
    const unsigned d = p.d, k = p.k;
    const bool ip = p.ip != 0;

    // This thread's cp.async slots: row (tid >> 2) + 64 q, chunk tid & 3.
    const int lr = tid >> 2, lc = tid & 3;
    // SCREEN: each row's power-of-two scale (the FP16 copy holds x / scale).
    __shared__ float rsx[SCREEN ? BM2 : 1];
    if (SCREEN && tid < BM2) {
        const unsigned gr = row0 + tid;
        rsx[tid] = gr < p.rows ? xa[p.indexed ? idx[p.r0 + gr] : p.xr0 + gr].y : 0.f;
    }
    unsigned src[2];
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const unsigned gr = row0 + lr + 64 * q;
        src[q] = gr < p.rows ? (p.indexed ? idx[p.r0 + gr] : p.xr0 + gr) : NOROW;
    }

    const unsigned c_lo = blockIdx.y * p.chunk;
    const unsigned c_hi = (c_lo + p.chunk < k) ? c_lo + p.chunk : k;
    const int KT = (d + BK2 - 1) / BK2;
    const int NT = c_hi > c_lo ? (c_hi - c_lo + TN - 1) / TN : 0;
    const int total = NT * KT;

    auto issue = [&](int i) {
        if (i < total) {
            const int nt = i / KT, kt = i - nt * KT;
            const unsigned n0 = c_lo + nt * TN, col = kt * BK2 + lc * 8;
            unsigned short* As = smem + (i % STAGES) * STAGE;
            unsigned short* Bs = As + BM2 * BK2;
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const bool ok = src[q] != NOROW && col < d;
                cp16(As + swz(lr + 64 * q, lc), ok ? x + (unsigned long long)src[q] * d + col : x, ok);
            }
            #pragma unroll
            for (int q = 0; q < TN / 64; ++q) {
                const unsigned n = n0 + lr + 64 * q;
                const bool ok = n < c_hi && col < d;
                cp16(Bs + swz(lr + 64 * q, lc), ok ? c + (unsigned long long)n * d + col : c, ok);
            }
        }
        cp_commit();
    };

    if (SCREEN && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) *cnt = 0u;
    float best[8], k2[8], k3[8];
    unsigned bj[8], j2[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        best[i] = (ip || SCREEN) ? -EM_INF : EM_INF; bj[i] = NOROW;
        k2[i] = -EM_INF; j2[i] = NOROW; k3[i] = -EM_INF;
    }
    if (!SCREEN) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) bj[i] = 0;
    }
    float acc[MI][NI][4];
    #pragma unroll
    for (int a = 0; a < MI; ++a)
        #pragma unroll
        for (int b = 0; b < NI; ++b)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[a][b][e] = 0.f;

    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) issue(i);
    for (int i = 0; i < total; ++i) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        issue(i + STAGES - 1);
        const unsigned short* As = smem + (i % STAGES) * STAGE;
        const unsigned short* Bs = As + BM2 * BK2;
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned af[MI][4], bf[NI][2];
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const int r = wr * 64 + mi * 16 + (lane & 15);
                ldm4(af[mi], As + swz(r, ks * 2 + (lane >> 4)));
            }
            #pragma unroll
            for (int nj = 0; nj < NI; nj += 2) {
                const int r = wc * WN + nj * 8 + (lane & 7) + ((lane >> 4) << 3);
                unsigned t[4];
                ldm4(t, Bs + swz(r, ks * 2 + ((lane >> 3) & 1)));
                bf[nj][0] = t[0]; bf[nj][1] = t[1]; bf[nj + 1][0] = t[2]; bf[nj + 1][1] = t[3];
            }
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi)
                #pragma unroll
                for (int ni = 0; ni < NI; ++ni) {
                    float* cacc = acc[mi][ni];
                    if (SCREEN)
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(cacc[0]), "+f"(cacc[1]), "+f"(cacc[2]), "+f"(cacc[3])
                            : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]),
                              "r"(bf[ni][0]), "r"(bf[ni][1]));
                    else
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(cacc[0]), "+f"(cacc[1]), "+f"(cacc[2]), "+f"(cacc[3])
                            : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]),
                              "r"(bf[ni][0]), "r"(bf[ni][1]));
                }
        }
        if ((i % KT) == KT - 1) {
            const unsigned n0 = c_lo + (i / KT) * TN;
            #pragma unroll
            for (int ni = 0; ni < NI; ++ni)
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const unsigned cj = n0 + wc * WN + ni * 8 + tig * 2 + e;
                    const float nrm = (!ip && cj < c_hi) ? cn[cj] : 0.f;
                    const float csc = (SCREEN && cj < c_hi) ? ca[cj].y : 0.f;
                    #pragma unroll
                    for (int mi = 0; mi < MI; ++mi)
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const float dot = acc[mi][ni][h * 2 + e];
                            acc[mi][ni][h * 2 + e] = 0.f;
                            const int r = mi * 2 + h;
                            if (SCREEN) {
                                // Scales are powers of two: kv is the dot of the rows as given.
                                const float kv = dot * (rsx[wr * 64 + mi * 16 + g + 8 * h] * csc);
                                const float key = ip ? kv : fmaf(2.0f, kv, -nrm);
                                if (cj < c_hi) {
                                    if (key > k2[r]) {
                                        k3[r] = fmaxf(k3[r], k2[r]);
                                        if (key > best[r]) { k2[r] = best[r]; j2[r] = bj[r]; best[r] = key; bj[r] = cj; }
                                        else { k2[r] = key; j2[r] = cj; }
                                    } else {
                                        k3[r] = fmaxf(k3[r], key);
                                    }
                                }
                            } else {
                                const float v = ip ? dot : nrm - 2.0f * dot;
                                if (cj < c_hi && better(ip, v, cj, best[r], bj[r])) { best[r] = v; bj[r] = cj; }
                            }
                        }
                }
        }
    }
    cp_wait<0>();
    __syncthreads();
    if (SCREEN) {
        // Row slot r: entries (wc * 4 + tig) * 2 + {0, 1}, dropped bound wc * 4 + tig.
        float* ek = reinterpret_cast<float*>(smem);
        unsigned* ej = reinterpret_cast<unsigned*>(smem) + BM2 * 32;
        float* eh = reinterpret_cast<float*>(smem) + BM2 * 64;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = wr * 64 + (i >> 1) * 16 + g + 8 * (i & 1);
            const int t = wc * 4 + tig;
            ek[r * 32 + 2 * t] = best[i]; ej[r * 32 + 2 * t] = bj[i];
            ek[r * 32 + 2 * t + 1] = k2[i]; ej[r * 32 + 2 * t + 1] = j2[i];
            eh[r * 16 + t] = k3[i];
        }
        __syncthreads();
        const unsigned gr = row0 + tid;
        if (tid < BM2 && gr < p.rows) {
            float* rk = ek + tid * 32;
            unsigned* rj = ej + tid * 32;
            float h = -EM_INF;
            for (int t = 0; t < 16; ++t) h = fmaxf(h, eh[tid * 16 + t]);
            const unsigned long long o = (unsigned long long)blockIdx.y * p.rows + gr;
            for (int q = 0; q < 5; ++q) {
                int bi = -1;
                for (int e = 0; e < 32; ++e) {
                    if (rj[e] == NOROW) continue;
                    if (bi < 0 || rk[e] > rk[bi] || (rk[e] == rk[bi] && rj[e] < rj[bi])) bi = e;
                }
                if (q < 4) {
                    cv[o * 4 + q] = bi < 0 ? -EM_INF : rk[bi];
                    cjv[o * 4 + q] = bi < 0 ? NOROW : rj[bi];
                } else if (bi >= 0) {
                    h = fmaxf(h, rk[bi]);
                }
                if (bi >= 0) rj[bi] = NOROW;
            }
            ch[o] = h;
        }
        return;
    }
    float* tb = reinterpret_cast<float*>(smem);
    unsigned* tj = reinterpret_cast<unsigned*>(smem) + 4 * BM2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = best[i];
        unsigned j = bj[i];
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            const float ov = __shfl_xor_sync(FULLMASK, v, off);
            const unsigned oj = __shfl_xor_sync(FULLMASK, j, off);
            if (better(ip, ov, oj, v, j)) { v = ov; j = oj; }
        }
        if (tig == 0) {
            const int r = wr * 64 + (i >> 1) * 16 + g + 8 * (i & 1);
            tb[wc * BM2 + r] = v;
            tj[wc * BM2 + r] = j;
        }
    }
    __syncthreads();
    const unsigned gr = row0 + tid;
    if (tid < BM2 && gr < p.rows) {
        float v = tb[tid];
        unsigned j = tj[tid];
        for (int w = 1; w < 4; ++w)
            if (better(ip, tb[w * BM2 + tid], tj[w * BM2 + tid], v, j)) { v = tb[w * BM2 + tid]; j = tj[w * BM2 + tid]; }
        const unsigned s = p.indexed ? idx[p.r0 + gr] : p.xr0 + gr;
        const float rn = ip ? 0.f : xn[s];
        if (p.parts == 1) {
            labels[p.r0 + gr] = j;
            scores[p.r0 + gr] = ip ? v : fmaxf(rn + v, 0.f);
        } else {
            const unsigned long long q = (unsigned long long)blockIdx.y * p.rows + gr;
            pv[q] = v;
            pj[q] = j;
            if (blockIdx.y == 0) pxn[gr] = rn;
        }
    }
}

#define TOP1B(NAME, TN, ST) \
extern "C" __global__ void __launch_bounds__(256) NAME( \
    const unsigned short* __restrict__ x, const unsigned* __restrict__ idx, \
    const unsigned short* __restrict__ c, const float* __restrict__ xn, const float* __restrict__ cn, \
    unsigned* __restrict__ labels, float* __restrict__ scores, \
    float* __restrict__ pv, unsigned* __restrict__ pj, float* __restrict__ pxn, const Bf16P2 p) \
{ top1_bf16_v2_body<TN, ST, false>(x, idx, c, xn, cn, labels, scores, pv, pj, pxn, p, 0, 0, 0, 0, 0, 0); }

TOP1B(top1_bf16_128_3, 128, 3)
TOP1B(top1_bf16_128_4, 128, 4)
TOP1B(top1_bf16_256_3, 256, 3)

// The screen reads FP16 copies (x / scale, see gather_f16s) and their aux rows.
#define SCREENK(NAME, TN, ST) \
extern "C" __global__ void __launch_bounds__(256) NAME( \
    const unsigned short* __restrict__ x, const unsigned* __restrict__ idx, \
    const unsigned short* __restrict__ c, const float* __restrict__ cn, \
    const float4* __restrict__ xa, const float4* __restrict__ ca, \
    float* __restrict__ cv, unsigned* __restrict__ cjv, float* __restrict__ ch, \
    unsigned* __restrict__ cnt, const Bf16P2 p) \
{ top1_bf16_v2_body<TN, ST, true>(x, idx, c, 0, cn, 0, 0, 0, 0, 0, p, cv, cjv, ch, cnt, xa, ca); }

SCREENK(screen_128_3, 128, 3)
SCREENK(screen_256_3, 256, 3)

// The exact FP32 answer from the screen. Per row: b = the best screen key over
// the parts, delta = twice the bound on |screen key - FP32 score|. Every
// centroid whose FP32 score can be the best has a key >= b - delta. If every
// dropped key is below that, the candidates are complete: score each with the
// FP32 kernel's arithmetic (one fmaf chain over d in order, then the BK = 16
// padding step; the L2 norm in its 16-lane order) and keep the best, lower id
// on ties. Otherwise the row goes on the list for the FP32 kernel.
//
// A warp owns 32 rows, one per lane. Columns are staged 32 at a time through
// shared memory (x rows and the lanes' candidate centroids, coalesced), and
// each lane runs its own chain over the staged tile in column order.
struct ScreenP { unsigned rows, k, d, ip, r0, xr0, indexed, parts, dense; float eps; };

extern "C" __global__ void __launch_bounds__(128) screen_verify(const float* __restrict__ x,
    const unsigned* __restrict__ idx, const float* __restrict__ c, const float* __restrict__ cn,
    const float4* __restrict__ xa, const float* __restrict__ cmax, const float* __restrict__ cv,
    const unsigned* __restrict__ cjv, const float* __restrict__ ch, unsigned* __restrict__ labels,
    float* __restrict__ scores, unsigned* __restrict__ lst, unsigned* __restrict__ cnt, const ScreenP p)
{
    constexpr int MAXC = 16;   // parts <= 4, four entries each
    __shared__ float xs[4][32][33];
    __shared__ float cs[4][32][33];
    __shared__ unsigned srow[4][32];
    __shared__ unsigned scand[4][MAXC][32];
    __shared__ unsigned sn[4][32];
    const unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned base = (blockIdx.x * 4 + w) * 32;
    if (base >= p.rows) return;
    const unsigned r = base + lane;
    const bool ip = p.ip != 0;
    const unsigned d = p.d;
    unsigned src = 0, nc = 0;
    bool go = false;   // this lane checks candidates (else: no row, or the fallback list)
    if (r < p.rows) {
        src = p.indexed ? idx[p.r0 + r] : p.xr0 + r;
        float b = -EM_INF, h = -EM_INF;
        for (unsigned q = 0; q < p.parts; ++q) {
            const unsigned long long o = (unsigned long long)q * p.rows + r;
            b = fmaxf(b, cv[o * 4]);
            h = fmaxf(h, ch[o]);
        }
        // aux = (squared norm, scale, flushed mass, 0) of this row; cmax = the
        // largest of each over the centroids.
        const float4 a = xa[p.dense ? r : src];
        const float s = sqrtf(a.x * cmax[0]) * 1.001f + 1e-30f;
        const float e = p.eps * s + (a.z * cmax[1] + cmax[2] * a.y) * 1.001f;
        const float delta = ip ? 2.f * e : 4.f * e + 4.8e-7f * (cmax[0] + 2.f * s);
        const float lim = b - delta;
        go = isfinite(b) && isfinite(lim) && h < lim;
        if (go) {
            for (unsigned q = 0; q < p.parts; ++q)
                for (int e4 = 0; e4 < 4; ++e4) {
                    const unsigned long long o = ((unsigned long long)q * p.rows + r) * 4 + e4;
                    const unsigned j = cjv[o];
                    if (j != NOROW && cv[o] >= lim) scand[w][nc++][lane] = j;
                }
        } else {
            lst[atomicAdd(cnt, 1u)] = r;
        }
    }
    srow[w][lane] = go ? src : NOROW;
    sn[w][lane] = go ? nc : 0u;
    const unsigned np = __reduce_max_sync(FULLMASK, go ? nc : 0u);
    if (np == 0) return;
    __syncwarp();
    float bv = ip ? -EM_INF : EM_INF;
    unsigned bj = 0;
    float q16[16];
    #pragma unroll
    for (int l = 0; l < 16; ++l) q16[l] = 0.f;
    // Candidates in groups of four: one sweep over the columns stages each x
    // tile once and runs up to four chains from it.
    for (unsigned g0 = 0; g0 < np; g0 += 4) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const bool norm = !ip && g0 == 0;
        for (unsigned t0 = 0; t0 < d; t0 += 32) {
            const unsigned t = t0 + lane;
            const unsigned n = d - t0 < 32 ? d - t0 : 32;
            // All 32 loads in flight before the first store (one latency per tile).
            {
                float v[32];
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const unsigned rs = srow[w][i];
                    v[i] = (rs != NOROW && t < d) ? __ldg(x + (unsigned long long)rs * d + t) : 0.f;
                }
                #pragma unroll
                for (int i = 0; i < 32; ++i) xs[w][i][lane] = v[i];
            }
            __syncwarp();
            if (norm && go) {
                if (n == 32) {
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) q16[i & 15] = fmaf(xs[w][lane][i], xs[w][lane][i], q16[i & 15]);
                } else {
                    for (unsigned i = 0; i < n; ++i) {
                        #pragma unroll
                        for (int l = 0; l < 16; ++l)
                            if ((i & 15) == (unsigned)l) q16[l] = fmaf(xs[w][lane][i], xs[w][lane][i], q16[l]);
                    }
                }
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned pc = g0 + u;
                if (pc >= np) break;
                {
                    float v[32];
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        const unsigned ji = pc < sn[w][i] ? scand[w][pc][i] : NOROW;
                        v[i] = (ji != NOROW && t < d) ? __ldg(c + (unsigned long long)ji * d + t) : 0.f;
                    }
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) cs[w][i][lane] = v[i];
                }
                __syncwarp();
                if (go && pc < nc) {
                    if (n == 32) {
                        #pragma unroll
                        for (int i = 0; i < 32; ++i) acc[u] = fmaf(xs[w][lane][i], cs[w][lane][i], acc[u]);
                    } else {
                        for (unsigned i = 0; i < n; ++i) acc[u] = fmaf(xs[w][lane][i], cs[w][lane][i], acc[u]);
                    }
                }
                __syncwarp();
            }
        }
        if (go) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned pc = g0 + u;
                if (pc >= nc) break;
                const unsigned j = scand[w][pc][lane];
                float f = acc[u];
                if (d % 16) f = fmaf(0.f, 0.f, f);
                const float v = ip ? f : fmaf(-2.0f, f, cn[j]);
                if (better(ip, v, j, bv, bj)) { bv = v; bj = j; }
            }
        }
    }
    if (!go) return;
    // The FP32 kernel's xor-shuffle order of the 16 partial norms (lane 0's value).
    #pragma unroll
    for (int off = 8; off >= 1; off >>= 1)
        #pragma unroll
        for (int l = 0; l < 16; ++l) if (l < off) q16[l] = q16[l] + q16[l + off];
    labels[p.r0 + r] = bj;
    scores[p.r0 + r] = ip ? bv : fmaxf(q16[0] + bv, 0.f);
}

// cmax[0..3] = the largest squared norm, scale and flushed mass over k aux rows
// (one block of 256 threads).
extern "C" __global__ void max_aux(const float4* __restrict__ a, float* __restrict__ out, unsigned k)
{
    __shared__ float m[3][256];
    float v0 = 0.f, v1 = 0.f, v2 = 0.f;
    for (unsigned j = threadIdx.x; j < k; j += 256) {
        const float4 t = a[j];
        v0 = fmaxf(v0, t.x); v1 = fmaxf(v1, t.y); v2 = fmaxf(v2, t.z);
    }
    m[0][threadIdx.x] = v0; m[1][threadIdx.x] = v1; m[2][threadIdx.x] = v2;
    __syncthreads();
    for (unsigned o = 128; o >= 1; o >>= 1) {
        if (threadIdx.x < o)
            for (int q = 0; q < 3; ++q) m[q][threadIdx.x] = fmaxf(m[q][threadIdx.x], m[q][threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x < 3) out[threadIdx.x] = m[threadIdx.x][0];
}

// FP16 copy for the screen: out[r] = fp16(x_row / scale), scale = the power of
// two just above max |x_row| (exact), elements below FP16's normal range (2^-14
// after scaling) set to zero. aux[r] = (sum of squares, scale, sum of |x| of
// the zeroed elements, 0) in FP32: the error bound's inputs.
__device__ __forceinline__ unsigned short to_h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
extern "C" __global__ void gather_f16s(const float* __restrict__ x, const unsigned* __restrict__ idx,
    unsigned short* __restrict__ out, float4* __restrict__ aux, unsigned r0, unsigned rows,
    unsigned d, unsigned indexed)
{
    const unsigned r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const unsigned lane = threadIdx.x & 31;
    if (r >= rows) return;
    const unsigned long long src = indexed ? idx[r0 + r] : (unsigned long long)(r0 + r);
    const float* px = x + src * d;
    unsigned short* po = out + (unsigned long long)r * d;
    float m = 0.f, sq = 0.f;
    for (unsigned t = lane; t < d; t += 32) { const float v = px[t]; m = fmaxf(m, fabsf(v)); sq = fmaf(v, v, sq); }
    for (int off = 16; off >= 1; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(FULLMASK, m, off));
        sq += __shfl_xor_sync(FULLMASK, sq, off);
    }
    int e = 0;
    if (m > 0.f && isfinite(m)) frexpf(m, &e);
    const float scale = ldexpf(1.f, e), inv = ldexpf(1.f, -e);
    float fl = 0.f;
    for (unsigned t = lane; t < d; t += 32) {
        const float v = px[t] * inv;
        const bool tiny = fabsf(v) < 6.103515625e-05f;   // 2^-14
        if (tiny) fl += fabsf(px[t]);
        po[t] = tiny ? (unsigned short)0 : to_h(v);
    }
    for (int off = 16; off >= 1; off >>= 1) fl += __shfl_xor_sync(FULLMASK, fl, off);
    if (lane == 0) aux[r] = make_float4(sq * 1.0001f, scale, fl * 1.0001f, 0.f);
}

// Combine the parts' bests per row (as top1_combine in the FP32 module).
extern "C" __global__ void top1_bf16_combine(const float* __restrict__ pv, const unsigned* __restrict__ pj,
    const float* __restrict__ pxn, unsigned* __restrict__ labels, float* __restrict__ scores,
    unsigned rows, unsigned parts, unsigned ip, unsigned r0)
{
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float v = pv[r];
    unsigned j = pj[r];
    for (unsigned q = 1; q < parts; ++q) {
        const float ov = pv[(unsigned long long)q * rows + r];
        const unsigned oj = pj[(unsigned long long)q * rows + r];
        if (better(ip != 0, ov, oj, v, j)) { v = ov; j = oj; }
    }
    labels[r0 + r] = j;
    scores[r0 + r] = ip ? v : fmaxf(pxn[r] + v, 0.f);
}

// fp32 -> bf16, round to nearest even (finite inputs).
__device__ __forceinline__ unsigned short to_bf(float f) {
    const unsigned u = __float_as_uint(f);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// out[r] = bf16(x[idx ? idx[r0 + r] : r0 + r]); xn[r] = fp32 squared norm (L2 only).
extern "C" __global__ void gather_bf16(const float* __restrict__ x, const unsigned* __restrict__ idx,
    unsigned short* __restrict__ out, float* __restrict__ xn, unsigned r0, unsigned rows,
    unsigned d, unsigned indexed, unsigned want_norm)
{
    const unsigned r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const unsigned lane = threadIdx.x & 31;
    if (r >= rows) return;
    const unsigned long long src = indexed ? idx[r0 + r] : (unsigned long long)(r0 + r);
    const float* px = x + src * d;
    unsigned short* po = out + (unsigned long long)r * d;
    float s = 0.f;
    for (unsigned t = lane; t < d; t += 32) { const float v = px[t]; po[t] = to_bf(v); s += v * v; }
    if (want_norm) {
        for (int off = 16; off >= 1; off >>= 1) s += __shfl_xor_sync(FULLMASK, s, off);
        if (lane == 0) xn[r] = s;
    }
}
"#;
