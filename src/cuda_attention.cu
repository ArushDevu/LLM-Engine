#include "cuda_attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "cuda_util.cuh"

namespace {

constexpr int kSingleThreads = 256;
constexpr int kSplitThreads = 128;
// split: a block takes at least kMinChunk positions, and a (head group, row)
// is cut into at most kMaxSplits blocks.
constexpr int kMinChunk = 32;
constexpr int kMaxSplits = 128;
// automatic: split up to this many rows; beyond, heads x rows blocks already
// fill the GPU.
constexpr std::size_t kSplitMaxRows = 8;
constexpr std::size_t kMaxSmem = 48 * 1024;

void check_launch() { cuda_check(cudaGetLastError(), "launch"); }

int split_count(int seq_len) { return std::min(kMaxSplits, (seq_len + kMinChunk - 1) / kMinChunk); }

// Positions per block for a context of n_ctx cut into at most splits blocks;
// the blocks past the context's end have nothing to do.
__host__ __device__ inline int chunk_size(int n_ctx, int splits) {
    const int c = (n_ctx + splits - 1) / splits;
    return c > kMinChunk ? c : kMinChunk;
}

std::size_t single_smem(const AttentionShape& s) {
    return (std::size_t(s.head_size) + std::size_t(s.seq_len)) * sizeof(float);
}

// One block per (query head, row): row i, at position *pos + i, attends to
// cached positions [0, *pos + i]. The query and the scores live in shared
// memory (head_size + seq_len floats). Threads split the positions for the
// scores and the head's dimensions for the weighted sum of values, which
// reads each value row coalesced. (Splitting the weighted sum across all 256
// threads in head_size-wide groups, with a final reduction, was slower: 11.9
// vs 9.8 us per call.)
__global__ void attention_single(int head_size, int group, int q_stride, int kv_stride,
                                 int out_stride, const int* pos, const float* qkv,
                                 const float* key_cache, const float* value_cache, float* out) {
    extern __shared__ float smem[];
    float* q = smem;
    float* scores = smem + head_size;
    const int h = blockIdx.x, i = blockIdx.y, n_ctx = *pos + i + 1;
    const int kv_off = (h / group) * head_size;

    const float* qg = qkv + std::size_t(i) * q_stride + h * head_size;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) q[d] = qg[d];
    __syncthreads();

    const float scale = 1.0f / sqrtf(float(head_size));
    float mx = -INFINITY;
    for (int t = threadIdx.x; t < n_ctx; t += blockDim.x) {
        const float* k = key_cache + std::size_t(t) * kv_stride + kv_off;
        float dot = 0.0f;
        for (int d = 0; d < head_size; ++d) dot += q[d] * k[d];
        scores[t] = dot * scale;
        mx = fmaxf(mx, scores[t]);
    }
    mx = block_reduce(mx, MaxOp{}, -INFINITY);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_ctx; t += blockDim.x) {
        scores[t] = expf(scores[t] - mx);
        sum += scores[t];
    }
    // block_reduce's barriers also make every score visible below.
    const float inv = 1.0f / block_reduce(sum, SumOp{}, 0.0f);

    float* oh = out + std::size_t(i) * out_stride + h * head_size;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        const float* v = value_cache + kv_off + d;
        float acc = 0.0f;
        for (int t = 0; t < n_ctx; ++t) acc += scores[t] * v[std::size_t(t) * kv_stride];
        oh[d] = acc * inv;
    }
}

// Split-KV attention (flash-decoding). Block (hg, i, s) takes the s-th chunk
// of row i's context for G query heads hg * G .. hg * G + G - 1, which share
// a kv head, so each key and value is loaded once for all G. For each head it
// writes the chunk's softmax unnormalized: the max score m, the sum l of
// exp(score - m), and the exp-weighted sum o of the values. The last block of
// a (head group, row) to finish (an atomic counter, reset by that block)
// merges the chunks in chunk order, so the result is deterministic and no
// second kernel is launched; a context that fits one chunk is normalized and
// written directly. Chunks past the context's end exit at once: the grid is
// sized for seq_len, so a captured graph serves every position.
//
// Keys: a group of lpp lanes (head_size / 4 rounded up to a power of two, at
// most 32) shares a position, each lane loading float4s of the key, and the
// dot product is summed with shuffles. Values: consecutive threads load
// consecutive float4s of a value row; the head_size / 4 threads of a row form
// a slot, slots stride over positions, and their sums are added in shared
// memory.
template <int G>
__global__ void __launch_bounds__(kSplitThreads)
    attention_split(int n_heads, int head_size, int group, int q_stride, int kv_stride,
                    int out_stride, int chunk_cap, const int* pos, const float* qkv,
                    const float* key_cache, const float* value_cache, float* part_o,
                    float2* part_ml, unsigned* counters, float* out) {
    extern __shared__ float4 smem4[];
    const int hs4 = head_size / 4;
    const int cap4 = (chunk_cap + 3) / 4 * 4;
    float4* q = smem4;                                           // G x hs4
    float* scores = reinterpret_cast<float*>(q + G * hs4);       // G x cap4
    float4* red = reinterpret_cast<float4*>(scores + G * cap4);  // slots x G x hs4
    __shared__ float2 ml[G];
    __shared__ bool last;

    const int hg = blockIdx.x, i = blockIdx.y, s = blockIdx.z, splits = gridDim.z;
    const int n_ctx = *pos + i + 1, chunk = chunk_size(n_ctx, splits);
    const int n = (n_ctx + chunk - 1) / chunk;  // chunks with work
    const int t0 = s * chunk;
    if (t0 >= n_ctx) return;  // uniform across the block
    const int len = min(chunk, n_ctx - t0);
    const int h0 = hg * G, kv_off = (h0 / group) * head_size;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, nwarps = blockDim.x / 32;

    // The G heads are contiguous in q.
    const float4* qg = reinterpret_cast<const float4*>(qkv + std::size_t(i) * q_stride +
                                                       std::size_t(h0) * head_size);
    for (int j = tid; j < G * hs4; j += blockDim.x) q[j] = qg[j];
    __syncthreads();

    // Scores, lpp lanes per position. Every lane runs the same trip count so
    // the shuffles see full warps.
    int lpp = 1;
    while (lpp < hs4 && lpp < 32) lpp *= 2;
    const int sub = tid % lpp, slot = tid / lpp, nslots = blockDim.x / lpp;
    const float scale = 1.0f / sqrtf(float(head_size));
    for (int base = 0; base < len; base += nslots) {
        const int t = base + slot;
        float dot[G];
        for (int g = 0; g < G; ++g) dot[g] = 0.0f;
        if (t < len) {
            const float4* k = reinterpret_cast<const float4*>(
                key_cache + std::size_t(t0 + t) * kv_stride + kv_off);
            for (int d = sub; d < hs4; d += lpp) {
                const float4 kv = __ldg(k + d);
                for (int g = 0; g < G; ++g) {
                    const float4 qv = q[g * hs4 + d];
                    dot[g] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            }
        }
        for (int o = lpp / 2; o > 0; o >>= 1)
            for (int g = 0; g < G; ++g) dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], o);
        if (t < len && sub == 0)
            for (int g = 0; g < G; ++g) scores[g * cap4 + t] = dot[g] * scale;
    }
    __syncthreads();

    // Softmax of the chunk, one warp per head.
    for (int g = warp; g < G; g += nwarps) {
        float* sc = scores + g * cap4;
        float m = -INFINITY;
        for (int t = lane; t < len; t += 32) m = fmaxf(m, sc[t]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float l = 0.0f;
        for (int t = lane; t < len; t += 32) {
            sc[t] = expf(sc[t] - m);
            l += sc[t];
        }
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane == 0) ml[g] = make_float2(m, l);
    }
    __syncthreads();

    // Weighted sum of values.
    const int d4 = tid % hs4, r = tid / hs4, slots = blockDim.x / hs4;
    if (r < slots) {
        float4 acc[G];
        for (int g = 0; g < G; ++g) acc[g] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float* vbase = value_cache + std::size_t(t0) * kv_stride + kv_off;
        for (int t = r; t < len; t += slots) {
            const float4 v =
                __ldg(reinterpret_cast<const float4*>(vbase + std::size_t(t) * kv_stride) + d4);
            for (int g = 0; g < G; ++g) {
                const float p = scores[g * cap4 + t];
                acc[g].x += p * v.x;
                acc[g].y += p * v.y;
                acc[g].z += p * v.z;
                acc[g].w += p * v.w;
            }
        }
        for (int g = 0; g < G; ++g) red[(r * G + g) * hs4 + d4] = acc[g];
    }
    __syncthreads();

    const std::size_t head_row = std::size_t(i) * n_heads + h0;  // (row, first head)
    for (int j = tid; j < G * hs4; j += blockDim.x) {
        const int g = j / hs4, d = j % hs4;
        float4 o = red[g * hs4 + d];
        for (int rr = 1; rr < slots; ++rr) {
            const float4 x = red[(rr * G + g) * hs4 + d];
            o.x += x.x;
            o.y += x.y;
            o.z += x.z;
            o.w += x.w;
        }
        if (n == 1) {
            const float inv = 1.0f / ml[g].y;
            reinterpret_cast<float4*>(out + std::size_t(i) * out_stride +
                                      std::size_t(h0 + g) * head_size)[d] =
                make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv);
        } else {
            reinterpret_cast<float4*>(part_o)[((head_row + g) * splits + s) * hs4 + d] = o;
        }
    }
    if (n == 1) return;
    if (tid < G) part_ml[(head_row + tid) * splits + s] = ml[tid];

    // Publish, then let the block that finishes last merge.
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        unsigned* c = counters + std::size_t(i) * gridDim.x + hg;
        last = atomicAdd(c, 1u) == unsigned(n - 1);
        if (last) *c = 0;  // ready for the next call
    }
    __syncthreads();
    if (!last) return;
    __threadfence();

    // Partials from other blocks are read through L2 (__ldcg): this SM's L1
    // may hold stale lines.
    for (int j = tid; j < G * hs4; j += blockDim.x) {
        const int g = j / hs4, d = j % hs4;
        const std::size_t b = (head_row + g) * splits;
        float m = -INFINITY;
        for (int c = 0; c < n; ++c) m = fmaxf(m, __ldcg(&part_ml[b + c]).x);
        float l = 0.0f;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int c = 0; c < n; ++c) {
            const float2 p = __ldcg(&part_ml[b + c]);
            const float w = expf(p.x - m);
            const float4 o = __ldcg(reinterpret_cast<const float4*>(part_o) + (b + c) * hs4 + d);
            l += w * p.y;
            acc.x += w * o.x;
            acc.y += w * o.y;
            acc.z += w * o.z;
            acc.w += w * o.w;
        }
        const float inv = 1.0f / l;
        reinterpret_cast<float4*>(out + std::size_t(i) * out_stride +
                                  std::size_t(h0 + g) * head_size)[d] =
            make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
    }
}

// Query heads per split block: the most of 8, 4, 2, 1 that divides the heads
// sharing a kv head.
int heads_per_block(const AttentionShape& s) {
    const int group = s.n_heads / s.n_kv_heads;
    for (int g : {8, 4, 2})
        if (group % g == 0) return g;
    return 1;
}

int split_chunk_cap(const AttentionShape& s) { return chunk_size(s.seq_len, split_count(s.seq_len)); }

std::size_t split_smem(const AttentionShape& s) {
    const std::size_t G = std::size_t(heads_per_block(s)), hs = std::size_t(s.head_size);
    const std::size_t cap4 = (std::size_t(split_chunk_cap(s)) + 3) / 4 * 4;
    const std::size_t slots = kSplitThreads / (hs / 4);
    return (G * hs + G * cap4 + slots * G * hs) * sizeof(float);
}

// automatic: split for few rows, or whenever single_block's scores wouldn't
// fit in shared memory.
AttentionKernel resolve(const AttentionShape& s, AttentionKernel k, std::size_t rows) {
    if (k != AttentionKernel::automatic) return k;
    return rows <= kSplitMaxRows || single_smem(s) > kMaxSmem ? AttentionKernel::split
                                                              : AttentionKernel::single_block;
}

// Scratch layout for split, for rows rows: o vectors, then (m, l) pairs, then
// one counter per (row, head); every section 16-byte aligned.
struct SplitScratch {
    std::size_t o, ml, counters, total;  // offsets in floats
};

SplitScratch split_scratch(const AttentionShape& s, std::size_t rows) {
    const std::size_t heads = rows * std::size_t(s.n_heads);
    const std::size_t blocks = heads * std::size_t(split_count(s.seq_len));
    auto up4 = [](std::size_t x) { return (x + 3) / 4 * 4; };
    SplitScratch l{};
    l.ml = up4(blocks * std::size_t(s.head_size));
    l.counters = l.ml + up4(2 * blocks);
    l.total = l.counters + heads;
    return l;
}

template <int G>
void launch_split(const AttentionShape& s, std::size_t rows, const int* pos, const float* q,
                  const float* kc, const float* vc, float* out, float* scratch) {
    const SplitScratch l = split_scratch(s, rows);
    const dim3 grid(unsigned(s.n_heads / G), unsigned(rows), unsigned(split_count(s.seq_len)));
    attention_split<G><<<grid, kSplitThreads, split_smem(s)>>>(
        s.n_heads, s.head_size, s.n_heads / s.n_kv_heads, int(s.q_stride), int(s.kv_stride),
        int(s.out_stride), split_chunk_cap(s), pos, q, kc, vc, scratch + l.o,
        reinterpret_cast<float2*>(scratch + l.ml), reinterpret_cast<unsigned*>(scratch + l.counters),
        out);
    check_launch();
}

} // namespace

void attention_check(const AttentionShape& s, AttentionKernel kernel) {
    if (s.n_heads <= 0 || s.n_kv_heads <= 0 || s.head_size <= 0 || s.seq_len <= 0 ||
        s.n_heads % s.n_kv_heads != 0)
        throw std::invalid_argument("attention: bad shape");
    if (std::size_t(s.seq_len) * s.kv_stride >= (std::size_t(1) << 31))
        throw std::invalid_argument("attention: cache past 32-bit indexing");
    const bool single = kernel == AttentionKernel::single_block;
    // split loads float4s.
    if (!single && (s.head_size % 4 || s.head_size > 4 * kSplitThreads || s.q_stride % 4 ||
                    s.kv_stride % 4 || s.out_stride % 4))
        throw std::invalid_argument("attention: split needs head_size and strides divisible by 4");
    // automatic falls back to split whenever single_block doesn't fit.
    const std::size_t need = single ? single_smem(s) : split_smem(s);
    if (need > kMaxSmem)
        throw std::invalid_argument("attention: seq_len " + std::to_string(s.seq_len) + " needs " +
                                    std::to_string(need) + " bytes of shared memory, more than " +
                                    std::to_string(kMaxSmem));
}

std::size_t attention_scratch_floats(const AttentionShape& s, std::size_t rows,
                                     AttentionKernel kernel) {
    if (kernel == AttentionKernel::single_block) return 0;
    // The most rows a call of up to rows rows splits.
    if (kernel == AttentionKernel::automatic && single_smem(s) <= kMaxSmem)
        rows = std::min(rows, kSplitMaxRows);
    return split_scratch(s, rows).total;
}

void attention_cuda(const AttentionShape& s, std::size_t rows, const int* pos_dev,
                    const float* q, const float* key_cache, const float* value_cache, float* out,
                    float* scratch, AttentionKernel kernel) {
    kernel = resolve(s, kernel, rows);
    if (kernel == AttentionKernel::single_block) {
        attention_single<<<dim3(unsigned(s.n_heads), unsigned(rows)), kSingleThreads,
                           single_smem(s)>>>(s.head_size, s.n_heads / s.n_kv_heads,
                                             int(s.q_stride), int(s.kv_stride), int(s.out_stride),
                                             pos_dev, q, key_cache, value_cache, out);
        check_launch();
        return;
    }
    switch (heads_per_block(s)) {
    case 8: launch_split<8>(s, rows, pos_dev, q, key_cache, value_cache, out, scratch); break;
    case 4: launch_split<4>(s, rows, pos_dev, q, key_cache, value_cache, out, scratch); break;
    case 2: launch_split<2>(s, rows, pos_dev, q, key_cache, value_cache, out, scratch); break;
    default: launch_split<1>(s, rows, pos_dev, q, key_cache, value_cache, out, scratch); break;
    }
}
