#include "matmul.h"

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "thread_pool.h"

namespace {

// Blocking parameters.
//   MR x NR  register tile: 4 x 16 floats = 8 AVX2 accumulators, leaving
//            registers free for the broadcast A value and the B row.
//   KC       depth of one pass; a KC x NR panel of B (16 KB) stays in L1.
//   MC       rows of A per block; an MC x KC block (64 KB) stays in L2.
//   NC       columns of B per block; a KC x NC block (1 MB) targets L3.
constexpr std::size_t MR = 4;
constexpr std::size_t NR = 16;
constexpr std::size_t KC = 256;
constexpr std::size_t MC = 64;
constexpr std::size_t NC = 1024;

static_assert(MC % MR == 0 && NC % NR == 0, "blocks must be whole tiles");

constexpr std::size_t ceil_div(std::size_t x, std::size_t d) { return (x + d - 1) / d; }
constexpr std::size_t round_up(std::size_t x, std::size_t to) { return ceil_div(x, to) * to; }

// Grow-only, cache-line-aligned scratch buffer. Packing writes every element
// the micro-kernel later reads, so the memory is never zeroed, and once it has
// grown to the working size a call does no allocation at all.
class Scratch {
public:
    float* get(std::size_t n) {
        if (n > cap_) {
            buf_.reset(static_cast<float*>(
                ::operator new(n * sizeof(float), std::align_val_t{kAlign})));
            cap_ = n;
        }
        return buf_.get();
    }

private:
    static constexpr std::size_t kAlign = 64;
    struct Free {
        void operator()(float* p) const { ::operator delete(p, std::align_val_t{kAlign}); }
    };
    std::unique_ptr<float, Free> buf_;
    std::size_t cap_ = 0;
};

template <class T>
std::string shape_str(const T& t) {
    return std::to_string(t.rows) + "x" + std::to_string(t.cols);
}

template <class BT>
void check_shapes(const Tensor& A, const BT& B) {
    if (A.cols == B.rows) return;
    throw std::invalid_argument("matmul: inner dimensions differ (A is " + shape_str(A) +
                                ", B is " + shape_str(B) + ")");
}

// B is fp32, or int8 with one scale per column (null for fp32); the
// kernels below are templates over the element type.

// Pack a kc x nc block of B into NR-wide column panels. Panel j holds
// B[p][j*NR .. j*NR + NR) for p = 0..kc-1 back to back, so the micro-kernel
// reads it with unit stride. Columns past nc are zero-filled. int8 B is
// dequantized here, so the micro-kernel only ever sees fp32.
template <class BT>
void pack_b(const BT* B, std::size_t ldb, std::size_t kc, std::size_t nc,
            const float* scale, float* out) {
    for (std::size_t j0 = 0; j0 < nc; j0 += NR) {
        const std::size_t nr = std::min(NR, nc - j0);
        for (std::size_t p = 0; p < kc; ++p) {
            const BT* src = B + p * ldb + j0;
            std::size_t c = 0;
            if constexpr (std::is_same_v<BT, float>) {
                for (; c < nr; ++c) out[c] = src[c];
            } else {
                for (; c < nr; ++c) out[c] = static_cast<float>(src[c]) * scale[j0 + c];
            }
            for (; c < NR; ++c) out[c] = 0.0f;
            out += NR;
        }
    }
}

// Pack an mc x kc block of A into MR-tall row panels, transposed so the MR
// values used at step p sit next to each other. Rows past mc are zero-filled.
void pack_a(const float* A, std::size_t lda, std::size_t mc, std::size_t kc,
            float* out) {
    for (std::size_t i0 = 0; i0 < mc; i0 += MR) {
        const std::size_t mr = std::min(MR, mc - i0);
        for (std::size_t p = 0; p < kc; ++p) {
            std::size_t r = 0;
            for (; r < mr; ++r) out[r] = A[(i0 + r) * lda + p];
            for (; r < MR; ++r) out[r] = 0.0f;
            out += MR;
        }
    }
}

// C[0:mr, 0:nr] += Ap * Bp over kc steps. Because both panels are padded to
// full MR x NR, the hot loop has fixed trip counts and no edge branches; the
// compiler unrolls it and keeps acc in vector registers. Edges are handled
// only on the store.
void micro_kernel(std::size_t kc, const float* __restrict Ap,
                  const float* __restrict Bp, float* __restrict C,
                  std::size_t ldc, std::size_t mr, std::size_t nr) {
    float acc[MR][NR] = {};
    for (std::size_t p = 0; p < kc; ++p) {
        for (std::size_t r = 0; r < MR; ++r) {
            const float a = Ap[r];
            for (std::size_t c = 0; c < NR; ++c) acc[r][c] += a * Bp[c];
        }
        Ap += MR;
        Bp += NR;
    }
    for (std::size_t r = 0; r < mr; ++r)
        for (std::size_t c = 0; c < nr; ++c) C[r * ldc + c] += acc[r][c];
}

// C += A * B on raw row-major buffers with leading dimensions. C must be
// initialized by the caller. Packing buffers are per thread, so disjoint row
// ranges of C can be computed concurrently. For int8 B, scale points at the
// scale of B's first column.
template <class BT>
void gemm(std::size_t m, std::size_t n, std::size_t k,
          const float* A, std::size_t lda,
          const BT* B, std::size_t ldb, const float* scale,
          float* C, std::size_t ldc) {
    // Thread-local so pool workers keep their buffers warm across calls.
    thread_local Scratch a_buf, b_buf;
    const std::size_t kc_max = std::min(KC, k);
    float* Bp = b_buf.get(kc_max * round_up(std::min(NC, n), NR));
    float* Ap = a_buf.get(round_up(std::min(MC, m), MR) * kc_max);

    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nc = std::min(NC, n - jc);
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kc = std::min(KC, k - pc);
            pack_b(B + pc * ldb + jc, ldb, kc, nc, scale ? scale + jc : nullptr, Bp);

            for (std::size_t ic = 0; ic < m; ic += MC) {
                const std::size_t mc = std::min(MC, m - ic);
                pack_a(A + ic * lda + pc, lda, mc, kc, Ap);

                // Panel offsets: panel jr/NR starts at (jr/NR) * kc * NR = jr * kc.
                for (std::size_t jr = 0; jr < nc; jr += NR) {
                    for (std::size_t ir = 0; ir < mc; ir += MR) {
                        micro_kernel(kc, Ap + ir * kc, Bp + jr * kc,
                                     C + (ic + ir) * ldc + jc + jr, ldc,
                                     std::min(MR, mc - ir), std::min(NR, nc - jr));
                    }
                }
            }
        }
    }
}

// Small-m path. With a handful of rows each element of B is used only m times,
// so the product is bound by reading B from memory, and packing B -- reading
// it, writing a copy, reading the copy -- only adds traffic.
//
// B is row-major, so the only access pattern that reads it in long sequential
// runs is row by row. Each task takes a contiguous range of k (rows of B),
// streams those rows once, and accumulates C[0:m, 0:n] into a private partial
// sum that stays in L1/L2; the partials are added up afterwards. Splitting by
// columns instead reads each row of B in short pieces spread across threads,
// which the hardware prefetchers handle poorly.
//
// Up to 4 rows this beats the packed path when B comes from DRAM (2.3x at
// m = 1, 1.6x at m = 4 for 4096 x 4096) and is even at m = 8. When B fits in
// L3 the packed path is ~20% faster at m = 4; weights of a real model don't.
constexpr std::size_t kStreamMaxRows = 4;
// Rows of B per task below which the partial-sum overhead isn't worth it.
constexpr std::size_t kStreamMinK = 64;

// acc[0:m, 0:n] = A[0:m, 0:k] * B[0:k, 0:n]; acc is dense with stride n.
// int8 B is summed unscaled; the caller applies the column scales once, to
// the reduced result.
template <class BT>
void gemm_stream(std::size_t m, std::size_t n, std::size_t k,
                 const float* __restrict A, std::size_t lda,
                 const BT* __restrict B, std::size_t ldb,
                 float* __restrict acc) {
    std::fill(acc, acc + m * n, 0.0f);
    for (std::size_t p = 0; p < k; ++p) {
        const BT* b = B + p * ldb;
        for (std::size_t r = 0; r < m; ++r) {
            const float a = A[r * lda + p];
            float* out = acc + r * n;
            for (std::size_t c = 0; c < n; ++c) out[c] += a * static_cast<float>(b[c]);
        }
    }
}

} // namespace

// Loop order i-k-j: the inner loop walks memory in a straight line (cache-friendly).
Tensor matmul_naive(const Tensor& A, const Tensor& B) {
    check_shapes(A, B);
    Tensor C(A.rows, B.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t k = 0; k < A.cols; ++k) {
            float a = A.at(i, k);
            for (std::size_t j = 0; j < B.cols; ++j) {
                C.at(i, j) += a * B.at(k, j);
            }
        }
    }
    return C;
}

Tensor matmul_blocked(const Tensor& A, const Tensor& B) {
    check_shapes(A, B);
    Tensor C(A.rows, B.cols);
    gemm(A.rows, B.cols, A.cols, A.data.data(), A.cols,
         B.data.data(), B.cols, nullptr, C.data.data(), C.cols);
    return C;
}

namespace {

// C is cut into a grid of tiles: MR-aligned slabs of rows by NR-aligned slabs
// of columns. Each task owns one tile and its own packing buffers, so there is
// no shared mutable state and no synchronization beyond the pool's completion
// wait.
//
// Rows are split first, since each row slab reuses its packed B the most.
// When m is too small to occupy every thread -- m = 1 for each token during
// generation -- the leftover threads go to column slabs. A column slab packs
// only its own slice of B, so splitting by columns adds no packing work, while
// splitting by rows makes every slab pack all of B again.
// B is k x n with leading dimension n; scale is null for fp32 B.
template <class BT>
Tensor parallel(const Tensor& A, const BT* B, const float* scale, std::size_t n,
                unsigned threads) {
    const std::size_t m = A.rows, k = A.cols;
    Tensor C(m, n);
    if (m == 0 || n == 0) return C;

    ThreadPool& pool = ThreadPool::global();
    if (threads == 0) threads = pool.size();
    // More tasks than pool threads would only run back to back.
    const std::size_t workers = std::min<std::size_t>(threads, pool.size());
    // Below one MC block of rows, or kMinCols columns, per task, splitting
    // costs more than it saves.
    constexpr std::size_t kMinCols = 4 * NR;

    if (m <= kStreamMaxRows) {
        // At least one task, so k = 0 still writes (zero) partials.
        const std::size_t tasks =
            std::max<std::size_t>(1, std::min(workers, ceil_div(k, kStreamMinK)));
        const std::size_t span = ceil_div(k, tasks);
        // One partial C per task, owned by this call's thread and reused.
        thread_local Scratch partial_buf;
        float* partials = partial_buf.get(tasks * m * n);
        auto run_span = [&](std::size_t t) {
            const std::size_t p0 = t * span;
            gemm_stream(m, n, std::min(span, k - p0), A.data.data() + p0, k,
                        B + p0 * n, n, partials + t * m * n);
        };
        if (tasks == 1) {
            run_span(0);
        } else {
            pool.run(tasks, run_span);
        }
        float* out = C.data.data();
        for (std::size_t t = 0; t < tasks; ++t) {
            const float* part = partials + t * m * n;
            for (std::size_t i = 0; i < m * n; ++i) out[i] += part[i];
        }
        if (scale) {
            for (std::size_t r = 0; r < m; ++r)
                for (std::size_t c = 0; c < n; ++c) out[r * n + c] *= scale[c];
        }
        return C;
    }
    const std::size_t row_tasks = std::min(workers, ceil_div(m, MC));
    const std::size_t col_tasks =
        std::min(std::max<std::size_t>(1, workers / row_tasks), ceil_div(n, kMinCols));
    const std::size_t row_slab = round_up(ceil_div(m, row_tasks), MR);
    const std::size_t col_slab = round_up(ceil_div(n, col_tasks), NR);
    const std::size_t row_tiles = ceil_div(m, row_slab);
    const std::size_t col_tiles = ceil_div(n, col_slab);

    auto run_tile = [&](std::size_t t) {
        const std::size_t row0 = (t / col_tiles) * row_slab;
        const std::size_t col0 = (t % col_tiles) * col_slab;
        gemm(std::min(row_slab, m - row0), std::min(col_slab, n - col0), k,
             A.data.data() + row0 * k, k,
             B + col0, n, scale ? scale + col0 : nullptr,
             C.data.data() + row0 * n + col0, n);
    };
    if (row_tiles * col_tiles == 1) {
        run_tile(0);
    } else {
        pool.run(row_tiles * col_tiles, run_tile);
    }
    return C;
}

} // namespace

Tensor matmul_parallel(const Tensor& A, const Tensor& B, unsigned threads) {
    check_shapes(A, B);
    return parallel(A, B.data.data(), nullptr, B.cols, threads);
}

Tensor matmul_parallel(const Tensor& A, const QTensor& B, unsigned threads) {
    check_shapes(A, B);
    return parallel(A, B.data.data(), B.scale.data(), B.cols, threads);
}
