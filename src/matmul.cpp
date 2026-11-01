#include "matmul.h"

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

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

std::string shape_str(const Tensor& t) {
    return std::to_string(t.rows) + "x" + std::to_string(t.cols);
}

void check_shapes(const Tensor& A, const Tensor& B) {
    if (A.cols == B.rows) return;
    throw std::invalid_argument("matmul: inner dimensions differ (A is " + shape_str(A) +
                                ", B is " + shape_str(B) + ")");
}

// Pack a kc x nc block of B into NR-wide column panels. Panel j holds
// B[p][j*NR .. j*NR + NR) for p = 0..kc-1 back to back, so the micro-kernel
// reads it with unit stride. Columns past nc are zero-filled.
void pack_b(const float* B, std::size_t ldb, std::size_t kc, std::size_t nc,
            float* out) {
    for (std::size_t j0 = 0; j0 < nc; j0 += NR) {
        const std::size_t nr = std::min(NR, nc - j0);
        for (std::size_t p = 0; p < kc; ++p) {
            const float* src = B + p * ldb + j0;
            std::size_t c = 0;
            for (; c < nr; ++c) out[c] = src[c];
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
// ranges of C can be computed concurrently.
void gemm(std::size_t m, std::size_t n, std::size_t k,
          const float* A, std::size_t lda,
          const float* B, std::size_t ldb,
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
            pack_b(B + pc * ldb + jc, ldb, kc, nc, Bp);

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
         B.data.data(), B.cols, C.data.data(), C.cols);
    return C;
}

// Each task computes a disjoint, MR-aligned slab of rows of C with its own
// packing buffers, so there is no shared mutable state and no synchronization
// beyond the pool's completion wait. The cost is that every task packs all of
// B, which is O(k*n) next to O(slab*k*n) compute; a shared packed B is a later
// step.
Tensor matmul_parallel(const Tensor& A, const Tensor& B, unsigned threads) {
    check_shapes(A, B);
    const std::size_t m = A.rows, n = B.cols, k = A.cols;
    Tensor C(m, n);
    if (m == 0 || n == 0) return C;

    ThreadPool& pool = ThreadPool::global();
    if (threads == 0) threads = pool.size();
    // Below one MC block per task, splitting costs more than it saves, and
    // more tasks than pool threads would only run back to back.
    const std::size_t tasks =
        std::min<std::size_t>({threads, pool.size(), ceil_div(m, MC)});
    const std::size_t slab = round_up(ceil_div(m, tasks), MR);

    if (tasks == 1) {
        gemm(m, n, k, A.data.data(), k, B.data.data(), n, C.data.data(), n);
        return C;
    }
    pool.run(ceil_div(m, slab), [&](std::size_t t) {
        const std::size_t row0 = t * slab;
        const std::size_t rows = std::min(slab, m - row0);
        gemm(rows, n, k, A.data.data() + row0 * k, k,
             B.data.data(), n, C.data.data() + row0 * n, n);
    });
    return C;
}
