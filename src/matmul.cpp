#include "matmul.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

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

void check_shapes(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) throw std::invalid_argument("matmul: shape mismatch");
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
// initialized by the caller. Each call owns its packing buffers, so disjoint
// row ranges of C can be computed concurrently.
void gemm(std::size_t m, std::size_t n, std::size_t k,
          const float* A, std::size_t lda,
          const float* B, std::size_t ldb,
          float* C, std::size_t ldc) {
    std::vector<float> Bp(KC * NC);
    std::vector<float> Ap(MC * KC);

    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nc = std::min(NC, n - jc);
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kc = std::min(KC, k - pc);
            pack_b(B + pc * ldb + jc, ldb, kc, nc, Bp.data());

            for (std::size_t ic = 0; ic < m; ic += MC) {
                const std::size_t mc = std::min(MC, m - ic);
                pack_a(A + ic * lda + pc, lda, mc, kc, Ap.data());

                // Panel offsets: panel jr/NR starts at (jr/NR) * kc * NR = jr * kc.
                for (std::size_t jr = 0; jr < nc; jr += NR) {
                    for (std::size_t ir = 0; ir < mc; ir += MR) {
                        micro_kernel(kc, Ap.data() + ir * kc, Bp.data() + jr * kc,
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
