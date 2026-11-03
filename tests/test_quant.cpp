#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#include "matmul.h"
#include "quant.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                         #cond);                                                 \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

Tensor random_tensor(std::size_t rows, std::size_t cols, float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    Tensor t(rows, cols);
    for (auto& x : t.data) x = dist(rng);
    return t;
}

// Every element lands within half a step of its column's scale, the largest
// magnitude in each column maps to +-127, and a zero column stays zero.
void test_round_trip() {
    std::mt19937 rng(3);
    Tensor B = random_tensor(37, 21, 2.0f, rng);
    for (std::size_t c = 0; c < B.cols; ++c) B.at(5, c) *= float(c + 1);  // spread the scales
    for (std::size_t r = 0; r < B.rows; ++r) B.at(r, 4) = 0.0f;
    const QTensor q = quantize_q8(B);
    const Tensor d = dequantize(q);
    CHECK(q.rows == B.rows && q.cols == B.cols && q.scale.size() == B.cols);
    CHECK(q.scale[4] == 0.0f);
    for (std::size_t c = 0; c < B.cols; ++c) {
        int max_q = 0;
        for (std::size_t r = 0; r < B.rows; ++r) {
            const int v = q.data[r * B.cols + c];
            max_q = std::max(max_q, std::abs(v));
            CHECK(std::fabs(d.at(r, c) - B.at(r, c)) <= q.scale[c] * 0.5f * (1 + 1e-5f));
        }
        CHECK(max_q == (c == 4 ? 0 : 127));
    }
}

// The int8 kernels must compute exactly the product with the dequantized
// matrix, up to float rounding: both the streaming path (m <= 4, scales
// applied after the split-K sum) and the packed path (dequantized while
// packing), across block edges and thread counts.
void test_matmul_matches_dequantized() {
    std::mt19937 rng(11);
    struct Shape { std::size_t m, k, n; };
    for (const Shape s : {Shape{1, 1, 1}, Shape{1, 300, 1030}, Shape{3, 1000, 70},
                          Shape{4, 4097, 33}, Shape{5, 257, 17}, Shape{64, 255, 128},
                          Shape{130, 70, 1030}, Shape{2, 0, 5}}) {
        const Tensor A = random_tensor(s.m, s.k, 1.0f, rng);
        const QTensor q = quantize_q8(random_tensor(s.k, s.n, 0.5f, rng));
        const Tensor want = matmul_naive(A, dequantize(q));
        for (unsigned t : {1u, 3u, 0u}) {
            const Tensor got = matmul_parallel(A, q, t);
            const float tol = 1e-5f * float(std::max<std::size_t>(s.k, 1));
            bool ok = got.rows == want.rows && got.cols == want.cols;
            for (std::size_t i = 0; ok && i < got.data.size(); ++i)
                ok = std::fabs(got.data[i] - want.data[i]) <= tol;
            if (!ok) {
                std::fprintf(stderr, "  mismatch at m=%zu k=%zu n=%zu threads=%u\n", s.m, s.k,
                             s.n, t);
                CHECK(false);
            }
        }
    }
}

void test_shape_mismatch() {
    std::string message;
    try {
        matmul_parallel(Tensor(2, 3), quantize_q8(Tensor(4, 2)));
    } catch (const std::invalid_argument& e) {
        message = e.what();
    }
    CHECK(message.find("2x3") != std::string::npos);
    CHECK(message.find("4x2") != std::string::npos);
}

} // namespace

int main() {
    test_round_trip();
    test_matmul_matches_dequantized();
    test_shape_mismatch();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
}
