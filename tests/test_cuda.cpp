// Checks every GPU kernel against the CPU reference. Exits with 77 (reported
// as skipped by ctest) when there is no CUDA device.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_matmul.h"
#include "matmul.h"
#include "ops.h"
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

Tensor random_tensor(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t(rows, cols);
    for (auto& x : t.data) x = dist(rng);
    return t;
}

// Same tolerance as the CPU tests: summation order differs from the
// reference, and the error grows with k.
bool close_to(const Tensor& got, const Tensor& want, std::size_t k) {
    if (got.rows != want.rows || got.cols != want.cols) return false;
    const float tol = 1e-5f * static_cast<float>(std::max<std::size_t>(k, 1));
    for (std::size_t i = 0; i < got.data.size(); ++i) {
        if (std::fabs(got.data[i] - want.data[i]) > tol) return false;
    }
    return true;
}

struct Kernel {
    const char* name;
    GpuKernel kernel;
    std::size_t max_m;
};

const Kernel kKernels[] = {
    {"naive", GpuKernel::naive, ~std::size_t{0}},
    {"tiled", GpuKernel::tiled, ~std::size_t{0}},
    {"tiled_split", GpuKernel::tiled_split, ~std::size_t{0}},
    {"skinny", GpuKernel::skinny, 8},
    {"automatic", GpuKernel::automatic, ~std::size_t{0}},
};

// Shapes on and around the tile edges (128 x 128 x 8), the skinny cutoff
// (m = 8, 9), k slices that don't divide k, and an empty inner dimension.
struct Shape { std::size_t m, k, n; };
constexpr Shape kShapes[] = {
    {1, 1, 1},     {1, 17, 1},     {3, 5, 7},      {8, 8, 128},    {9, 9, 129},
    {127, 300, 129}, {128, 256, 128}, {200, 513, 33}, {130, 70, 1030},
    {1, 4096, 288}, {1, 288, 4096}, {2, 1000, 70},  {4, 4097, 33},  {7, 64, 4111},
    {8, 513, 2049}, {16, 4096, 300}, {33, 1029, 257}, {64, 2000, 4096},
    {1, 0, 5},      {12, 0, 3},
};

void test_against_reference() {
    std::mt19937 rng(42);
    for (const Shape& s : kShapes) {
        const Tensor A = random_tensor(s.m, s.k, rng);
        const Tensor B = random_tensor(s.k, s.n, rng);
        const Tensor want = matmul_naive(A, B);
        for (const Kernel& k : kKernels) {
            if (s.m > k.max_m) continue;
            if (!close_to(matmul_cuda(A, B, k.kernel), want, s.k)) {
                std::fprintf(stderr, "  [%s] mismatch at m=%zu k=%zu n=%zu\n", k.name, s.m,
                             s.k, s.n);
                CHECK(false);
            }
        }
    }
}

// The output is reallocated to the right shape, and reusing it across calls
// of different shapes (as the skinny path's partial buffer is) is safe.
void test_reuse() {
    std::mt19937 rng(5);
    DeviceTensor C(3, 3);
    for (const Shape& s : {Shape{1, 2000, 300}, Shape{4, 100, 5000}, Shape{64, 3000, 64},
                           Shape{200, 64, 64}}) {
        const Tensor A = random_tensor(s.m, s.k, rng);
        const Tensor B = random_tensor(s.k, s.n, rng);
        const DeviceTensor dA(A), dB(B);
        matmul_cuda(dA, dB, C);
        CHECK(C.rows() == s.m && C.cols() == s.n);
        CHECK(close_to(C.download(), matmul_naive(A, B), s.k));
    }
}

template <class F>
std::string invalid_argument_of(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor c = a;
    for (std::size_t i = 0; i < c.data.size(); ++i) c.data[i] += b.data[i];
    return c;
}

// Every MatmulFusion option on every kernel: in the skinny kernel the norm
// is fused, elsewhere it runs as a separate pass first. Shapes
// cover one and several k slices, and the skinny / split / tiled cutoffs.
void test_fusion() {
    std::mt19937 rng(9);
    for (const Shape& s : {Shape{1, 288, 300}, Shape{3, 100, 70}, Shape{8, 513, 129},
                           Shape{9, 64, 33}, Shape{70, 40, 130}}) {
        const Tensor A = random_tensor(s.m, s.k, rng);
        const Tensor B = random_tensor(s.k, s.n, rng);
        const Tensor C0 = random_tensor(s.m, s.n, rng);
        Tensor w_host = random_tensor(1, s.k, rng);
        const std::vector<float> w = w_host.data;
        const DeviceTensor dA(A), dB(B), dW(w_host);

        struct Case {
            const char* name;
            MatmulFusion fusion;
            Tensor want;
        };
        MatmulFusion acc, norm, norm_acc;
        acc.accumulate = norm_acc.accumulate = true;
        norm.rms_weight = norm_acc.rms_weight = dW.data();
        const Tensor plain = matmul_naive(A, B);
        const Tensor normed = matmul_naive(rmsnorm(A, w), B);
        const Case cases[] = {
            {"accumulate", acc, add(C0, plain)},
            {"rms", norm, normed},
            {"rms + accumulate", norm_acc, add(C0, normed)},
        };
        for (const Kernel& k : kKernels) {
            if (s.m > k.max_m) continue;
            for (const Case& c : cases) {
                DeviceTensor dC(C0);
                matmul_cuda(s.m, s.n, s.k, dA.data(), dB.data(), dC.data(), k.kernel, c.fusion);
                if (!close_to(dC.download(), c.want, 2 * s.k)) {
                    std::fprintf(stderr, "  [%s] %s mismatch at m=%zu k=%zu n=%zu\n", k.name,
                                 c.name, s.m, s.k, s.n);
                    CHECK(false);
                }
            }
        }
    }
}

// int8 B on every kernel, plain and with each fusion, against the fp32
// reference on the dequantized weights: the kernels dequantize as they load,
// so the only difference left is summation order.
void test_q8() {
    std::mt19937 rng(11);
    // Plus shapes for the skinny kernel's 4-column path (4 | n): a partial
    // last block, and one k slice (C written directly).
    std::vector<Shape> shapes(std::begin(kShapes), std::end(kShapes));
    shapes.insert(shapes.end(), {{3, 100, 1028}, {8, 1000, 4100}, {6, 20, 36}, {1, 4096, 4}});
    for (const Shape& s : shapes) {
        const Tensor A = random_tensor(s.m, s.k, rng);
        const QTensor Bq = quantize_q8(random_tensor(s.k, s.n, rng));
        const Tensor want = matmul_naive(A, dequantize(Bq));
        for (const Kernel& k : kKernels) {
            if (s.m > k.max_m) continue;
            if (!close_to(matmul_cuda(A, Bq, k.kernel), want, s.k)) {
                std::fprintf(stderr, "  [%s] q8 mismatch at m=%zu k=%zu n=%zu\n", k.name, s.m,
                             s.k, s.n);
                CHECK(false);
            }
        }
    }

    for (const Shape& s : {Shape{1, 288, 300}, Shape{5, 768, 288}, Shape{2, 20, 64},
                           Shape{3, 288, 30}, Shape{70, 40, 130}}) {
        const Tensor A = random_tensor(s.m, s.k, rng);
        const QTensor Bq = quantize_q8(random_tensor(s.k, s.n, rng));
        const Tensor C0 = random_tensor(s.m, s.n, rng);
        const Tensor w_host = random_tensor(1, s.k, rng);
        const DeviceTensor dA(A), dW(w_host);
        const DeviceQTensor dB(Bq);
        MatmulFusion f;
        f.accumulate = true;
        f.rms_weight = dW.data();
        const Tensor want = add(C0, matmul_naive(rmsnorm(A, w_host.data), dequantize(Bq)));
        for (const Kernel& k : kKernels) {
            if (s.m > k.max_m) continue;
            DeviceTensor dC(C0);
            matmul_cuda(s.m, s.n, s.k, dA.data(), dB.data(), dB.scale(), dC.data(), k.kernel, f);
            if (!close_to(dC.download(), want, 2 * s.k)) {
                std::fprintf(stderr, "  [%s] q8 fused mismatch at m=%zu k=%zu n=%zu\n", k.name,
                             s.m, s.k, s.n);
                CHECK(false);
            }
        }
    }
}

void test_bad_shapes() {
    const std::string msg = invalid_argument_of([] { matmul_cuda(Tensor(2, 3), Tensor(4, 2)); });
    CHECK(msg.find("2x3") != std::string::npos && msg.find("4x2") != std::string::npos);
    CHECK(!invalid_argument_of([] {
               matmul_cuda(Tensor(2, 3), quantize_q8(Tensor(4, 2)));
           }).empty());
    CHECK(!invalid_argument_of([] {
               matmul_cuda(Tensor(9, 2), Tensor(2, 2), GpuKernel::skinny);
           }).empty());

}

} // namespace

int main() try {
    if (!cuda_available()) {
        std::puts("no CUDA device, skipping");
        return 77;
    }
    test_against_reference();
    test_reuse();
    test_fusion();
    test_q8();
    test_bad_shapes();
    cuda_synchronize();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
} catch (const std::exception& e) {
    // A CUDA error leaves the device unusable, so report and stop.
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
