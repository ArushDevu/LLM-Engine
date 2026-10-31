#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#include "matmul.h"

namespace {

int failures = 0;

// Name of the kernel under test, printed with every failure so the output
// says which implementation broke, not just which line.
const char* current_kernel = "";

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "%s:%d: [%s] CHECK failed: %s\n", __FILE__, \
                         __LINE__, current_kernel, #cond);                   \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

using MatmulFn = Tensor (*)(const Tensor&, const Tensor&);

struct Kernel {
    const char* name;
    MatmulFn fn;
};

const Kernel kKernels[] = {
    {"naive", matmul_naive},
    {"blocked", matmul_blocked},
    {"parallel", [](const Tensor& A, const Tensor& B) { return matmul_parallel(A, B); }},
};

void test_known_values(MatmulFn matmul) {
    Tensor A(2, 3), B(3, 2);
    float v = 1.0f;
    for (auto& x : A.data) x = v++;   // A = [[1,2,3],[4,5,6]]
    for (auto& x : B.data) x = v++;   // B = [[7,8],[9,10],[11,12]]

    Tensor C = matmul(A, B);
    CHECK(C.rows == 2 && C.cols == 2);
    CHECK(C.at(0, 0) == 58.0f);
    CHECK(C.at(0, 1) == 64.0f);
    CHECK(C.at(1, 0) == 139.0f);
    CHECK(C.at(1, 1) == 154.0f);
}

// The error should name both shapes so a caller can see what went wrong
// without a debugger.
void test_shape_mismatch(MatmulFn matmul) {
    std::string message;
    try {
        matmul(Tensor(2, 3), Tensor(4, 2));
    } catch (const std::invalid_argument& e) {
        message = e.what();
    }
    CHECK(!message.empty());
    CHECK(message.find("2x3") != std::string::npos);
    CHECK(message.find("4x2") != std::string::npos);
}

Tensor random_tensor(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t(rows, cols);
    for (auto& x : t.data) x = dist(rng);
    return t;
}

// The optimized kernels sum in a different order than the reference, so
// results differ by rounding. Error grows with k, so scale the tolerance by it.
bool close_to(const Tensor& got, const Tensor& want, std::size_t k) {
    if (got.rows != want.rows || got.cols != want.cols) return false;
    const float tol = 1e-5f * static_cast<float>(std::max<std::size_t>(k, 1));
    for (std::size_t i = 0; i < got.data.size(); ++i) {
        if (std::fabs(got.data[i] - want.data[i]) > tol) return false;
    }
    return true;
}

// Shapes chosen to land on and around every tile/block boundary
// (MR=4, NR=16, KC=256, MC=64, NC=1024), including degenerate 1-wide cases.
struct Shape { std::size_t m, k, n; };
constexpr Shape kShapes[] = {
    {1, 1, 1},    {1, 17, 1},    {3, 5, 7},      {4, 256, 16},
    {5, 257, 17}, {63, 300, 65}, {64, 255, 128}, {130, 70, 1030},
    {200, 513, 33},
};

void test_against_reference(MatmulFn matmul) {
    std::mt19937 rng(42);
    for (const Shape& s : kShapes) {
        Tensor A = random_tensor(s.m, s.k, rng);
        Tensor B = random_tensor(s.k, s.n, rng);
        if (!close_to(matmul(A, B), matmul_naive(A, B), s.k)) {
            std::fprintf(stderr, "  mismatch at m=%zu k=%zu n=%zu\n", s.m, s.k, s.n);
            CHECK(false);
        }
    }
}

// Thread counts that do and don't divide the row count evenly, plus more
// threads than there are row blocks.
void test_parallel_thread_counts() {
    std::mt19937 rng(7);
    Tensor A = random_tensor(301, 129, rng);
    Tensor B = random_tensor(129, 47, rng);
    const Tensor want = matmul_naive(A, B);
    for (unsigned t : {1u, 2u, 3u, 5u, 64u, 0u}) {
        if (!close_to(matmul_parallel(A, B, t), want, A.cols)) {
            std::fprintf(stderr, "  mismatch with threads=%u\n", t);
            CHECK(false);
        }
    }
}

} // namespace

int main() {
    for (const Kernel& kernel : kKernels) {
        current_kernel = kernel.name;
        test_known_values(kernel.fn);
        test_shape_mismatch(kernel.fn);
        // Comparing the reference against itself proves nothing.
        if (kernel.fn != matmul_naive) test_against_reference(kernel.fn);
    }

    current_kernel = "parallel";
    test_parallel_thread_counts();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
}
