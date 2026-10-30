#include <cstdio>
#include <stdexcept>

#include "matmul.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,      \
                         __LINE__, #cond);                                   \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

template <class MatmulFn>
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

template <class MatmulFn>
void test_shape_mismatch(MatmulFn matmul) {
    bool threw = false;
    try {
        matmul(Tensor(2, 3), Tensor(4, 2));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_known_values(matmul_naive);
    test_shape_mismatch(matmul_naive);

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
}
