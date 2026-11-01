#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "ops.h"

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

bool near(float got, float want, float tol = 1e-5f) { return std::fabs(got - want) <= tol; }

Tensor from_rows(std::size_t rows, std::size_t cols, const std::vector<float>& values) {
    Tensor t(rows, cols);
    t.data = values;
    return t;
}

void test_rmsnorm_known_values() {
    // Row [3, 4]: mean(x^2) = 12.5, so scale = 1/sqrt(12.5).
    Tensor x = from_rows(2, 2, {3.0f, 4.0f, 0.0f, 0.0f});
    Tensor y = rmsnorm(x, {1.0f, 2.0f}, 0.0f);
    const float s = 1.0f / std::sqrt(12.5f);
    CHECK(near(y.at(0, 0), 3.0f * s));
    CHECK(near(y.at(0, 1), 4.0f * s * 2.0f));
    // eps keeps an all-zero row finite instead of dividing by zero.
    Tensor z = rmsnorm(x, {1.0f, 1.0f});
    CHECK(z.at(1, 0) == 0.0f && z.at(1, 1) == 0.0f);
}

// With unit weights, every output row has mean square 1.
void test_rmsnorm_unit_rms() {
    Tensor x = from_rows(1, 5, {-2.0f, 0.5f, 7.0f, 1.0f, -3.0f});
    Tensor y = rmsnorm(x, std::vector<float>(5, 1.0f), 0.0f);
    float ss = 0.0f;
    for (float v : y.data) ss += v * v;
    CHECK(near(ss / 5.0f, 1.0f));
}

void test_rmsnorm_weight_mismatch() {
    std::string message;
    try {
        rmsnorm(Tensor(2, 3), {1.0f, 1.0f});
    } catch (const std::invalid_argument& e) {
        message = e.what();
    }
    CHECK(message.find("2 entries") != std::string::npos);
    CHECK(message.find("3 columns") != std::string::npos);
}

void test_softmax_known_values() {
    Tensor x = from_rows(2, 3, {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f});
    softmax_rows(x);
    const float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f);
    const float sum = e1 + e2 + e3;
    CHECK(near(x.at(0, 0), e1 / sum));
    CHECK(near(x.at(0, 1), e2 / sum));
    CHECK(near(x.at(0, 2), e3 / sum));
    for (std::size_t c = 0; c < 3; ++c) CHECK(near(x.at(1, c), 1.0f / 3.0f));
}

// Logits this large overflow exp() unless the row max is subtracted first.
void test_softmax_large_logits() {
    Tensor x = from_rows(1, 2, {1000.0f, 1000.0f});
    softmax_rows(x);
    CHECK(near(x.at(0, 0), 0.5f));
    CHECK(near(x.at(0, 1), 0.5f));
}

void test_silu_known_values() {
    Tensor x = from_rows(1, 3, {0.0f, 1.0f, -1.0f});
    silu(x);
    CHECK(x.at(0, 0) == 0.0f);
    CHECK(near(x.at(0, 1), 1.0f / (1.0f + std::exp(-1.0f))));
    CHECK(near(x.at(0, 2), -1.0f / (1.0f + std::exp(1.0f))));
}

} // namespace

int main() {
    test_rmsnorm_known_values();
    test_rmsnorm_unit_rms();
    test_rmsnorm_weight_mismatch();
    test_softmax_known_values();
    test_softmax_large_logits();
    test_silu_known_values();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
}
