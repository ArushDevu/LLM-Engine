#include "ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

Tensor rmsnorm(const Tensor& x, const std::vector<float>& weight, float eps) {
    if (weight.size() != x.cols) {
        throw std::invalid_argument("rmsnorm: weight has " + std::to_string(weight.size()) +
                                    " entries, x has " + std::to_string(x.cols) + " columns");
    }
    Tensor y(x.rows, x.cols);
    if (x.cols == 0) return y;
    for (std::size_t r = 0; r < x.rows; ++r) {
        const float* in = x.data.data() + r * x.cols;
        float* out = y.data.data() + r * x.cols;
        // Accumulate in double: a long row of floats loses precision otherwise.
        double ss = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) ss += double(in[c]) * in[c];
        const float scale = static_cast<float>(
            1.0 / std::sqrt(ss / static_cast<double>(x.cols) + eps));
        for (std::size_t c = 0; c < x.cols; ++c) out[c] = in[c] * scale * weight[c];
    }
    return y;
}

void softmax_rows(Tensor& x) {
    if (x.cols == 0) return;
    for (std::size_t r = 0; r < x.rows; ++r) {
        float* row = x.data.data() + r * x.cols;
        const float max = *std::max_element(row, row + x.cols);
        float sum = 0.0f;
        for (std::size_t c = 0; c < x.cols; ++c) {
            row[c] = std::exp(row[c] - max);
            sum += row[c];
        }
        // sum >= 1: the max element contributes exp(0).
        const float inv = 1.0f / sum;
        for (std::size_t c = 0; c < x.cols; ++c) row[c] *= inv;
    }
}

void silu(Tensor& x) {
    for (float& v : x.data) v = v / (1.0f + std::exp(-v));
}
