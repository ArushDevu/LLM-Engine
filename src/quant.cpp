#include "quant.h"

#include <algorithm>
#include <cmath>

QTensor quantize_q8(const Tensor& B) {
    QTensor q;
    q.rows = B.rows;
    q.cols = B.cols;
    q.data.resize(B.data.size());
    q.scale.assign(B.cols, 0.0f);
    for (std::size_t r = 0; r < B.rows; ++r)
        for (std::size_t c = 0; c < B.cols; ++c)
            q.scale[c] = std::max(q.scale[c], std::fabs(B.at(r, c)));
    std::vector<float> inv(B.cols);
    for (std::size_t c = 0; c < B.cols; ++c) {
        q.scale[c] /= 127.0f;
        inv[c] = q.scale[c] > 0.0f ? 1.0f / q.scale[c] : 0.0f;
    }
    for (std::size_t r = 0; r < B.rows; ++r) {
        for (std::size_t c = 0; c < B.cols; ++c) {
            // |B * inv| <= 127 up to rounding of inv; the clamp covers that.
            const long v = std::lround(B.at(r, c) * inv[c]);
            q.data[r * B.cols + c] = static_cast<std::int8_t>(std::clamp(v, -127L, 127L));
        }
    }
    return q;
}

Tensor dequantize(const QTensor& B) {
    Tensor t(B.rows, B.cols);
    for (std::size_t r = 0; r < B.rows; ++r)
        for (std::size_t c = 0; c < B.cols; ++c)
            t.at(r, c) = static_cast<float>(B.data[r * B.cols + c]) * B.scale[c];
    return t;
}
