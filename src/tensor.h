#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

// A 2D matrix stored as one flat array (row-major).
// Element (r, c) lives at index r * cols + c.
struct Tensor {
    std::size_t rows;
    std::size_t cols;
    std::vector<float> data;

    Tensor() : rows(0), cols(0) {}
    Tensor(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

    float& at(std::size_t r, std::size_t c)       { return data[r * cols + c]; }
    float  at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

// int8 matrix with one fp32 scale per column: element (r, c) stands for
// data[r * cols + c] * scale[c]. Weights are stored (in x out), so a column
// is one output channel. See quantize_q8() in quant.h.
struct QTensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<std::int8_t> data;
    std::vector<float> scale;  // cols
};
