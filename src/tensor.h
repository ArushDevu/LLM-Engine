#pragma once
#include <vector>
#include <cstddef>

// A 2D matrix stored as one flat array (row-major).
// Element (r, c) lives at index r * cols + c.
struct Tensor {
    std::size_t rows;
    std::size_t cols;
    std::vector<float> data;

    Tensor(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

    float& at(std::size_t r, std::size_t c)       { return data[r * cols + c]; }
    float  at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};
