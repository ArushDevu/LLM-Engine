#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>

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

// C = A x B   (A: m x k, B: k x n, C: m x n)
// Loop order i-k-j: the inner loop walks memory in a straight line (cache-friendly).
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) throw std::invalid_argument("matmul: shape mismatch");
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