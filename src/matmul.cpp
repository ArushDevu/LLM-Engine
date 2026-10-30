#include "matmul.h"

#include <stdexcept>

namespace {

void check_shapes(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) throw std::invalid_argument("matmul: shape mismatch");
}

} // namespace

// Loop order i-k-j: the inner loop walks memory in a straight line (cache-friendly).
Tensor matmul_naive(const Tensor& A, const Tensor& B) {
    check_shapes(A, B);
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
