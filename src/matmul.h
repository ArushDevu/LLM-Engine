#pragma once
#include "tensor.h"

// All variants compute C = A x B   (A: m x k, B: k x n, C: m x n)
// and throw std::invalid_argument on a shape mismatch.

// Reference implementation. Kept simple on purpose: tests check the
// optimized kernels against it.
Tensor matmul_naive(const Tensor& A, const Tensor& B);
