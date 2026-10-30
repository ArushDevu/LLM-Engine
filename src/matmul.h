#pragma once
#include "tensor.h"

// All variants compute C = A x B   (A: m x k, B: k x n, C: m x n)
// and throw std::invalid_argument on a shape mismatch.

// Reference implementation. Kept simple on purpose: tests check the
// optimized kernels against it.
Tensor matmul_naive(const Tensor& A, const Tensor& B);

// Cache-blocked, single-threaded. Packs A and B into contiguous panels and
// runs a register-tiled micro-kernel over them (Goto/BLIS structure).
Tensor matmul_blocked(const Tensor& A, const Tensor& B);
