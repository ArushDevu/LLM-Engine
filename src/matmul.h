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

// Blocked kernel split across threads by rows of C. threads == 0 means
// std::thread::hardware_concurrency(). Small problems use fewer threads.
Tensor matmul_parallel(const Tensor& A, const Tensor& B, unsigned threads = 0);

// Same as above with int8 weights: C = A x dequantize(B). The int8 values are
// widened to fp32 inside the kernels, so B is read from memory at a quarter
// of the fp32 size.
Tensor matmul_parallel(const Tensor& A, const QTensor& B, unsigned threads = 0);
