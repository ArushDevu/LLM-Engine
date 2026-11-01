#pragma once
#include <vector>

#include "tensor.h"

// Row-wise building blocks for the transformer forward pass. Each row of a
// Tensor is one token's vector, so every op works on rows independently.

// y = x / sqrt(mean(x^2) + eps) * weight, per row. weight has one entry per
// column. Throws std::invalid_argument if weight.size() != x.cols.
Tensor rmsnorm(const Tensor& x, const std::vector<float>& weight, float eps = 1e-5f);

// In place: each row becomes exp(x - max) / sum(exp(x - max)). Subtracting
// the row max keeps exp() from overflowing on large logits.
void softmax_rows(Tensor& x);

// In place: x * sigmoid(x), the activation in Llama's feed-forward block.
void silu(Tensor& x);
