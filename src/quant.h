#pragma once
#include "tensor.h"

// Symmetric per-column int8: scale[c] = max |B[:, c]| / 127 and each element
// rounds to the nearest multiple of it, so the error is at most scale[c] / 2.
// An all-zero column gets scale 0.
QTensor quantize_q8(const Tensor& B);

Tensor dequantize(const QTensor& B);
