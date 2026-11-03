#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// For .cu files: throws std::runtime_error naming the failed operation.
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(err));
}
