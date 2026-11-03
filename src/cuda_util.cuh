#pragma once
// Helpers shared by the .cu files.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Throws std::runtime_error naming the failed operation.
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(err));
}

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};
struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

// Reduces v across the block; every thread gets the result. blockDim.x must
// be a multiple of 32, and every thread of the block must call it. identity
// pads the second, cross-warp stage.
template <class Op>
__device__ float block_reduce(float v, Op op, float identity) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();  // a previous call may still be reading part
    if (lane == 0) part[warp] = v;
    __syncthreads();
    v = lane < int(blockDim.x / 32) ? part[lane] : identity;
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// RMSNorm's eps, as in ops.h.
constexpr float kRmsEps = 1e-5f;
