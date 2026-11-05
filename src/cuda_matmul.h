#pragma once
#include <cstddef>
#include <cstdint>

#include "tensor.h"

// Matmul on an NVIDIA GPU, fp32 activations times fp32 or int8 weights. Built only with -DENGINE_CUDA=ON. Calls run
// on the calling thread's default stream (the library is compiled with
// --default-stream per-thread, so that stream can be captured into a CUDA
// graph) and return before the GPU finishes, except download() and the
// Tensor overload of matmul_cuda(), which wait. Not safe to call from
// several host threads at once.

// Row-major fp32 matrix in GPU memory. Move-only; frees its memory on
// destruction.
class DeviceTensor {
public:
    DeviceTensor() = default;
    DeviceTensor(std::size_t rows, std::size_t cols);  // contents undefined
    explicit DeviceTensor(const Tensor& t);            // copies t to the GPU
    ~DeviceTensor();

    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&& other) noexcept;
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    // Copies t into this tensor, reallocating if the shape differs.
    void upload(const Tensor& t);
    Tensor download() const;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    float* data() { return data_; }
    const float* data() const { return data_; }

private:
    std::size_t rows_ = 0, cols_ = 0;
    float* data_ = nullptr;
};

// int8 matrix with one fp32 scale per column in GPU memory, as QTensor.
// Move-only; frees its memory on destruction.
class DeviceQTensor {
public:
    DeviceQTensor() = default;
    explicit DeviceQTensor(const QTensor& t);  // copies t to the GPU
    ~DeviceQTensor();

    DeviceQTensor(DeviceQTensor&& other) noexcept;
    DeviceQTensor& operator=(DeviceQTensor&& other) noexcept;
    DeviceQTensor(const DeviceQTensor&) = delete;
    DeviceQTensor& operator=(const DeviceQTensor&) = delete;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    const std::int8_t* data() const { return data_; }
    const float* scale() const { return scale_; }

private:
    std::size_t rows_ = 0, cols_ = 0;
    std::int8_t* data_ = nullptr;
    float* scale_ = nullptr;  // cols
};

enum class GpuKernel {
    automatic,    // skinny for m <= 8, tiled_split up to 64, tiled beyond
    naive,        // one thread per element of C; the reference
    tiled,        // 128 x 128 shared-memory tiles, 8 x 8 per thread
    tiled_split,  // 32 x 128 tiles, 2 x 8 per thread, k split across blocks
    skinny,       // m <= 8 only: split-K, one thread per column of C
};

// C = A x B; C is reallocated if it isn't m x n. Throws std::invalid_argument
// on a shape mismatch (or skinny with m > 8) and std::runtime_error on a
// CUDA error.
void matmul_cuda(const DeviceTensor& A, const DeviceTensor& B, DeviceTensor& C,
                 GpuKernel kernel = GpuKernel::automatic);

// Neighbouring ops folded into a product, so a decode step launches fewer
// kernels. The skinny kernel normalizes A while it reads it; the others run
// the norm as a separate pass first. (Fusing SwiGLU the same way was tried
// and dropped: recomputing silu in every column's load chain made the down
// projection 5.6 us slower, against a 1.1 us kernel saved.)
struct MatmulFusion {
    // C += A x B instead of C = A x B: a residual add.
    bool accumulate = false;
    // Multiply rmsnorm(A) instead of A: each row of A scaled by
    // 1 / sqrt(mean(row^2) + 1e-5) and by these k weights, as rmsnorm() in
    // ops.h.
    const float* rms_weight = nullptr;
};

// Same on raw device pointers: A is m x k, B k x n, C m x n, all dense and
// row-major. Throws std::invalid_argument for skinny with m > 8 or matrices
// past 32-bit indexing.
void matmul_cuda(std::size_t m, std::size_t n, std::size_t k, const float* A, const float* B,
                 float* C, GpuKernel kernel = GpuKernel::automatic,
                 const MatmulFusion& fusion = {});

// Same with int8 B: C = A x dequantize(B), B k x n int8 with n column
// scales. Every kernel widens and scales B as it loads it, so B is read at a
// quarter of the fp32 bytes and the arithmetic matches matmul_parallel()'s
// int8 path.
void matmul_cuda(std::size_t m, std::size_t n, std::size_t k, const float* A,
                 const std::int8_t* B, const float* scale, float* C,
                 GpuKernel kernel = GpuKernel::automatic, const MatmulFusion& fusion = {});

// Bumped whenever matmul_cuda() reallocates its internal split-K scratch. A
// CUDA graph captured before the bump points at freed memory and must be
// captured again.
std::size_t cuda_scratch_generation();

// Copies A and B to the GPU, multiplies, copies C back.
Tensor matmul_cuda(const Tensor& A, const Tensor& B, GpuKernel kernel = GpuKernel::automatic);
Tensor matmul_cuda(const Tensor& A, const QTensor& B, GpuKernel kernel = GpuKernel::automatic);

// Waits for all queued GPU work; throws std::runtime_error if any failed.
void cuda_synchronize();

// False if there is no usable CUDA device.
bool cuda_available();
