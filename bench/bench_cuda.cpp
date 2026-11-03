// GPU matmul throughput against two ceilings: cuBLAS SGEMM for compute, and
// a device-to-device copy for memory bandwidth.
//
//   bench_cuda
//
// Each cell is the median of several runs, each timed on the host from
// launch to cudaDeviceSynchronize(), so launch overhead (~5-10 us) is
// included; that's what a caller issuing one product at a time pays.
// cuBLAS is a yardstick only, nothing in the engine uses it.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_matmul.h"

namespace {

using Clock = std::chrono::steady_clock;

void check_cublas(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublas error " + std::to_string(static_cast<int>(s)));
}

Tensor random_tensor(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t(rows, cols);
    for (auto& x : t.data) x = dist(rng);
    return t;
}

template <class F>
double median_seconds(F&& f, int reps) {
    f();  // warm-up
    cuda_synchronize();
    std::vector<double> times;
    for (int r = 0; r < reps; ++r) {
        const auto start = Clock::now();
        f();
        cuda_synchronize();
        times.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    std::nth_element(times.begin(), times.begin() + reps / 2, times.end());
    return times[reps / 2];
}

// Row-major C = A x B is column-major C^T = B^T x A^T, so cuBLAS gets the
// operands swapped and no transposes.
void cublas_matmul(cublasHandle_t h, const DeviceTensor& A, const DeviceTensor& B,
                   DeviceTensor& C) {
    const float one = 1.0f, zero = 0.0f;
    const int m = int(A.rows()), n = int(B.cols()), k = int(A.cols());
    check_cublas(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, B.data(), n, A.data(),
                             k, &zero, C.data(), n));
}

} // namespace

int main() try {
    if (!cuda_available()) {
        std::puts("no CUDA device");
        return 1;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle));
    std::mt19937 rng(0);

    // Bandwidth ceiling: a copy reads and writes every byte.
    {
        constexpr std::size_t kBytes = std::size_t{256} << 20;
        DeviceTensor src(kBytes / sizeof(float), 1), dst(kBytes / sizeof(float), 1);
        const double secs = median_seconds(
            [&] { cudaMemcpyAsync(dst.data(), src.data(), kBytes, cudaMemcpyDeviceToDevice); },
            21);
        std::printf("%s, %d SMs; device copy %.1f GB/s (read + write)\n\n", prop.name,
                    prop.multiProcessorCount, 2.0 * double(kBytes) / secs / 1e9);
    }

    std::printf("square fp32 matmul, kernel only; cells are ms / GFLOPS (median)\n\n");
    std::printf("%6s | %17s | %17s | %17s\n", "N", "naive", "tiled", "cuBLAS");
    std::printf("-------+-------------------+-------------------+------------------\n");
    for (std::size_t n : {512, 1024, 2048, 4096}) {
        const DeviceTensor A(random_tensor(n, n, rng)), B(random_tensor(n, n, rng));
        DeviceTensor C(n, n);
        const double flops = 2.0 * double(n) * double(n) * double(n);
        const int reps = n <= 1024 ? 51 : 11;
        std::printf("%6zu", n);
        for (int which = 0; which < 3; ++which) {
            const double secs = median_seconds(
                [&] {
                    if (which == 0) matmul_cuda(A, B, C, GpuKernel::naive);
                    if (which == 1) matmul_cuda(A, B, C, GpuKernel::tiled);
                    if (which == 2) cublas_matmul(handle, A, B, C);
                },
                reps);
            std::printf(" | %8.3f %8.0f", secs * 1e3, flops / secs / 1e9);
        }
        std::printf("\n");
    }

    constexpr std::size_t kDim = 4096;
    std::printf("\nm x %zu times %zu x %zu; cells are ms / GB/s of B. 'with copies' also\n"
                "uploads A and downloads C (B stays on the GPU, as weights would)\n\n",
                kDim, kDim, kDim);
    std::printf("%6s | %17s | %17s | %17s\n", "m", "engine", "cuBLAS", "engine with copies");
    std::printf("-------+-------------------+-------------------+------------------\n");
    const Tensor W = random_tensor(kDim, kDim, rng);
    const DeviceTensor dW(W);
    const double w_bytes = double(W.data.size() * sizeof(float));
    for (std::size_t m : {1, 4, 8, 16, 32, 64}) {
        const Tensor X = random_tensor(m, kDim, rng);
        DeviceTensor dX(X), dC(m, kDim);
        const double ours = median_seconds([&] { matmul_cuda(dX, dW, dC); }, 101);
        const double blas = median_seconds([&] { cublas_matmul(handle, dX, dW, dC); }, 101);
        const double copies = median_seconds(
            [&] {
                dX.upload(X);
                matmul_cuda(dX, dW, dC);
                dC.download();
            },
            101);
        std::printf("%6zu | %8.3f %8.1f | %8.3f %8.1f | %8.3f %8.1f\n", m, ours * 1e3,
                    w_bytes / ours / 1e9, blas * 1e3, w_bytes / blas / 1e9, copies * 1e3,
                    w_bytes / copies / 1e9);
    }
    cublasDestroy(handle);
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
