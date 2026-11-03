#include "cuda_matmul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(err));
}

std::string shape_str(const DeviceTensor& t) {
    return std::to_string(t.rows()) + "x" + std::to_string(t.cols());
}

constexpr int ceil_div(std::size_t x, int d) { return static_cast<int>((x + d - 1) / d); }

// Reference: thread (x, y) computes C[y][x]. Consecutive threads take
// consecutive columns, so each step of the k loop reads B coalesced.
__global__ void sgemm_naive(int m, int n, int k, const float* __restrict__ A,
                            const float* __restrict__ B, float* __restrict__ C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[row * k + p] * B[p * n + col];
    C[row * n + col] = acc;
}

// Shared-memory tiled SGEMM. A block computes a BM x BN tile of C, stepping
// through k BK at a time: the block stages a BM x BK slab of A and a BK x BN
// slab of B in shared memory, then each thread accumulates a TM x TN
// sub-tile in registers, so every value loaded from shared memory feeds TM
// or TN multiply-adds.
//
// A thread's rows and columns are strided by the thread grid (row ty + i *
// BM / TM, column tx + j * BN / TN) rather than contiguous: consecutive
// threads then read consecutive shared-memory words, with no bank
// conflicts, and store consecutive columns of C. The A slab is stored
// transposed, padded by 4 so the transposing store hits 8 distinct banks.
//
// With Split, blocks along blockIdx.z each take one slice of k, span deep,
// and write their partial C to out + z * m * n. Medium m (9..64) uses 32- and
// 64-row tiles split this way: 128-row tiles would be mostly padding there,
// and too few blocks to fill the GPU. The slice bookkeeping is compiled out
// of the unsplit kernel, where it cost 10% at N = 4096 (34.2 -> 37.6 ms).
constexpr int BN = 128, BK = 8, TN = 8;
constexpr int kTiledThreads = 256;

template <int BM, int TM, bool Split>
__global__ void __launch_bounds__(kTiledThreads)
sgemm_tiled(int m, int n, int k, int span, const float* __restrict__ A,
            const float* __restrict__ B, float* __restrict__ out) {
    static_assert((BM / TM) * (BN / TN) == kTiledThreads, "one thread per sub-tile");
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN];

    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int tx = threadIdx.x % (BN / TN), ty = threadIdx.x / (BN / TN);
    const int p_begin = Split ? blockIdx.z * span : 0;
    const int p_end = Split ? min(k, p_begin + span) : k;
    float* C = Split ? out + static_cast<std::size_t>(blockIdx.z) * m * n : out;

    float acc[TM][TN] = {};
    float a[TM], b[TN];
    for (int p0 = p_begin; p0 < p_end; p0 += BK) {
        // Out-of-range elements load as zero, so edge tiles need no
        // special case in the inner loop.
        for (int i = threadIdx.x; i < BM * BK; i += kTiledThreads) {
            const int r = i / BK, c = i % BK;
            const int gr = row0 + r, gc = p0 + c;
            As[c][r] = gr < m && gc < p_end ? A[gr * k + gc] : 0.0f;
        }
        for (int i = threadIdx.x; i < BK * BN; i += kTiledThreads) {
            const int r = i / BN, c = i % BN;
            const int gr = p0 + r, gc = col0 + c;
            Bs[r][c] = gr < p_end && gc < n ? B[gr * n + gc] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BK; ++p) {
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[p][ty + i * (BM / TM)];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[p][tx + j * (BN / TN)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + ty + i * (BM / TM);
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col0 + tx + j * (BN / TN);
            if (c < n) C[r * n + c] = acc[i][j];
        }
    }
}

// Skinny path, m <= 8: the GPU analogue of the CPU's split-K streaming path.
// With so few rows each element of B is used at most 8 times, so the product
// is bound by reading B. Thread c of a k slice streams column c of that
// slice; a warp reads 32 consecutive floats of a row of B per step. n
// columns alone are too few threads to keep enough loads in flight, so k is
// split across blockIdx.y and the slices' partial sums are added by a second
// kernel, in a fixed order (no atomics, so results are deterministic).
constexpr int kSkinnyMaxRows = 8;
constexpr int kSkinnyThreads = 256;

template <int M>
__global__ void __launch_bounds__(kSkinnyThreads)
sgemm_skinny(int n, int k, int span, const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const int p0 = blockIdx.y * span, p1 = min(k, p0 + span);
    float acc[M] = {};
    for (int p = p0; p < p1; ++p) {
        const float b = B[p * n + c];
#pragma unroll
        for (int r = 0; r < M; ++r) acc[r] += A[r * k + p] * b;
    }
#pragma unroll
    for (int r = 0; r < M; ++r) out[(blockIdx.y * M + r) * n + c] = acc[r];
}

// C[i] = sum over slices s of partial[s * mn + i].
__global__ void sum_slices(int mn, int slices, const float* __restrict__ partial,
                           float* __restrict__ C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn) return;
    float acc = 0.0f;
    for (int s = 0; s < slices; ++s) acc += partial[s * mn + i];
    C[i] = acc;
}

// Grow-only device scratch for split-K partial sums.
float* partial_buffer(std::size_t floats) {
    static float* buf = nullptr;
    static std::size_t cap = 0;
    if (floats > cap) {
        if (buf) check(cudaFree(buf), "free partials");
        check(cudaMalloc(&buf, floats * sizeof(float)), "alloc partials");
        cap = floats;
    }
    return buf;
}

template <int M>
void launch_skinny(int n, int k, int span, int slices, const float* A, const float* B,
                   float* out) {
    sgemm_skinny<M><<<dim3(ceil_div(n, kSkinnyThreads), slices), kSkinnyThreads>>>(n, k, span,
                                                                                     A, B, out);
}

int sm_count() {
    static const int sms = [] {
        int device = 0, n = 0;
        check(cudaGetDevice(&device), "get device");
        check(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device), "get SMs");
        return n;
    }();
    return sms;
}

template <int BM, int TM>
void run_tiled(int m, int n, int k, int slices, const float* A, const float* B, float* C) {
    const dim3 grid(ceil_div(n, BN), ceil_div(m, BM));
    const int span = ceil_div(std::size_t(ceil_div(k, slices)), BK) * BK;
    slices = ceil_div(k, span);
    if (slices == 1) {
        sgemm_tiled<BM, TM, false><<<grid, kTiledThreads>>>(m, n, k, k, A, B, C);
        return;
    }
    float* partial = partial_buffer(std::size_t(slices) * m * n);
    sgemm_tiled<BM, TM, true><<<dim3(grid.x, grid.y, slices), kTiledThreads>>>(m, n, k, span, A,
                                                                               B, partial);
    sum_slices<<<ceil_div(std::size_t(m) * n, 256), 256>>>(m * n, slices, partial, C);
}

// 32- or 64-row tiles, the smallest that m fills, with k split until there
// are about 8 blocks per SM, each slice at least 256 deep.
void run_tiled_split(int m, int n, int k, const float* A, const float* B, float* C) {
    const int rows = m <= 32 ? 32 : 64;
    const int blocks = ceil_div(n, BN) * ceil_div(std::size_t(m), rows);
    const int slices = std::max(1, std::min(ceil_div(8 * sm_count(), blocks), ceil_div(k, 256)));
    if (rows == 32) {
        run_tiled<32, 2>(m, n, k, slices, A, B, C);
    } else {
        run_tiled<64, 4>(m, n, k, slices, A, B, C);
    }
}

void run_skinny(int m, int n, int k, const float* A, const float* B, float* C) {
    // Aim for about 64 resident warps per SM over the whole grid, but keep
    // slices at least 32 rows of B deep so the partials stay a small
    // fraction of the traffic.
    const int want_threads = sm_count() * 2048;
    int slices = std::max(1, std::min(want_threads / std::max(n, 1), ceil_div(k, 32)));
    const int span = ceil_div(k, slices);
    slices = ceil_div(k, span);

    float* out = slices == 1 ? C : partial_buffer(std::size_t(slices) * m * n);
    using Launch = void (*)(int, int, int, int, const float*, const float*, float*);
    static constexpr Launch kLaunch[kSkinnyMaxRows] = {
        launch_skinny<1>, launch_skinny<2>, launch_skinny<3>, launch_skinny<4>,
        launch_skinny<5>, launch_skinny<6>, launch_skinny<7>, launch_skinny<8>};
    kLaunch[m - 1](n, k, span, slices, A, B, out);
    if (slices > 1) sum_slices<<<ceil_div(std::size_t(m) * n, 256), 256>>>(m * n, slices, out, C);
}

} // namespace

DeviceTensor::DeviceTensor(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {
    if (rows * cols) check(cudaMalloc(&data_, rows * cols * sizeof(float)), "alloc");
}

DeviceTensor::DeviceTensor(const Tensor& t) : DeviceTensor(t.rows, t.cols) { upload(t); }

DeviceTensor::~DeviceTensor() { cudaFree(data_); }

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept
    : rows_(std::exchange(other.rows_, 0)), cols_(std::exchange(other.cols_, 0)),
      data_(std::exchange(other.data_, nullptr)) {}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
    return *this;
}

void DeviceTensor::upload(const Tensor& t) {
    if (t.rows != rows_ || t.cols != cols_) *this = DeviceTensor(t.rows, t.cols);
    if (!t.data.empty())
        check(cudaMemcpy(data_, t.data.data(), t.data.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "upload");
}

Tensor DeviceTensor::download() const {
    Tensor t(rows_, cols_);
    if (!t.data.empty())
        check(cudaMemcpy(t.data.data(), data_, t.data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "download");
    return t;
}

void matmul_cuda(const DeviceTensor& A, const DeviceTensor& B, DeviceTensor& C,
                 GpuKernel kernel) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("matmul_cuda: inner dimensions differ (A is " +
                                    shape_str(A) + ", B is " + shape_str(B) + ")");
    const std::size_t m = A.rows(), n = B.cols(), k = A.cols();
    if (kernel == GpuKernel::automatic) {
        kernel = m <= kSkinnyMaxRows ? GpuKernel::skinny
                 : m <= 64           ? GpuKernel::tiled_split
                                     : GpuKernel::tiled;
    }
    if (kernel == GpuKernel::skinny && m > kSkinnyMaxRows)
        throw std::invalid_argument("matmul_cuda: skinny kernel takes at most 8 rows, A is " +
                                    shape_str(A));
    // Kernels index with int.
    constexpr std::size_t kMaxElems = 1u << 31;
    if (m * k >= kMaxElems || k * n >= kMaxElems || m * n >= kMaxElems)
        throw std::invalid_argument("matmul_cuda: more than 2^31 elements (A is " +
                                    shape_str(A) + ", B is " + shape_str(B) + ")");
    if (C.rows() != m || C.cols() != n) C = DeviceTensor(m, n);
    if (m == 0 || n == 0) return;
    if (k == 0) {
        check(cudaMemsetAsync(C.data(), 0, m * n * sizeof(float)), "memset");
        return;
    }

    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    switch (kernel) {
        case GpuKernel::naive:
            sgemm_naive<<<dim3(ceil_div(n, 16), ceil_div(m, 16)), dim3(16, 16)>>>(
                mi, ni, ki, A.data(), B.data(), C.data());
            break;
        case GpuKernel::tiled:
            run_tiled<128, 8>(mi, ni, ki, 1, A.data(), B.data(), C.data());
            break;
        case GpuKernel::tiled_split:
            run_tiled_split(mi, ni, ki, A.data(), B.data(), C.data());
            break;
        default:
            run_skinny(mi, ni, ki, A.data(), B.data(), C.data());
            break;
    }
    check(cudaGetLastError(), "launch");
}

Tensor matmul_cuda(const Tensor& A, const Tensor& B, GpuKernel kernel) {
    const DeviceTensor dA(A), dB(B);
    DeviceTensor dC;
    matmul_cuda(dA, dB, dC, kernel);
    return dC.download();
}

void cuda_synchronize() { check(cudaDeviceSynchronize(), "synchronize"); }

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
