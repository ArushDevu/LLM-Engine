#include "cuda_matmul.h"

#include "cuda_util.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

std::string shape_str(const DeviceTensor& t) {
    return std::to_string(t.rows()) + "x" + std::to_string(t.cols());
}

constexpr int ceil_div(std::size_t x, int d) { return static_cast<int>((x + d - 1) / d); }

// B is fp32, or int8 with one scale per column (null for fp32), as on the
// CPU; the kernels are templates over the element type. int8 is widened and
// scaled as it is loaded, so the arithmetic is fp32 either way and matches
// matmul_parallel(), which dequantizes the same way when it packs B.
template <class BT>
__device__ __forceinline__ float load_b(const BT* __restrict__ B, int idx,
                                        const float* __restrict__ scale, int col) {
    if constexpr (std::is_same_v<BT, float>) {
        return B[idx];
    } else {
        return static_cast<float>(B[idx]) * scale[col];
    }
}

// Reference: thread (x, y) computes C[y][x]. Consecutive threads take
// consecutive columns, so each step of the k loop reads B coalesced.
template <class BT>
__global__ void sgemm_naive(int m, int n, int k, const float* __restrict__ A,
                            const BT* __restrict__ B, const float* __restrict__ scale,
                            float* __restrict__ C, bool accumulate) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[row * k + p] * load_b(B, p * n + col, scale, col);
    float* c = C + row * n + col;
    *c = accumulate ? *c + acc : acc;
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
// accumulate (C += A x B) applies to the unsplit kernel's store; split
// partials are added into C by sum_slices.
constexpr int BN = 128, BK = 8, TN = 8;
constexpr int kTiledThreads = 256;

template <int BM, int TM, bool Split, class BT>
__global__ void __launch_bounds__(kTiledThreads)
sgemm_tiled(int m, int n, int k, int span, const float* __restrict__ A,
            const BT* __restrict__ B, const float* __restrict__ scale,
            float* __restrict__ out, bool accumulate) {
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
            Bs[r][c] = gr < p_end && gc < n ? load_b(B, gr * n + gc, scale, gc) : 0.0f;
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
            if (c < n) C[r * n + c] = accumulate ? C[r * n + c] + acc[i][j] : acc[i][j];
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
//
// With Norm, the skinny kernel also applies MatmulFusion's RMSNorm while it
// reads A, so a decode step needs no separate norm kernel: each block first
// computes its rows' scales over all of k (A is at most 8 rows), which cost
// +0.6 us on a 5-10 us product against a 1.7 us kernel saved.
constexpr int kSkinnyMaxRows = 8;
constexpr int kSkinnyThreads = 256;

// RMSNorm scales of A's M rows over all of k. Call before any thread drops
// out: block_reduce needs the whole block.
template <int M>
__device__ __forceinline__ void rms_row_scales(const float* __restrict__ A, int k,
                                               float (&row_scale)[M]) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
        float ss = 0.0f;
        for (int j = threadIdx.x; j < k; j += blockDim.x) ss += A[r * k + j] * A[r * k + j];
        row_scale[r] = 1.0f / sqrtf(block_reduce(ss, SumOp{}, 0.0f) / float(k) + kRmsEps);
    }
}

template <int M, bool Norm, class BT>
__global__ void __launch_bounds__(kSkinnyThreads)
sgemm_skinny(int n, int k, int span, const float* __restrict__ A,
             const float* __restrict__ rms_weight, const BT* __restrict__ B,
             const float* __restrict__ scale, float* __restrict__ out, bool accumulate) {
    [[maybe_unused]] float row_scale[M];
    if constexpr (Norm) rms_row_scales<M>(A, k, row_scale);
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const int p0 = blockIdx.y * span, p1 = min(k, p0 + span);
    float acc[M] = {};
    for (int p = p0; p < p1; ++p) {
        const float b = load_b(B, p * n + c, scale, c);
#pragma unroll
        for (int r = 0; r < M; ++r) {
            float a = A[r * k + p];
            if constexpr (Norm) a = a * row_scale[r] * rms_weight[p];
            acc[r] += a * b;
        }
    }
#pragma unroll
    for (int r = 0; r < M; ++r) {
        float* o = out + (blockIdx.y * M + r) * n + c;
        *o = accumulate ? *o + acc[r] : acc[r];
    }
}

// The skinny kernel for int8 B when 4 divides n. One byte per thread per
// step left each warp reading only 32 bytes of a row of B at a time; here a
// thread takes 4 consecutive columns as one 32-bit load, so a warp reads 128
// (m = 1-8 products 1.7-2x faster). 8 or 16 columns per thread were slower:
// fewer threads in flight and M x V accumulators. A thread's columns keep
// their scales over the whole k loop, so it sums raw int8 values and scales
// each sum once at the end, as the CPU streaming path does (~3% of the gain).
constexpr int kQ8Cols = 4;

template <int M, bool Norm>
__global__ void __launch_bounds__(kSkinnyThreads)
sgemm_skinny_q8(int n, int k, int span, const float* __restrict__ A,
                const float* __restrict__ rms_weight, const std::int8_t* __restrict__ B,
                const float* __restrict__ scale, float* __restrict__ out, bool accumulate) {
    constexpr int V = kQ8Cols;
    [[maybe_unused]] float row_scale[M];
    if constexpr (Norm) rms_row_scales<M>(A, k, row_scale);
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (c >= n) return;
    const int p0 = blockIdx.y * span, p1 = min(k, p0 + span);
    float acc[M][V] = {};
    for (int p = p0; p < p1; ++p) {
        const int w = __ldg(reinterpret_cast<const int*>(B + p * n + c));
        float b[V];
#pragma unroll
        for (int j = 0; j < V; ++j) b[j] = static_cast<float>((w << (24 - 8 * j)) >> 24);
#pragma unroll
        for (int r = 0; r < M; ++r) {
            float a = A[r * k + p];
            if constexpr (Norm) a = a * row_scale[r] * rms_weight[p];
#pragma unroll
            for (int j = 0; j < V; ++j) acc[r][j] += a * b[j];
        }
    }
#pragma unroll
    for (int j = 0; j < V; ++j) {
        const float s = scale[c + j];
#pragma unroll
        for (int r = 0; r < M; ++r) {
            float* o = out + (blockIdx.y * M + r) * n + c + j;
            *o = accumulate ? *o + acc[r][j] * s : acc[r][j] * s;
        }
    }
}

// C[i] = sum over slices s of partial[s * mn + i], added to C[i] if
// accumulate.
__global__ void sum_slices(int mn, int slices, const float* __restrict__ partial,
                           float* __restrict__ C, bool accumulate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn) return;
    float acc = 0.0f;
    for (int s = 0; s < slices; ++s) acc += partial[s * mn + i];
    C[i] = accumulate ? C[i] + acc : acc;
}

// The norm as a separate pass, for kernels that don't fuse it:
// out = rmsnorm(A) * rms_weight, one block per row.
__global__ void rmsnorm_rows(int k, const float* __restrict__ A,
                             const float* __restrict__ rms_weight, float* __restrict__ out) {
    const float* in = A + static_cast<std::size_t>(blockIdx.x) * k;
    float* o = out + static_cast<std::size_t>(blockIdx.x) * k;
    float ss = 0.0f;
    for (int j = threadIdx.x; j < k; j += blockDim.x) ss += in[j] * in[j];
    const float scale = 1.0f / sqrtf(block_reduce(ss, SumOp{}, 0.0f) / float(k) + kRmsEps);
    for (int j = threadIdx.x; j < k; j += blockDim.x) o[j] = in[j] * scale * rms_weight[j];
}

// Grow-only device scratch: split-K partial sums, and A after a separate
// norm pass. Each growth bumps the generation, since it moves a buffer.
std::size_t g_scratch_generation = 0;

class Scratch {
public:
    float* get(std::size_t floats) {
        if (floats > cap_) {
            if (buf_) cuda_check(cudaFree(buf_), "free scratch");
            cuda_check(cudaMalloc(&buf_, floats * sizeof(float)), "alloc scratch");
            cap_ = floats;
            ++g_scratch_generation;
        }
        return buf_;
    }

private:
    float* buf_ = nullptr;
    std::size_t cap_ = 0;
};

float* partial_buffer(std::size_t floats) {
    static Scratch s;
    return s.get(floats);
}

float* norm_buffer(std::size_t floats) {
    static Scratch s;
    return s.get(floats);
}

// vec: int8 B with 4 | n and aligned, run by sgemm_skinny_q8.
template <bool Norm, class BT>
void launch_skinny(int m, dim3 grid, int n, int k, int span, const float* A,
                   const float* rms_weight, const BT* B, const float* scale, float* out,
                   bool accumulate, bool vec) {
    switch (m) {
#define ENGINE_SKINNY_CASE(M)                                                                  \
    case M:                                                                                    \
        if constexpr (std::is_same_v<BT, std::int8_t>) {                                       \
            if (vec) {                                                                         \
                sgemm_skinny_q8<M, Norm><<<grid, kSkinnyThreads>>>(n, k, span, A, rms_weight,  \
                                                                B, scale, out, accumulate);    \
                break;                                                                         \
            }                                                                                  \
        }                                                                                      \
        sgemm_skinny<M, Norm><<<grid, kSkinnyThreads>>>(n, k, span, A, rms_weight, B, scale,   \
                                                     out, accumulate);                         \
        break;
        ENGINE_SKINNY_CASE(1) ENGINE_SKINNY_CASE(2) ENGINE_SKINNY_CASE(3) ENGINE_SKINNY_CASE(4)
        ENGINE_SKINNY_CASE(5) ENGINE_SKINNY_CASE(6) ENGINE_SKINNY_CASE(7) ENGINE_SKINNY_CASE(8)
#undef ENGINE_SKINNY_CASE
    }
}

int sm_count() {
    static const int sms = [] {
        int device = 0, n = 0;
        cuda_check(cudaGetDevice(&device), "get device");
        cuda_check(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device), "get SMs");
        return n;
    }();
    return sms;
}

template <int BM, int TM, class BT>
void run_tiled(int m, int n, int k, int slices, const float* A, const BT* B, const float* scale,
               float* C, bool accumulate) {
    const dim3 grid(ceil_div(n, BN), ceil_div(m, BM));
    const int span = ceil_div(std::size_t(ceil_div(k, slices)), BK) * BK;
    slices = ceil_div(k, span);
    if (slices == 1) {
        sgemm_tiled<BM, TM, false><<<grid, kTiledThreads>>>(m, n, k, k, A, B, scale, C,
                                                            accumulate);
        return;
    }
    float* partial = partial_buffer(std::size_t(slices) * m * n);
    sgemm_tiled<BM, TM, true><<<dim3(grid.x, grid.y, slices), kTiledThreads>>>(
        m, n, k, span, A, B, scale, partial, false);
    sum_slices<<<ceil_div(std::size_t(m) * n, 256), 256>>>(m * n, slices, partial, C, accumulate);
}

// 32- or 64-row tiles, the smallest that m fills, with k split until there
// are about 8 blocks per SM, each slice at least 256 deep.
template <class BT>
void run_tiled_split(int m, int n, int k, const float* A, const BT* B, const float* scale,
                     float* C, bool accumulate) {
    const int rows = m <= 32 ? 32 : 64;
    const int blocks = ceil_div(n, BN) * ceil_div(std::size_t(m), rows);
    const int slices = std::max(1, std::min(ceil_div(8 * sm_count(), blocks), ceil_div(k, 256)));
    if (rows == 32) {
        run_tiled<32, 2>(m, n, k, slices, A, B, scale, C, accumulate);
    } else {
        run_tiled<64, 4>(m, n, k, slices, A, B, scale, C, accumulate);
    }
}

template <class BT>
void run_skinny(int m, int n, int k, const float* A, const float* rms_weight, const BT* B,
                const float* scale, float* C, bool accumulate) {
    // Four-byte loads need every row of B 4-byte aligned.
    const bool vec = std::is_same_v<BT, std::int8_t> && n % kQ8Cols == 0 &&
                     reinterpret_cast<std::uintptr_t>(B) % kQ8Cols == 0;
    const int col_threads = vec ? n / kQ8Cols : n;

    // Aim for about 64 resident warps per SM over the whole grid, but keep
    // slices at least 32 rows of B deep so the partials stay a small
    // fraction of the traffic.
    const int want_threads = sm_count() * 2048;
    int slices = std::max(1, std::min(want_threads / std::max(col_threads, 1), ceil_div(k, 32)));
    const int span = ceil_div(k, slices);
    slices = ceil_div(k, span);

    // With one slice the kernel writes C itself, accumulating if asked.
    float* out = slices == 1 ? C : partial_buffer(std::size_t(slices) * m * n);
    const bool direct_accumulate = slices == 1 && accumulate;
    const dim3 grid(ceil_div(std::size_t(col_threads), kSkinnyThreads), slices);
    if (rms_weight) {
        launch_skinny<true>(m, grid, n, k, span, A, rms_weight, B, scale, out, direct_accumulate,
                            vec);
    } else {
        launch_skinny<false>(m, grid, n, k, span, A, rms_weight, B, scale, out, direct_accumulate,
                             vec);
    }
    if (slices > 1)
        sum_slices<<<ceil_div(std::size_t(m) * n, 256), 256>>>(m * n, slices, out, C, accumulate);
}

template <class BT>
void matmul_impl(std::size_t m, std::size_t n, std::size_t k, const float* A, const BT* B,
                 const float* scale, float* C, GpuKernel kernel, const MatmulFusion& fusion) {
    auto shapes = [&] {
        return "(A is " + std::to_string(m) + "x" + std::to_string(k) + ", B is " +
               std::to_string(k) + "x" + std::to_string(n) + ")";
    };
    if (kernel == GpuKernel::automatic) {
        kernel = m <= kSkinnyMaxRows ? GpuKernel::skinny
                 : m <= 64           ? GpuKernel::tiled_split
                                     : GpuKernel::tiled;
    }
    if (kernel == GpuKernel::skinny && m > kSkinnyMaxRows)
        throw std::invalid_argument("matmul_cuda: skinny kernel takes at most 8 rows " +
                                    shapes());
    // Kernels index with int.
    constexpr std::size_t kMaxElems = std::size_t{1} << 31;
    if (m * k >= kMaxElems || k * n >= kMaxElems || m * n >= kMaxElems)
        throw std::invalid_argument("matmul_cuda: more than 2^31 elements " + shapes());
    if (m == 0 || n == 0) return;
    if (k == 0) {
        if (!fusion.accumulate) cuda_check(cudaMemsetAsync(C, 0, m * n * sizeof(float)), "memset");
        return;
    }

    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    const float* rms_weight = fusion.rms_weight;
    if (rms_weight && kernel != GpuKernel::skinny) {
        float* a = norm_buffer(m * k);
        rmsnorm_rows<<<mi, 256>>>(ki, A, rms_weight, a);
        A = a;
        rms_weight = nullptr;
    }
    switch (kernel) {
        case GpuKernel::naive:
            sgemm_naive<<<dim3(ceil_div(n, 16), ceil_div(m, 16)), dim3(16, 16)>>>(
                mi, ni, ki, A, B, scale, C, fusion.accumulate);
            break;
        case GpuKernel::tiled:
            run_tiled<128, 8>(mi, ni, ki, 1, A, B, scale, C, fusion.accumulate);
            break;
        case GpuKernel::tiled_split:
            run_tiled_split(mi, ni, ki, A, B, scale, C, fusion.accumulate);
            break;
        default:
            run_skinny(mi, ni, ki, A, rms_weight, B, scale, C, fusion.accumulate);
            break;
    }
    cuda_check(cudaGetLastError(), "launch");
}

} // namespace

DeviceTensor::DeviceTensor(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {
    if (rows * cols) cuda_check(cudaMalloc(&data_, rows * cols * sizeof(float)), "alloc");
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
        cuda_check(cudaMemcpy(data_, t.data.data(), t.data.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload");
}

Tensor DeviceTensor::download() const {
    Tensor t(rows_, cols_);
    if (!t.data.empty())
        cuda_check(cudaMemcpy(t.data.data(), data_, t.data.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "download");
    return t;
}

DeviceQTensor::DeviceQTensor(const QTensor& t) : rows_(t.rows), cols_(t.cols) {
    if (t.data.size() != rows_ * cols_ || t.scale.size() != cols_)
        throw std::invalid_argument("DeviceQTensor: data or scale size doesn't match the shape");
    try {
        if (!t.data.empty()) {
            cuda_check(cudaMalloc(&data_, t.data.size()), "alloc");
            cuda_check(cudaMemcpy(data_, t.data.data(), t.data.size(), cudaMemcpyHostToDevice),
                       "upload");
        }
        if (!t.scale.empty()) {
            cuda_check(cudaMalloc(&scale_, t.scale.size() * sizeof(float)), "alloc");
            cuda_check(cudaMemcpy(scale_, t.scale.data(), t.scale.size() * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       "upload");
        }
    } catch (...) {
        cudaFree(data_);
        cudaFree(scale_);
        throw;
    }
}

DeviceQTensor::~DeviceQTensor() {
    cudaFree(data_);
    cudaFree(scale_);
}

DeviceQTensor::DeviceQTensor(DeviceQTensor&& other) noexcept
    : rows_(std::exchange(other.rows_, 0)), cols_(std::exchange(other.cols_, 0)),
      data_(std::exchange(other.data_, nullptr)), scale_(std::exchange(other.scale_, nullptr)) {}

DeviceQTensor& DeviceQTensor::operator=(DeviceQTensor&& other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
    std::swap(scale_, other.scale_);
    return *this;
}

void matmul_cuda(const DeviceTensor& A, const DeviceTensor& B, DeviceTensor& C,
                 GpuKernel kernel) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("matmul_cuda: inner dimensions differ (A is " +
                                    shape_str(A) + ", B is " + shape_str(B) + ")");
    if (C.rows() != A.rows() || C.cols() != B.cols()) C = DeviceTensor(A.rows(), B.cols());
    matmul_cuda(A.rows(), B.cols(), A.cols(), A.data(), B.data(), C.data(), kernel);
}

void matmul_cuda(std::size_t m, std::size_t n, std::size_t k, const float* A, const float* B,
                 float* C, GpuKernel kernel, const MatmulFusion& fusion) {
    matmul_impl(m, n, k, A, B, static_cast<const float*>(nullptr), C, kernel, fusion);
}

void matmul_cuda(std::size_t m, std::size_t n, std::size_t k, const float* A,
                 const std::int8_t* B, const float* scale, float* C, GpuKernel kernel,
                 const MatmulFusion& fusion) {
    matmul_impl(m, n, k, A, B, scale, C, kernel, fusion);
}

Tensor matmul_cuda(const Tensor& A, const Tensor& B, GpuKernel kernel) {
    const DeviceTensor dA(A), dB(B);
    DeviceTensor dC;
    matmul_cuda(dA, dB, dC, kernel);
    return dC.download();
}

Tensor matmul_cuda(const Tensor& A, const QTensor& B, GpuKernel kernel) {
    if (A.cols != B.rows)
        throw std::invalid_argument("matmul_cuda: inner dimensions differ (A is " +
                                    std::to_string(A.rows) + "x" + std::to_string(A.cols) +
                                    ", B is " + std::to_string(B.rows) + "x" +
                                    std::to_string(B.cols) + ")");
    const DeviceTensor dA(A);
    const DeviceQTensor dB(B);
    DeviceTensor dC(A.rows, B.cols);
    matmul_cuda(A.rows, B.cols, A.cols, dA.data(), dB.data(), dB.scale(), dC.data(), kernel);
    return dC.download();
}

std::size_t cuda_scratch_generation() { return g_scratch_generation; }

void cuda_synchronize() { cuda_check(cudaDeviceSynchronize(), "synchronize"); }

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
