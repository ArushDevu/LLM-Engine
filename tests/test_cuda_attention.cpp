// Checks both GPU attention kernels against a double-precision reference:
// multi-head, grouped-query (every heads-per-block case) and multi-query,
// short and long contexts, one and several rows. Exits with 77 (reported as
// skipped by ctest) when there is no CUDA device.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_attention.h"
#include "cuda_matmul.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                         #cond);                                                 \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr float kTol = 5e-5f;

Tensor random_tensor(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t(rows, cols);
    for (auto& x : t.data) x = dist(rng);
    return t;
}

// out as attention_cuda() defines it, in double.
std::vector<double> reference(const AttentionShape& s, std::size_t rows, int pos, const Tensor& q,
                              const Tensor& kc, const Tensor& vc) {
    const std::size_t hs = std::size_t(s.head_size), group = std::size_t(s.n_heads / s.n_kv_heads);
    std::vector<double> out(rows * s.out_stride, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        const std::size_t n_ctx = std::size_t(pos) + i + 1;
        for (std::size_t h = 0; h < std::size_t(s.n_heads); ++h) {
            const std::size_t kv_off = (h / group) * hs;
            std::vector<double> sc(n_ctx);
            double mx = -INFINITY;
            for (std::size_t t = 0; t < n_ctx; ++t) {
                double dot = 0.0;
                for (std::size_t d = 0; d < hs; ++d)
                    dot += double(q.data[i * s.q_stride + h * hs + d]) *
                           double(kc.data[t * s.kv_stride + kv_off + d]);
                sc[t] = dot / std::sqrt(double(hs));
                mx = std::max(mx, sc[t]);
            }
            double sum = 0.0;
            for (double& x : sc) sum += x = std::exp(x - mx);
            for (std::size_t d = 0; d < hs; ++d) {
                double acc = 0.0;
                for (std::size_t t = 0; t < n_ctx; ++t)
                    acc += sc[t] * double(vc.data[t * s.kv_stride + kv_off + d]);
                out[i * s.out_stride + h * hs + d] = acc / sum;
            }
        }
    }
    return out;
}

struct Case {
    int heads, kv_heads, head_size, seq_len;
};

const char* name(AttentionKernel k) {
    switch (k) {
    case AttentionKernel::single_block: return "single_block";
    case AttentionKernel::split: return "split";
    default: return "automatic";
    }
}

// Every (pos, rows) of a case through kernel; the second identical call must
// give the same bits (the split kernel's counters reset, and its merge order
// is fixed).
void test_case(const Case& c, AttentionKernel kernel, std::initializer_list<int> positions,
               std::size_t rows, std::mt19937& rng) {
    const std::size_t hs = std::size_t(c.head_size), kv = std::size_t(c.kv_heads) * hs;
    const std::size_t dim = std::size_t(c.heads) * hs;
    // q rows strided like the model's qkv buffer.
    const AttentionShape s{c.heads, c.kv_heads, c.head_size, c.seq_len, dim + 2 * kv, kv, dim};
    try {
        attention_check(s, kernel);
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "  %s rejected %d/%d x %d, seq_len %d: %s\n", name(kernel), c.heads,
                     c.kv_heads, c.head_size, c.seq_len, e.what());
        ++failures;
        return;
    }
    const Tensor q = random_tensor(rows, s.q_stride, rng);
    const Tensor kc = random_tensor(std::size_t(c.seq_len), kv, rng);
    const Tensor vc = random_tensor(std::size_t(c.seq_len), kv, rng);
    const DeviceTensor dq(q), dk(kc), dv(vc);
    DeviceTensor out(rows, dim);
    const std::size_t scratch_n = attention_scratch_floats(s, rows, kernel);
    DeviceTensor scratch(1, std::max<std::size_t>(1, scratch_n));
    cudaMemset(scratch.data(), 0, scratch_n * sizeof(float));
    int* pos_dev = nullptr;
    cudaMalloc(&pos_dev, sizeof(int));

    for (int pos : positions) {
        if (std::size_t(pos) + rows > std::size_t(c.seq_len)) continue;
        cudaMemcpy(pos_dev, &pos, sizeof pos, cudaMemcpyHostToDevice);
        attention_cuda(s, rows, pos_dev, dq.data(), dk.data(), dv.data(), out.data(),
                       scratch.data(), kernel);
        const Tensor got = out.download();
        attention_cuda(s, rows, pos_dev, dq.data(), dk.data(), dv.data(), out.data(),
                       scratch.data(), kernel);
        const Tensor again = out.download();
        const std::vector<double> want = reference(s, rows, pos, q, kc, vc);
        double diff = 0.0;
        for (std::size_t j = 0; j < want.size(); ++j)
            diff = std::max(diff, std::fabs(double(got.data[j]) - want[j]));
        if (diff > kTol) {
            std::fprintf(stderr, "  %s %d/%d x %d, seq_len %d, pos %d, %zu rows: max diff %g\n",
                         name(kernel), c.heads, c.kv_heads, c.head_size, c.seq_len, pos, rows,
                         diff);
            ++failures;
        }
        CHECK(got.data == again.data);
    }
    cudaFree(pos_dev);
}

void test_kernels() {
    std::mt19937 rng(7);
    const AttentionKernel all[] = {AttentionKernel::single_block, AttentionKernel::split,
                                   AttentionKernel::automatic};
    // Positions cover one chunk, a chunk boundary, many chunks and the end.
    const Case cases[] = {
        {6, 6, 48, 256},    // stories15M: one head per split block
        {8, 2, 64, 1000},   // grouped-query, 4 heads per block
        {8, 1, 32, 600},    // multi-query, 8 heads per block
        {4, 2, 16, 300},    // 2 heads per block, tiny heads
        {6, 2, 32, 300},    // 3 heads per kv head: one per block
    };
    for (const Case& c : cases) {
        for (AttentionKernel k : all) {
            for (std::size_t rows : {1, 3, 8, 12})
                test_case(c, k, {0, 31, 32, 33, 200, c.seq_len - int(rows)}, rows, rng);
        }
    }
}

// Past single_block's shared-memory cap: split and automatic still work.
void test_long_context() {
    std::mt19937 rng(11);
    const Case c{4, 1, 128, 20000};
    const std::size_t kv = 128, dim = 4 * 128;
    const AttentionShape s{c.heads, c.kv_heads, c.head_size, c.seq_len, dim + 2 * kv, kv, dim};
    bool threw = false;
    try {
        attention_check(s, AttentionKernel::single_block);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    test_case(c, AttentionKernel::split, {0, 5000, 19999}, 1, rng);
    test_case(c, AttentionKernel::automatic, {12345, 19990}, 10, rng);
}

void test_bad_shapes() {
    auto rejects = [](const AttentionShape& s, AttentionKernel k) {
        try {
            attention_check(s, k);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    // head_size not a multiple of 4: split can't load float4s, single_block can.
    CHECK(rejects({2, 2, 6, 64, 36, 12, 12}, AttentionKernel::split));
    CHECK(!rejects({2, 2, 6, 64, 36, 12, 12}, AttentionKernel::single_block));
    CHECK(rejects({6, 4, 8, 64, 96, 32, 48}, AttentionKernel::automatic));  // 6 % 4 heads
    CHECK(rejects({4, 4, 8, 0, 64, 32, 32}, AttentionKernel::automatic));
}

} // namespace

int main() try {
    if (!cuda_available()) {
        std::puts("no CUDA device, skipping");
        return 77;
    }
    test_kernels();
    test_long_context();
    test_bad_shapes();
    cuda_synchronize();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
