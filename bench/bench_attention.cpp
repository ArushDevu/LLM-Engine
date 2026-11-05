// GPU decode attention (one query row against a KV cache), single_block vs
// split, at several context lengths.
//
//   bench_attention
//
// Each cell is the median over runs of a CUDA graph of kCalls back-to-back
// calls, divided by kCalls: the per-call cost inside a captured decode step,
// without the ~15 us host launch cost a lone launch pays on Windows. GB/s
// counts the keys and values read.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_attention.h"
#include "cuda_matmul.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kCalls = 50;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

Tensor random_tensor(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t(rows, cols);
    for (auto& x : t.data) x = dist(rng);
    return t;
}

// Per-call seconds of kernel inside a graph; negative if the kernel rejects
// the shape.
double time_call(const AttentionShape& s, AttentionKernel kernel, const int* pos,
                 const DeviceTensor& q, const DeviceTensor& kc, const DeviceTensor& vc,
                 DeviceTensor& out, DeviceTensor& scratch) {
    try {
        attention_check(s, kernel);
    } catch (const std::invalid_argument&) {
        return -1.0;
    }
    cudaStream_t stream = cudaStreamPerThread;
    cudaGraph_t graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    for (int c = 0; c < kCalls; ++c)
        attention_cuda(s, 1, pos, q.data(), kc.data(), vc.data(), out.data(), scratch.data(),
                       kernel);
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    cudaGraphExec_t exec;
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cudaGraphDestroy(graph);
    std::vector<double> times;
    for (int r = 0; r < 16; ++r) {
        const auto start = Clock::now();
        check(cudaGraphLaunch(exec, stream), "launch");
        check(cudaStreamSynchronize(stream), "sync");
        if (r > 0) times.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    cudaGraphExecDestroy(exec);
    std::nth_element(times.begin(), times.begin() + long(times.size() / 2), times.end());
    return times[times.size() / 2] / kCalls;
}

struct Model {
    const char* name;
    int heads, kv_heads, head_size;
};

} // namespace

int main() try {
    if (!cuda_available()) {
        std::puts("no CUDA device");
        return 1;
    }
    std::mt19937 rng(0);
    int* pos = nullptr;
    check(cudaMalloc(&pos, sizeof(int)), "alloc pos");

    std::printf("decode attention, 1 row, per call in a graph; cells are us / GB/s of K and V\n\n");
    std::printf("%-26s %7s | %17s | %17s | %7s\n", "model", "n_ctx", "single_block", "split",
                "speed-up");
    std::printf("-----------------------------------+-------------------+-------------------+"
                "---------\n");
    const Model models[] = {
        {"stories15M (6 x 48)", 6, 6, 48},
        {"7B MHA (32 x 128)", 32, 32, 128},
        {"8B GQA (32 x 128, 8 kv)", 32, 8, 128},
    };
    for (const Model& m : models) {
        for (int n_ctx : {32, 256, 1024, 4096, 16384}) {
            if (m.heads == 6 && n_ctx > 256) continue;  // stories15M's seq_len is 256
            const std::size_t kv = std::size_t(m.kv_heads) * std::size_t(m.head_size);
            const std::size_t dim = std::size_t(m.heads) * std::size_t(m.head_size);
            const AttentionShape s{m.heads, m.kv_heads, m.head_size, n_ctx, dim, kv, dim};
            const DeviceTensor q(random_tensor(1, dim, rng));
            const DeviceTensor kc(random_tensor(std::size_t(n_ctx), kv, rng));
            const DeviceTensor vc(random_tensor(std::size_t(n_ctx), kv, rng));
            DeviceTensor out(1, dim);
            DeviceTensor scratch(1, std::max<std::size_t>(
                                        1, attention_scratch_floats(s, 1, AttentionKernel::split)));
            check(cudaMemset(scratch.data(), 0, scratch.cols() * sizeof(float)), "zero scratch");
            const int p = n_ctx - 1;
            check(cudaMemcpy(pos, &p, sizeof p, cudaMemcpyHostToDevice), "upload pos");
            const double bytes = 2.0 * double(n_ctx) * double(kv) * sizeof(float);
            const double single =
                time_call(s, AttentionKernel::single_block, pos, q, kc, vc, out, scratch);
            const double split = time_call(s, AttentionKernel::split, pos, q, kc, vc, out, scratch);
            std::printf("%-26s %7d | ", m.name, n_ctx);
            if (single > 0)
                std::printf("%8.1f %8.1f", single * 1e6, bytes / single / 1e9);
            else
                std::printf("%17s", "n/a");
            std::printf(" | %8.1f %8.1f | ", split * 1e6, bytes / split / 1e9);
            if (single > 0)
                std::printf("%6.2fx\n", single / split);
            else
                std::printf("%7s\n", "-");
        }
    }
    cudaFree(pos);
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
