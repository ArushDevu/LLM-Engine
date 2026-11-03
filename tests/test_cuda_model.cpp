// Checks the GPU forward pass against the CPU one, which test_model checks
// against a from-scratch reference. Exits with 77 (reported as skipped by
// ctest) when there is no CUDA device.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "checkpoint_fixture.h"
#include "cuda_model.h"
#include "model.h"

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

constexpr float kTol = 1e-4f;

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

Checkpoint load_random(int heads, int kv_heads, bool shared) {
    const RawModel m = make_model(heads, kv_heads, shared, std::uint32_t(heads * 10 + kv_heads));
    const std::string path = temp_path("engine_test_cuda_model.bin");
    write_checkpoint(m, path);
    Checkpoint ckpt = load_checkpoint(path);
    std::filesystem::remove(path);
    return ckpt;
}

bool check_close(const std::vector<float>& gpu, const std::vector<float>& cpu, const char* what,
                 std::size_t pos) {
    const float diff = max_abs_diff(gpu, cpu);
    if (diff <= kTol) return true;
    std::fprintf(stderr, "  %s at pos %zu: max diff %g\n", what, pos, double(diff));
    return false;
}

// Decode over the whole context, then prefill in chunks of 1, 1 and 5 rows
// (the skinny matmul and the split-tile path), a decode step on that cache,
// and a whole-context prefill that rewinds over it.
void test_matches_cpu(int heads, int kv_heads, bool shared) {
    const Checkpoint ckpt = load_random(heads, kv_heads, shared);
    Transformer cpu(ckpt);
    CudaTransformer gpu(ckpt);
    const std::size_t T = ckpt.config.seq_len;
    const auto V = int(ckpt.config.vocab_size);

    std::vector<int> tokens;
    for (std::size_t pos = 0; pos < T; ++pos) {
        tokens.push_back(int(pos * 7 + 3) % V);
        CHECK(check_close(gpu.forward(tokens[pos], pos), cpu.forward(tokens[pos], pos),
                          "forward", pos));
    }

    std::size_t pos = 0;
    for (std::size_t chunk : {1, 1, 5}) {
        const std::vector<int> part(tokens.begin() + long(pos), tokens.begin() + long(pos + chunk));
        CHECK(check_close(gpu.prefill(part, pos), cpu.prefill(part, pos), "prefill", pos));
        pos += chunk;
    }
    CHECK(check_close(gpu.forward(tokens[pos], pos), cpu.forward(tokens[pos], pos),
                      "forward after prefill", pos));
    CHECK(check_close(gpu.prefill(tokens, 0), cpu.prefill(tokens, 0), "full prefill", 0));
}

template <class E, class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const E&) {
        return true;
    }
    return false;
}

// Same rejections as the CPU model, and a rejected call leaves the cache
// usable.
void test_bad_inputs() {
    const Checkpoint ckpt = load_random(4, 2, true);
    CudaTransformer gpu(ckpt);
    Transformer cpu(ckpt);
    const auto V = int(ckpt.config.vocab_size);
    const std::size_t T = ckpt.config.seq_len;
    CHECK(throws<std::out_of_range>([&] { gpu.forward(V, 0); }));
    CHECK(throws<std::out_of_range>([&] { gpu.forward(-1, 0); }));
    CHECK(throws<std::out_of_range>([&] { gpu.forward(0, 1); }));  // skips pos 0
    CHECK(throws<std::invalid_argument>([&] { gpu.prefill({}, 0); }));
    CHECK(throws<std::out_of_range>([&] { gpu.prefill(std::vector<int>(T + 1, 0), 0); }));
    CHECK(check_close(gpu.prefill({1, 2, 3}, 0), cpu.prefill({1, 2, 3}, 0), "after errors", 0));
}

} // namespace

int main() try {
    if (!cuda_available()) {
        std::puts("no CUDA device, skipping");
        return 77;
    }
    test_matches_cpu(4, 4, true);   // multi-head, tied classifier
    test_matches_cpu(4, 2, false);  // grouped-query, separate classifier
    test_matches_cpu(4, 1, true);   // multi-query
    test_bad_inputs();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
