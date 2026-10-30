// Matmul throughput across sizes and kernels.
//
//   bench_matmul [size ...]      default sizes: 128 256 512 1024 2048
//
// Each measurement is the median of several timed runs after one warm-up
// run, so a single preempted or cold-cache run doesn't move the number.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "matmul.h"

namespace {

using Clock = std::chrono::steady_clock;

// Naive is too slow to be worth timing past this.
constexpr std::size_t kNaiveMaxSize = 1024;

// Consumed after each run so the compiler can't drop the work.
volatile float g_sink;

Tensor random_tensor(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t(n, n);
    for (auto& x : t.data) x = dist(rng);
    return t;
}

// Enough reps for a stable median on small sizes without making 2048 slow.
int reps_for(std::size_t n) {
    if (n <= 256) return 21;
    if (n <= 1024) return 7;
    return 3;
}

template <class F>
double median_seconds(F&& f, int reps) {
    g_sink = f().data[0];  // warm-up: page in buffers, spin up clocks
    std::vector<double> times;
    times.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        const auto start = Clock::now();
        Tensor C = f();
        const auto end = Clock::now();
        g_sink = C.data[0];
        times.push_back(std::chrono::duration<double>(end - start).count());
    }
    std::nth_element(times.begin(), times.begin() + reps / 2, times.end());
    return times[reps / 2];
}

void print_cell(double secs, std::size_t n) {
    const double gflops = 2.0 * n * n * n / secs / 1e9;
    std::printf(" | %9.2f %8.1f", secs * 1e3, gflops);
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::size_t> sizes = {128, 256, 512, 1024, 2048};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::strtoul(argv[i], nullptr, 10));
    }

    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::printf("square fp32 matmul, %u hardware threads; cells are ms / GFLOPS (median)\n\n",
                threads);
    std::printf("%6s | %18s | %18s | %18s\n", "N", "naive", "blocked", "parallel");
    std::printf("-------+--------------------+--------------------+-------------------\n");

    std::mt19937 rng(0);
    for (std::size_t n : sizes) {
        const Tensor A = random_tensor(n, rng);
        const Tensor B = random_tensor(n, rng);
        const int reps = reps_for(n);

        std::printf("%6zu", n);
        if (n <= kNaiveMaxSize) {
            print_cell(median_seconds([&] { return matmul_naive(A, B); }, reps), n);
        } else {
            std::printf(" | %18s", "-");
        }
        print_cell(median_seconds([&] { return matmul_blocked(A, B); }, reps), n);
        print_cell(median_seconds([&] { return matmul_parallel(A, B); }, reps), n);
        std::printf("\n");
        std::fflush(stdout);
    }
}
