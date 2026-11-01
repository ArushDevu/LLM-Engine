#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "thread_pool.h"

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

// Every index runs exactly once, for task counts below, at and above the
// thread count.
void test_each_task_runs_once() {
    ThreadPool pool(4);
    for (std::size_t tasks : {1u, 3u, 4u, 5u, 100u}) {
        std::vector<std::atomic<int>> hits(tasks);
        pool.run(tasks, [&](std::size_t i) { hits[i].fetch_add(1); });
        for (auto& h : hits) CHECK(h.load() == 1);
    }
}

// Back-to-back jobs are where a straggling worker from the previous job could
// claim a task of the next one.
void test_many_back_to_back_runs() {
    ThreadPool pool(8);
    for (int round = 0; round < 2000; ++round) {
        std::atomic<std::size_t> sum{0};
        pool.run(8, [&](std::size_t i) { sum.fetch_add(i + 1); });
        CHECK(sum.load() == 36);
    }
}

void test_single_thread_pool() {
    ThreadPool pool(1);
    CHECK(pool.size() == 1);
    int count = 0;  // no workers, so no data race
    pool.run(10, [&](std::size_t) { ++count; });
    CHECK(count == 10);
}

// Callers on different threads share one pool; run() serializes them.
void test_concurrent_callers() {
    ThreadPool pool(4);
    std::atomic<int> total{0};
    std::vector<std::thread> callers;
    for (int c = 0; c < 4; ++c) {
        callers.emplace_back([&] {
            for (int r = 0; r < 200; ++r) pool.run(6, [&](std::size_t) { total.fetch_add(1); });
        });
    }
    for (auto& t : callers) t.join();
    CHECK(total.load() == 4 * 200 * 6);
}

} // namespace

int main() {
    test_each_task_runs_once();
    test_many_back_to_back_runs();
    test_single_thread_pool();
    test_concurrent_callers();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
}
