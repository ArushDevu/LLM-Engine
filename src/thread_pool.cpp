#include "thread_pool.h"

#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define ENGINE_CPU_RELAX() _mm_pause()
#else
#define ENGINE_CPU_RELAX() std::this_thread::yield()
#endif

// Protocol
//
// generation_ is odd while run() writes a job and even once it is published.
// A worker joins a published generation g by incrementing busy_ and then
// checking generation_ is still g; all of these are seq_cst. run() makes
// generation_ odd before it waits for busy_ == 0, so either it sees the
// worker's increment and waits, or the worker sees the odd value and backs
// off. A worker that joined therefore reads job_ and tasks_ while nobody
// writes them, without a lock.
//
// The mutex and condition variables are only for sleeping. A thread
// announces it is about to sleep (sleepers_, caller_waiting_) before its last
// check of the condition, and the waking side checks the announcement after
// changing the condition, both seq_cst, so one of them always sees the other.

namespace {

// Polls before a thread falls back to blocking on a condition variable. A
// condvar wake-up costs ~100 us on Windows, far more than a decode-sized
// matmul (6-40 us), while jobs during generation arrive every few tens of
// microseconds. ~2^15 pauses is a few hundred microseconds on current x86:
// long enough to bridge consecutive jobs of one forward pass, short enough
// that an idle pool goes to sleep almost at once.
constexpr int kSpinIters = 1 << 15;

template <class Pred>
bool spin_until(Pred&& done) {
    for (int i = 0; i < kSpinIters; ++i) {
        if (done()) return true;
        ENGINE_CPU_RELAX();
    }
    return done();
}

} // namespace

ThreadPool::ThreadPool(unsigned threads) {
    const unsigned workers = std::max(1u, threads) - 1;
    workers_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    stop_.store(true);
    { std::lock_guard<std::mutex> lk(mu_); }
    wake_.notify_all();
    for (auto& t : workers_) t.join();
}

ThreadPool& ThreadPool::global() {
    static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

void ThreadPool::run(std::size_t tasks, const std::function<void(std::size_t)>& fn) {
    if (tasks == 0) return;
    std::lock_guard<std::mutex> serial(run_mu_);

    generation_.fetch_add(1);  // odd: writing
    // A worker that joined the previous job late may still be in drain()
    // holding that job's pointer; resetting next_ under it would hand it a
    // task of this job. It has nothing left to claim, so it leaves promptly.
    while (busy_.load() != 0) ENGINE_CPU_RELAX();
    job_ = &fn;
    tasks_ = tasks;
    remaining_.store(tasks, std::memory_order_relaxed);
    next_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1);  // even: published

    if (tasks > 1 && sleepers_.load() > 0) {
        { std::lock_guard<std::mutex> lk(mu_); }
        wake_.notify_all();
    }

    drain(fn, tasks);

    // Only wait for the tasks, not for every worker to wake: stragglers find
    // nothing to claim and are handled by the busy_ wait above.
    auto finished = [&] { return remaining_.load() == 0; };
    if (spin_until(finished)) return;
    std::unique_lock<std::mutex> lk(mu_);
    caller_waiting_.store(true);
    done_.wait(lk, finished);
    caller_waiting_.store(false);
}

void ThreadPool::worker_loop() {
    std::uint64_t seen = 0;
    std::uint64_t g = 0;
    auto ready = [&] {
        g = generation_.load();
        return stop_.load() || (g != seen && g % 2 == 0);
    };
    for (;;) {
        if (!spin_until(ready)) {
            std::unique_lock<std::mutex> lk(mu_);
            sleepers_.fetch_add(1);
            wake_.wait(lk, ready);
            sleepers_.fetch_sub(1);
        }
        if (stop_.load()) return;

        busy_.fetch_add(1);
        if (generation_.load() != g) {  // run() started writing a newer job
            busy_.fetch_sub(1);
            continue;
        }
        seen = g;
        drain(*job_, tasks_);
        busy_.fetch_sub(1);
    }
}

void ThreadPool::drain(const std::function<void(std::size_t)>& fn, std::size_t tasks) {
    for (;;) {
        const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
        if (i >= tasks) return;
        fn(i);
        // The caller's load of 0 then sees every task's writes.
        if (remaining_.fetch_sub(1) == 1 && caller_waiting_.load()) {
            { std::lock_guard<std::mutex> lk(mu_); }
            done_.notify_all();
        }
    }
}
