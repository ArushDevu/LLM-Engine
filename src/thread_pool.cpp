#include "thread_pool.h"

#include <algorithm>

ThreadPool::ThreadPool(unsigned threads) {
    const unsigned workers = std::max(1u, threads) - 1;
    workers_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
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
    {
        std::unique_lock<std::mutex> lk(mu_);
        // A worker that woke late for the previous job may still be in
        // drain() holding that job's pointer; resetting next_ under it would
        // hand it a task of this job. Wait for it to leave first.
        done_.wait(lk, [&] { return busy_ == 0; });
        job_ = &fn;
        tasks_ = tasks;
        remaining_ = tasks;
        next_.store(0, std::memory_order_relaxed);
        ++generation_;
    }
    if (tasks > 1) wake_.notify_all();

    drain(fn, tasks);

    // Only wait for the tasks, not for every worker to wake: stragglers find
    // nothing to claim and are handled by the busy_ wait above.
    std::unique_lock<std::mutex> lk(mu_);
    done_.wait(lk, [&] { return remaining_ == 0; });
}

void ThreadPool::worker_loop() {
    std::uint64_t seen = 0;
    for (;;) {
        const std::function<void(std::size_t)>* fn;
        std::size_t tasks;
        {
            std::unique_lock<std::mutex> lk(mu_);
            wake_.wait(lk, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            fn = job_;
            tasks = tasks_;
            ++busy_;
        }
        drain(*fn, tasks);
        {
            std::lock_guard<std::mutex> lk(mu_);
            --busy_;
        }
        done_.notify_all();
    }
}

void ThreadPool::drain(const std::function<void(std::size_t)>& fn, std::size_t tasks) {
    for (;;) {
        const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
        if (i >= tasks) return;
        fn(i);
        bool last;
        {
            std::lock_guard<std::mutex> lk(mu_);
            last = --remaining_ == 0;
        }
        if (last) done_.notify_all();
    }
}
