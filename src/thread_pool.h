#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Fork-join pool with persistent workers. Threads are created once, so a
// parallel call costs a wake-up instead of a thread spawn per worker; workers
// also keep their thread-local scratch (e.g. GEMM packing buffers) warm.
class ThreadPool {
public:
    // threads counts the calling thread, so threads - 1 workers are created.
    explicit ThreadPool(unsigned threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Shared pool sized to std::thread::hardware_concurrency().
    static ThreadPool& global();

    // Runs fn(i) for every i in [0, tasks) and returns once all have finished.
    // The caller works on tasks too instead of idling. Calls from different
    // threads are serialized; fn must not call run() on the same pool.
    void run(std::size_t tasks, const std::function<void(std::size_t)>& fn);

    // Worker threads plus the caller.
    unsigned size() const { return static_cast<unsigned>(workers_.size()) + 1; }

private:
    void worker_loop();
    // Claims and runs tasks of the current job until none are left.
    void drain(const std::function<void(std::size_t)>& fn, std::size_t tasks);

    std::vector<std::thread> workers_;
    std::mutex run_mu_;  // one job at a time

    std::mutex mu_;
    std::condition_variable wake_;  // workers: a new job was posted
    std::condition_variable done_;  // caller: tasks finished / workers idle
    const std::function<void(std::size_t)>* job_ = nullptr;
    std::size_t tasks_ = 0;
    std::size_t remaining_ = 0;    // tasks not yet finished
    std::size_t busy_ = 0;         // workers inside drain()
    std::uint64_t generation_ = 0; // bumped per job so workers spot new work
    bool stop_ = false;

    std::atomic<std::size_t> next_{0};  // next task index to claim
};
