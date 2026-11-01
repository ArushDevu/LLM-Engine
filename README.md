# LLM-Engine

A large language model inference engine built from scratch in C++ and CUDA.
No PyTorch, no ML libraries. Every tensor op, every kernel, and every layer
of the transformer is written by hand.

## Goals
- Run a Llama-style model end to end and generate text
- Write custom CUDA kernels for the hot paths (matmul, attention, softmax)
- Measure and document every optimization with benchmarks

## Roadmap
- [x] Tensor type + naive matmul with benchmark
- [x] Multithreaded / cache-optimized CPU matmul
- [ ] Load model weights (stories15M)
- [ ] Full transformer forward pass on CPU
- [ ] Text generation with sampling
- [ ] CUDA kernels: matmul, softmax, RMSNorm, attention
- [ ] KV cache
- [ ] int8 quantization
- [ ] Benchmarks vs llama.cpp

## Benchmarks
Square fp32 matmul, median of several runs after a warm-up
(`bench_matmul`). AMD Ryzen 7 7840HS (8C/16T, AVX-512), GCC 16, `-O3 -march=native`.

| N    | Naive (i-k-j) | Blocked, 1 thread | Parallel, 16 threads |
|------|---------------|-------------------|----------------------|
| 512  | 26.3 GFLOPS   | 62.3 GFLOPS       | 217.0 GFLOPS         |
| 1024 | 22.8 GFLOPS   | 55.8 GFLOPS       | 225.7 GFLOPS         |
| 2048 | -             | 50.9 GFLOPS       | 309.1 GFLOPS         |

Laptop numbers drift by 10-20% with temperature and power state, so compare
changes by running old and new builds back to back, not against this table.

### How the CPU matmul works
`src/matmul.cpp` follows the Goto/BLIS structure:

1. **Blocking.** Loop over blocks of N (`NC`), K (`KC`) and M (`MC`) sized so
   each working set fits a cache level: a `KC x NR` panel of B in L1, an
   `MC x KC` block of A in L2.
2. **Packing.** Copy each block of A and B into contiguous, zero-padded panels
   so the inner loop reads both with unit stride.
3. **Micro-kernel.** A 4x16 register tile accumulates over `KC` steps. Panels
   are padded to full tiles, so the hot loop has fixed trip counts and no edge
   branches; the compiler vectorizes it to FMA. Edges are handled only on the
   final store.
4. **Threads.** `matmul_parallel` splits C into disjoint slabs of rows and runs
   them on a persistent pool (`src/thread_pool.h`), so a call costs a wake-up
   rather than a thread spawn, and there's no locking around C.
5. **No allocation in steady state.** Packing buffers are thread-local,
   64-byte aligned and grow-only. Packing overwrites everything the kernel
   reads, so they are never zeroed.

Known limits and next steps:
- Every thread packs all of B; sharing one packed copy would cut memory traffic.
- The micro-kernel relies on auto-vectorization. Hand-written AVX-512
  intrinsics and a tuned tile shape are the next step toward peak.

## Build
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build        # correctness tests vs. the naive reference
./build/bench_matmul          # or: ./build/bench_matmul 256 1024
```

Pass `-DENGINE_NATIVE=OFF` to build a binary that runs on other machines.
