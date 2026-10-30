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
| 512  | 26.6 GFLOPS   | 61.3 GFLOPS       | 77.4 GFLOPS          |
| 1024 | 25.3 GFLOPS   | 63.9 GFLOPS       | 175.2 GFLOPS         |
| 2048 | -             | 60.4 GFLOPS       | 253.6 GFLOPS         |

Expect a few percent noise between runs on a laptop.

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
4. **Threads.** `matmul_parallel` gives each thread a disjoint slab of rows of
   C with its own packing buffers, so there's no locking.

Known limits and next steps:
- Threads are spawned per call, so the parallel version loses to
  single-threaded below ~256. A persistent thread pool fixes this.
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
