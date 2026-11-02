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

Decode-shaped products, one to a few token rows times a 4096 x 4096 weight
matrix, are bound by streaming the weights from memory rather than by FLOPs.
A parallel streaming read on this machine peaks at 51.7 GB/s.

| m (rows) | Time    | GFLOPS | Weight bandwidth |
|----------|---------|--------|------------------|
| 1        | 1.48 ms | 22.7   | 45.5 GB/s        |
| 4        | 1.87 ms | 71.9   | 35.9 GB/s        |
| 16       | 3.08 ms | 174.3  | 21.8 GB/s        |
| 64       | 6.06 ms | 354.2  | 11.1 GB/s        |

Laptop numbers drift by 10-20% with temperature and power state, so compare
changes by running old and new builds back to back, not against these tables.

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
4. **Threads.** `matmul_parallel` cuts C into a grid of disjoint tiles and
   runs them on a persistent pool (`src/thread_pool.h`), so a call costs a
   wake-up rather than a thread spawn, and there's no locking around C. Rows
   are split first; when m is too small to occupy every thread (m = 1 per
   generated token), the rest go to column slabs, which pack only their own
   slice of B.
6. **Streaming path for m <= 4.** With so few rows, packing B only adds
   memory traffic. Each task instead takes a contiguous range of B's rows,
   reads them strictly in order into a private partial C that stays in
   cache, and the partials are summed at the end (split-K). This is the only
   split that reads a row-major B sequentially; splitting by columns gave
   ~20 GB/s because each thread jumps 16 KB between short reads.
5. **No allocation in steady state.** Packing buffers are thread-local,
   64-byte aligned and grow-only. Packing overwrites everything the kernel
   reads, so they are never zeroed.

Known limits and next steps:
- Each row slab packs all of B; sharing one packed copy would cut memory
  traffic for large m.
- Between m = 4 and m = 16 neither path reaches DRAM bandwidth. Storing
  weights pre-packed (or transposed) at load time would let both paths read
  B sequentially without per-call packing.
- When B fits in L3 the packed path is ~20% faster than streaming at m = 4;
  the cutoff is tuned for weights in DRAM.
- The micro-kernel relies on auto-vectorization. A sweep of tile shapes
  (4x16 up to 14x32, KC 128/256) moved single-thread speed by under 10%,
  so the tile isn't the bottleneck; profile before hand-writing intrinsics.

## Build
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build        # correctness tests vs. the naive reference
./build/bench_matmul          # or: ./build/bench_matmul 256 1024
```

Pass `-DENGINE_NATIVE=OFF` to build a binary that runs on other machines.
