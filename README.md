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
- [x] Load model weights (stories15M)
- [x] Full transformer forward pass on CPU
- [x] Text generation with sampling
- [x] KV cache
- [ ] Batched prompt prefill
- [ ] CUDA kernels: matmul, softmax, RMSNorm, attention
- [ ] int8 quantization
- [ ] Benchmarks vs llama.cpp

## Running a model
Checkpoints use the llama2.c format. To try the 15M-parameter TinyStories model:

```
mkdir models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
cmake -S . -B build -G Ninja && cmake --build build
build/generate models/stories15M.bin -t 0.8 -p 0.9 -i "One day, Tom found a"
```

`-t 0` is greedy; with no prompt it tells the familiar "Once upon a time,
there was a little girl named Lily..." story. Text goes to stdout, load time
and decode speed to stderr.

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

Token generation with stories15M (dim 288, 6 layers, 32k vocab), greedy,
256 positions:

| Thread pool              | ms / token | tokens / s |
|--------------------------|------------|------------|
| Condvar wake per job     | 5.4-5.9    | 168-186    |
| Spin, then sleep         | 2.04-2.12  | 472-491    |

One token is 25 matmuls, most of them 6-40 us of work. A condition-variable
wake-up costs ~100 us on Windows, so the old pool spent most of each token
waking threads. Per token the model streams ~61 MB of weights (37 MB of it the
classifier), a floor of ~1.2 ms at the measured 51.7 GB/s. The forward pass
alone runs at ~1.9 ms.

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
   wake-up rather than a thread spawn, and there's no locking around C.
   Idle workers spin for a few hundred microseconds before sleeping, and
   joining a job takes no lock, so back-to-back calls start in about a
   microsecond. Rows
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

### How the forward pass works
`src/model.cpp` loads a llama2.c checkpoint and runs one token at a time:

- **Weights stored transposed.** The file holds projections as (out x in);
  they're stored (in x out) at load so `x * W` is a plain row-times-matrix
  product, and the m = 1 streaming path reads each weight row once, in order.
- **Fused projections.** q, k and v read the same input, as do the gate and
  up projections, so each pair or triple is concatenated by columns: 4
  matmuls per layer instead of 7, and fewer pool dispatches per token.
- **KV cache.** Each position's rotated keys and values are kept per layer,
  so a new token runs attention against the cache instead of recomputing the
  prefix. Grouped-query attention maps each query head to its kv head.
- **Tested against a reference.** `tests/test_model.cpp` writes small random
  checkpoints (multi-head, grouped-query, multi-query; tied and separate
  classifier) and checks every position's logits against a from-scratch,
  double-precision implementation that reads the file's own layout.

Known limits and next steps:
- Prompts go through the model one token at a time. Running the whole
  prompt as one m = n_prompt pass would turn it into proper GEMMs.
- Decode is memory-bound: fp32 weights put the floor at ~1.2 ms/token here.
  Smaller weight types (bf16, int8) cut it directly. Splitting the k = 288
  products into more than 5 split-K tasks was slower: the partial sums cost
  more than the extra bandwidth bought.
- A tied classifier is stored twice, as the embedding and transposed.
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
