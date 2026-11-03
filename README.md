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
- [x] Batched prompt prefill
- [x] int8 weight quantization
- [ ] CUDA kernels: matmul, softmax, RMSNorm, attention
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
there was a little girl named Lily..." story. `-w q8` stores the weights as
int8. Text goes to stdout, load time, prefill and decode speed to stderr.

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
A parallel streaming read on this machine peaks at 51.7 GB/s. With int8
weights B is 16.8 MB, a quarter of the bytes (same run for both columns):

| m (rows) | fp32 B  | GB/s | int8 B  | GB/s | Speed-up |
|----------|---------|------|---------|------|----------|
| 1        | 1.99 ms | 33.8 | 0.22 ms | 75.9 | 9.0x     |
| 4        | 2.13 ms | 31.5 | 0.87 ms | 19.4 | 2.4x     |
| 16       | 3.78 ms | 17.7 | 1.82 ms | 9.2  | 2.1x     |
| 64       | 6.30 ms | 10.7 | 5.34 ms | 3.1  | 1.2x     |

int8 at m = 1 beats the DRAM peak because most of B stays in the 16 MB L3
between runs; m >= 4 is bound by widening int8 to fp32 (one core does
~2 elements per cycle), not by bytes. In about one call in ten a straggling
worker adds 0.5-1 ms; medians are stable, p90s aren't.

Token generation with stories15M (dim 288, 6 layers, 32k vocab), greedy,
256 positions:

| Thread pool              | ms / token | tokens / s |
|--------------------------|------------|------------|
| Condvar wake per job     | 5.4-5.9    | 168-186    |
| Spin, then sleep         | 2.04-2.12  | 472-491    |

Weights and prompt handling, five interleaved runs of each build, medians
(this session ran ~30% slower than the table above, so compare rows here
with each other only). The prompt is 158 tokens:

| Build                            | Decode, ms / token | Prompt, 158 tok    |
|----------------------------------|--------------------|--------------------|
| fp32, prompt one token at a time | 2.75               | 433 ms             |
| fp32, batched prefill            | 2.74               | 39 ms (4.1k tok/s) |
| int8 (`-w q8`), batched prefill  | 1.12               | 35 ms              |

Prefill replaces 158 m = 1 passes, each streaming all 61 MB of weights,
with one pass of m = 158 GEMMs, and skips the classifier (37 of those 61 MB)
for every row but the last. Greedy output is byte-identical to the
token-at-a-time path.

int8 accuracy, teacher-forced against fp32 on the same tokens:

| Text                    | Tokens | Perplexity fp32 / int8 | Mean KL (nats) | Top-1 agree |
|-------------------------|--------|------------------------|----------------|-------------|
| Hand-written story      | 157    | 7.388 / 7.400          | 9.3e-4         | 99.4%       |
| fp32 greedy output      | 221    | 1.457 / 1.458          | 6.4e-4         | 99.5%       |
| fp32 sampled, t = 1     | 169    | 2.249 / 2.244          | 6.7e-4         | 98.8%       |

Greedy text from the two matches for the first ~200 tokens, then they take
different, equally plausible endings.

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
   microsecond. Rows are split first; when m is too small to occupy every
   thread (m = 1 per generated token), the rest go to column slabs, which
   pack only their own slice of B.
5. **Streaming path for m <= 4.** With so few rows, packing B only adds
   memory traffic. Each task instead takes a contiguous range of B's rows,
   reads them strictly in order into a private partial C that stays in
   cache, and the partials are summed at the end (split-K). This is the only
   split that reads a row-major B sequentially; splitting by columns gave
   ~20 GB/s because each thread jumps 16 KB between short reads.
6. **No allocation in steady state.** Packing buffers are thread-local,
   64-byte aligned and grow-only. Packing overwrites everything the kernel
   reads, so they are never zeroed.

### How the forward pass works
`src/model.cpp` loads a llama2.c checkpoint and runs the prompt in one batch,
then one token at a time:

- **Weights stored transposed.** The file holds projections as (out x in);
  they're stored (in x out) at load so `x * W` is a plain row-times-matrix
  product, and the m = 1 streaming path reads each weight row once, in order.
- **Fused projections.** q, k and v read the same input, as do the gate and
  up projections, so each pair or triple is concatenated by columns: 4
  matmuls per layer instead of 7, and fewer pool dispatches per token.
- **KV cache.** Each position's rotated keys and values are kept per layer,
  so a new token runs attention against the cache instead of recomputing the
  prefix. Grouped-query attention maps each query head to its kv head.
- **Batched prefill.** `prefill()` runs n tokens as n-row activations
  through the same code as `forward()`, which is just n = 1: projections
  become m = n GEMMs, each row's keys and values go into the cache before
  attention, and row i attends to positions up to pos + i. Attention is one
  pool task per (row, head).
- **int8 weights.** `WeightType::q8` quantizes each projection and the
  classifier at load, symmetric with one scale per output channel (a column
  of the stored (in x out) matrix), then frees the fp32 copy. The
  streaming path widens int8 to fp32 in its inner loop and applies the
  scales once, after the split-K sum; the packed path dequantizes while it
  packs B, so the micro-kernel is unchanged. Quantizing adds ~100 ms to
  load.
- **Tested against a reference.** `tests/test_model.cpp` writes small random
  checkpoints (multi-head, grouped-query, multi-query; tied and separate
  classifier) and checks every position's logits against a from-scratch,
  double-precision implementation that reads the file's own layout, for
  token-by-token decode, chunked prefill and int8 (against the reference
  run on the int8-rounded weights).

Known limits and next steps:
- fp32 decode is memory-bound: the weights put the floor at ~1.2 ms/token
  here. Splitting the k = 288 products into more than 5 split-K tasks was
  slower: the partial sums cost more than the extra bandwidth bought.
- int8 decode is now bound by compute in the int8-to-fp32 widening, not by
  bytes. GCC widens in two steps (8->16->32 bit) with mixed vector widths;
  an AVX-512 kernel that widens once, or int8 dot products against
  quantized activations (VNNI), is the next step. Doing four rows of B per
  pass over the accumulators sped fp32 up single-threaded (3.5 -> 2.2 ms)
  but changed neither path with 16 threads, so it isn't in.
- Prefill reaches ~50 GFLOPS: m = 158 against k = 288 is too thin to fill
  16 threads with MC = 64 row blocks. Very long prompts would also want
  their activations chunked.
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
