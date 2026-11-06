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
- [x] CUDA matmul kernels
- [x] CUDA forward pass: weights, KV cache and activations on the GPU
- [x] int8 weights on the GPU
- [x] Faster GPU attention (split-KV)
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
int8; `-d gpu` runs on the GPU (a CUDA build), with either weight type. Text goes to
stdout, load time, prefill and decode speed to stderr.

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

### GPU matmul
RTX 4050 Laptop (20 SMs, sm_89, 24 MB L2), CUDA 13.4, MSVC host compiler
(`bench_cuda`). Times run from launch to `cudaDeviceSynchronize()`, so they
include the ~5-10 us launch cost. A device-to-device copy moves 168 GB/s
(read + write), the bandwidth ceiling; cuBLAS SGEMM is the compute
yardstick and isn't used by the engine.

| N    | Naive     | Tiled          | cuBLAS    | CPU, 16 threads |
|------|-----------|----------------|-----------|-----------------|
| 1024 | 0.80 TFLOPS | 3.5 TFLOPS (53%) | 6.6 TFLOPS | 0.23 TFLOPS   |
| 2048 | 0.80 TFLOPS | 4.3 TFLOPS (56%) | 7.7 TFLOPS | 0.31 TFLOPS   |
| 4096 | 0.74 TFLOPS | 4.0 TFLOPS (60%) | 6.7 TFLOPS | -             |

Decode shapes, m x 4096 times a 4096 x 4096 B already on the GPU:

| m  | Engine           | cuBLAS   | Engine + copy A in, C out | CPU fp32 |
|----|------------------|----------|---------------------------|----------|
| 1  | 0.38 ms, 176 GB/s | 0.40 ms | 0.47 ms                   | 1.99 ms  |
| 4  | 0.38 ms, 175 GB/s | 0.38 ms | 0.46 ms                   | 2.13 ms  |
| 8  | 0.40 ms, 168 GB/s | 0.40 ms | 0.55 ms                   | -        |
| 16 | 0.55 ms          | 0.40 ms  | 0.73 ms                   | 3.78 ms  |
| 32 | 0.57 ms          | 0.41 ms  | 0.80 ms                   | -        |
| 64 | 0.72 ms          | 0.42 ms  | 1.19 ms                   | 6.30 ms  |

Up to m = 8 the engine reads B at the copy ceiling and matches cuBLAS.
Each call's PCIe round trip for A and C costs ~0.1 ms, a quarter of the
product, so a GPU forward pass must keep activations on the device between
layers, not just weights.

How the kernels work (`src/cuda_matmul.cu`):

- **Tiled (m > 64).** A 256-thread block computes a 128 x 128 tile of C,
  staging 128 x 8 of A (transposed, padded against bank conflicts) and
  8 x 128 of B in shared memory per step; each thread accumulates 8 x 8
  outputs in registers. A thread's rows and columns are strided across the
  tile, so a warp reads consecutive shared-memory words and writes
  consecutive columns of C.
- **Split tiles (9 <= m <= 64).** The same kernel with 32- or 64-row tiles
  and k split across `blockIdx.z` until there are ~8 blocks per SM. With
  128-row tiles, m = 64 ran at 1.29 ms on 32 blocks; split, 0.72 ms.
- **Skinny (m <= 8).** One thread per column of C streams its column of a k
  slice, so each warp reads 128 contiguous bytes of a row of B per step; k is
  split so ~64 warps per SM keep loads in flight. Partial sums go to a
  scratch buffer and a second kernel adds them in a fixed order, so results
  are deterministic (no atomics).

Known limits: tiled is at 53-60% of cuBLAS (next: float4 loads,
double-buffered shared memory, warp tiles); m = 16-64 trails cuBLAS by
~1.4-1.7x; indexing is 32-bit, so each matrix must stay below 2^31 elements.

### GPU forward pass
stories15M, five interleaved runs of each, medians; CPU numbers from the
GCC build. Greedy and seeded sampled text are identical on GPU and CPU.

| Device             | Decode, ms / token | Prefill, 158 tok | Load   |
|--------------------|--------------------|------------------|--------|
| CPU, fp32          | 2.02               | 24 ms            | 81 ms  |
| CPU, int8 (`-w q8`) | 0.99              | 21 ms            | 192 ms |
| GPU, fp32 (`-d gpu`) | 0.61             | 5.9 ms           | 270 ms |

`CudaTransformer` (`src/cuda_model.cu`) keeps the weights, KV cache and
activations in GPU memory; a call uploads its position and token ids and
downloads only the logits. Each layer runs the qkv matmul with RMSNorm
fused in, RoPE plus the cache write, attention (split over positions for
decode; see GPU attention below), the output projection with the residual add fused
in, gate/up with RMSNorm fused in, SwiGLU, and the down projection with the
second residual add.

Unfused that was ~100 kernels per token, and a launch costs 14-19 us of host time on
this Windows driver, so launched one by one a token took 2.15 ms (one run),
almost all of it launch overhead. The first decode step is instead captured
into a CUDA graph and replayed with one launch per token: 0.64 ms. Kernels
read the position and token from device memory, so one graph serves every
position; it is captured again only if a buffer it points at moves
(activation buffers are grow-only, and the matmul's split-K scratch reports
a generation).

Then fusion, measured per kernel with Nsight Systems (graph nodes traced
individually) and kept only where the product's extra cost was below the
kernel it removed:

| Fused into the matmul          | Matmul cost | Kernel removed | Kept |
|--------------------------------|-------------|----------------|------|
| RMSNorm (qkv, gate/up)         | +0.6 us     | 1.7 us         | yes  |
| RMSNorm (classifier)           | +0 to 2 us  | 1.7 us         | yes  |
| Residual add (out, down)       | ~0          | 0.8 us         | yes  |
| SwiGLU (down projection)       | +5.6 us     | 1.1 us         | no   |

The skinny kernel normalizes A as it reads it, each block first computing
the row's scale; the add becomes `C += A x B` (`MatmulFusion`). Kernels per
token went from 92.6 to 67.7, GPU time from 504 to 478 us, and decode from
0.640 to 0.607 ms (medians of seven interleaved runs; text unchanged).
Splitting attention's weighted sum across all 256 threads was also tried
and was slower (11.9 vs 9.8 us per call), so attention is unchanged.

Where the 0.61 ms goes now: 213 us is the classifier and 160 us the layer
matmuls, all reading weights at ~170 GB/s, so ~0.37 ms is the fp32 floor;
attention is 58 us (6 blocks on 20 SMs, latency-bound); the rest is the
small kernels and the gaps between ~68 graph nodes. (Attention has since
been split over positions; see below.)

### GPU attention
Decode attention was one block per head: 6 blocks on 20 SMs for stories15M,
each thread reading its own key row, with the whole context's scores in
shared memory (a ~12k seq_len cap). `src/cuda_attention.cu` now splits
decode (up to 8 rows) flash-decoding style:

- **Split over positions.** Block (head group, row, s) takes the s-th chunk
  of the context (at least 32 positions, at most 128 chunks) and keeps only
  that chunk's scores, so seq_len is no longer capped by shared memory. The
  grid is sized for seq_len and chunks past the context exit at once, so
  one captured graph still serves every position.
- **Merge in the last block.** Each chunk writes its max, exp-sum and
  unnormalized value sum; the block that finishes last (an atomic counter,
  which it resets) rescales and adds them in chunk order. Results are
  deterministic and no second kernel is launched. A context that fits one
  chunk is written directly.
- **Coalesced loads.** Keys: a group of head_size / 4 lanes (rounded to a
  power of two) shares a position, loading float4s, and sums with shuffles.
  Values: consecutive threads load consecutive float4s of a row.
- **One block per kv head.** Grouped-query heads sharing a kv head (up to 8)
  run in one block, so each key and value is read once, not once per head.

Prefill (more rows) keeps the old kernel: rows x heads blocks already fill
the GPU. Per call inside a CUDA graph, one query row, n_ctx = seq_len
(`bench_attention`); GB/s counts keys and values read once:

| Shape                     | n_ctx | Old            | Split           | Speed-up |
|---------------------------|-------|----------------|-----------------|----------|
| stories15M (6 x 48)       | 32    | 3.7 us         | 4.0 us          | 0.92x    |
| stories15M (6 x 48)       | 256   | 12.6 us        | 6.5 us          | 1.94x    |
| 7B MHA (32 x 128)         | 1024  | 315 us, 106 GB/s | 199 us, 169 GB/s | 1.58x |
| 7B MHA (32 x 128)         | 4096  | 1308 us, 103 GB/s | 811 us, 166 GB/s | 1.61x |
| 7B MHA (32 x 128)         | 16384 | n/a (cap)      | 3082 us, 174 GB/s | -      |
| 8B GQA (32 x 128, 8 kv)   | 1024  | 129 us         | 21.7 us         | 5.96x    |
| 8B GQA (32 x 128, 8 kv)   | 4096  | 664 us         | 215 us, 156 GB/s | 3.09x   |
| 8B GQA (32 x 128, 8 kv)   | 16384 | n/a (cap)      | 795 us, 169 GB/s | -       |

From 1k positions (past the 24 MB L2) the split kernel reads at the
~170 GB/s DRAM ceiling; smaller caches stay in L2 between the benchmark's
calls, so those rows beat it. Only a context of under ~32 positions is a
little slower, where the old kernel was one block's worth of work anyway.
Chunk sizes of 16-128 and caps of 64-128 chunks were all within noise past
1k positions; 32 and 128 were best at 256.

stories15M decodes at most 256 positions, so end to end the gain is small:
seven interleaved runs, medians, fp32 0.677 -> 0.658 ms/tok, int8
0.375 -> 0.368 ms/tok; prefill unchanged; greedy and seeded sampled text
byte-identical. `test_cuda_attention` checks both kernels against a
double-precision reference for MHA, GQA (1, 2, 4 and 8 heads per block) and
MQA, 1-12 rows, contexts up to 20k, and that a repeated call gives the same
bits.

### int8 weights on the GPU
`-d gpu -w q8` stores the projections and classifier as int8 with the CPU's
per-output-channel scales (the embedding stays fp32). Every kernel is a
template over B's element type and widens and scales int8 as it loads it,
exactly as the CPU path does while packing, so the arithmetic stays fp32
and the GPU agrees with the CPU int8 model to float rounding:
`test_cuda_model` runs every decode and prefill check for both weight
types, and greedy and seeded sampled text are identical on CPU and GPU.

stories15M, seven interleaved runs of each, medians (this session ran ~10%
slower than the fp32 table above):

| GPU weights | Decode, ms / token | Prefill, 158 tok | Load   |
|-------------|--------------------|------------------|--------|
| fp32        | 0.669              | 7.3 ms           | 306 ms |
| int8        | 0.342              | 8.2 ms           | 417 ms |

Decode is 1.96x faster: the weights drop from ~61 MB to ~15 MB per token.
Prefill is ~1 ms slower, since the tiled kernels are compute-bound and now
also convert every B element they stage; load adds quantization on the host.

The m x 4096 x 4096 product (`bench_cuda`, int8 column):

| m  | fp32              | int8                         |
|----|-------------------|------------------------------|
| 1  | 0.387 ms          | 0.069 ms, 242 GB/s of int8   |
| 4  | 0.395 ms          | 0.139 ms                     |
| 8  | 0.425 ms          | 0.222 ms                     |
| 16 | 0.568 ms          | 0.562 ms                     |
| 64 | 0.704 ms          | 0.726 ms                     |

At m = 1 the 16.8 MB of int8 fits in the 24 MB L2, so repeated runs beat
the DRAM ceiling. m >= 16 is unchanged, already compute-bound.

**Four columns per thread.** The skinny kernel above loaded one byte per
thread per step, so a warp read just 32 bytes of a row of B at a time.
When 4 divides n (every stories15M shape), int8 B now goes to a variant
where each thread takes 4 consecutive columns as one 32-bit load (128 bytes
per warp), sums raw int8 values, and scales each column's sum once at the
end, as the CPU streaming path does. Same run, old and new kernel:

| m  | One column / thread | Four columns / thread      | Speed-up |
|----|---------------------|----------------------------|----------|
| 1  | 0.060 ms            | 0.034 ms, 492 GB/s (in L2) | 1.76x    |
| 4  | 0.131 ms            | 0.068 ms                   | 1.93x    |
| 8  | 0.214 ms            | 0.121 ms                   | 1.77x    |

The wider load is the gain: scaling per element in the new kernel costs
only ~3%. Eight and sixteen columns per thread were slower (m = 4: 0.117
and 0.176 ms), with fewer threads in flight and M x V accumulators each.
stories15M int8 decode 0.311 -> 0.283 ms/tok (seven interleaved runs,
medians); fp32 unchanged; greedy and seeded sampled text byte-identical.
Prefill is unchanged (3.7 ms for 77 tokens with `CUDA_MODULE_LOADING=EAGER`);
under the default lazy loading the reported prefill also includes loading
each kernel on first use, ~2 ms of it.

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

### CUDA build (Windows)
nvcc only works with MSVC as the host compiler, and MSVC objects can't link
into a MinGW build, so the GPU code builds in its own tree with `cl`. Needs
Visual Studio 2022 Build Tools (C++ workload) and the CUDA Toolkit. From an
"x64 Native Tools" prompt, with the toolkit's `bin\x64` on `PATH` for the
cuBLAS DLL the benchmark uses:

```bat
cmake -S . -B build-cuda -G Ninja -DCMAKE_CXX_COMPILER=cl -DENGINE_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda   # CPU tests, GPU kernels and GPU model vs. the CPU
build-cuda\bench_cuda
build-cuda\bench_attention
build-cuda\generate models\stories15M.bin -t 0 -d gpu
```

The kernels are compiled for the GPU in the machine (`native`); pass
`-DCMAKE_CUDA_ARCHITECTURES=89` (or another) to target a different one.
