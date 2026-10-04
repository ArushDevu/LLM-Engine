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
- [ ] Multithreaded / cache-optimized CPU matmul
- [ ] Load model weights (stories15M)
- [ ] Full transformer forward pass on CPU
- [ ] Text generation with sampling
- [ ] CUDA kernels: matmul, softmax, RMSNorm, attention
- [ ] KV cache
- [ ] int8 quantization
- [ ] Benchmarks vs llama.cpp

## Benchmarks
| Version | 512x512 matmul | GFLOPS |
|---------|----------------|--------|
| Naive CPU (i-k-j loop) | TBD | TBD |

## Build
```bash
cmake -S . -B build
cmake --build build
./build/engine
```