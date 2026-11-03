#pragma once
#include <cstddef>
#include <vector>

#include "cuda_matmul.h"
#include "model.h"

struct CUgraphExec_st;  // cudaGraphExec_t, without including the CUDA headers

// The Transformer's forward pass on the GPU, fp32. Weights, the KV cache and
// every activation stay in GPU memory; a call uploads its position and token
// ids and downloads only the final logits. Same interface and position rules
// as Transformer, and the same results up to float rounding. Built only with
// -DENGINE_CUDA=ON; not safe to use from several host threads at once.
//
// A decode step is ~100 small kernels, and on Windows each launch costs
// ~15 us of host time, several times the work. So the first forward()
// captures the whole one-token step into a CUDA graph, and later calls
// replay it with one launch. Kernels read the position and token from device
// memory, so one graph serves every position; it is captured again only when
// a buffer it points at has moved.
class CudaTransformer {
public:
    // Copies the weights to the GPU. Throws std::runtime_error on a CUDA
    // error, or if seq_len is too long for attention's shared-memory scores.
    explicit CudaTransformer(const Checkpoint& ckpt);
    ~CudaTransformer();

    CudaTransformer(const CudaTransformer&) = delete;
    CudaTransformer& operator=(const CudaTransformer&) = delete;

    // As Transformer::forward() and Transformer::prefill().
    const std::vector<float>& forward(int token, std::size_t pos);
    const std::vector<float>& prefill(const std::vector<int>& tokens, std::size_t pos);

    const Config& config() const { return cfg_; }

private:
    const std::vector<float>& run(const int* tokens, std::size_t n, std::size_t pos);
    // Queues the whole pass for n rows on the calling thread's stream: copy
    // the feed in, run every layer, copy the logits out.
    void enqueue(std::size_t n);
    // Grows the activation buffers to hold n rows.
    void reserve_rows(std::size_t n);
    std::size_t buffers_generation() const;

    struct Layer {
        DeviceTensor rms_att, wqkv, wo, rms_ffn, w13, w2;
    };

    Config cfg_;
    DeviceTensor token_embedding_, rms_final_, wcls_;
    std::vector<Layer> layers_;
    std::vector<DeviceTensor> key_cache_, value_cache_;  // per layer: seq_len x kv_dim
    DeviceTensor rope_cos_, rope_sin_;                   // seq_len x head_size / 2
    std::size_t filled_ = 0;

    // Activations, rows_ rows each, grow-only; only the first n are used.
    std::size_t rows_ = 0, own_generation_ = 0;
    DeviceTensor x_, xn_, qkv_, att_, proj_, gu_, hb_;
    DeviceTensor last_, logits_dev_;  // 1 x dim, 1 x vocab_size

    // Pinned host buffers, so the copies can be graph nodes. feed is
    // [pos, tokens...]; feed_dev_ is its device copy.
    int* feed_host_ = nullptr;
    int* feed_dev_ = nullptr;
    float* logits_host_ = nullptr;
    std::vector<float> logits_;

    CUgraphExec_st* decode_graph_ = nullptr;
    std::size_t graph_generation_ = 0;  // buffers_generation() at capture
};
