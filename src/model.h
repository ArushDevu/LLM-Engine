#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "tensor.h"

// Llama-architecture decoder: pre-norm residual blocks of RMSNorm, rotary
// multi-head attention (optionally grouped-query) and a SwiGLU feed-forward.

struct Config {
    std::size_t dim;         // residual stream width
    std::size_t hidden_dim;  // feed-forward inner width
    std::size_t n_layers;
    std::size_t n_heads;     // query heads
    std::size_t n_kv_heads;  // key/value heads; fewer than n_heads is GQA
    std::size_t vocab_size;
    std::size_t seq_len;     // positions the KV cache holds

    std::size_t head_size() const { return dim / n_heads; }
    std::size_t kv_dim() const { return n_kv_heads * head_size(); }
};

// Projections are stored (in x out), the transpose of the checkpoint, so a
// row of activations times a weight is a plain matmul and the m <= 4 path
// streams each weight row once, in order. Projections that read the same
// input are concatenated by columns, so a layer issues 4 matmuls, not 7.
struct LayerWeights {
    std::vector<float> rms_att;  // dim
    Tensor wqkv;                 // dim x (dim + 2 kv_dim): q | k | v
    Tensor wo;                   // dim x dim
    std::vector<float> rms_ffn;  // dim
    Tensor w13;                  // dim x 2 hidden_dim: gate (w1) | up (w3)
    Tensor w2;                   // hidden_dim x dim
};

struct Weights {
    Tensor token_embedding;  // vocab_size x dim
    std::vector<LayerWeights> layers;
    std::vector<float> rms_final;  // dim
    Tensor wcls;                   // dim x vocab_size
};

struct Checkpoint {
    Config config;
    Weights weights;
};

// Loads a llama2.c checkpoint (legacy version-0 format): seven int32 header
// fields in Config order, then fp32 weights, little-endian. A negative
// vocab_size marks a classifier stored separately instead of tied to the
// embedding. Throws std::runtime_error if the file can't be read, the header
// is inconsistent, or the file size doesn't match the header.
Checkpoint load_checkpoint(const std::string& path);

// Runs the model one token at a time. Keys and values of every position fed
// so far are cached, so each new token attends to the past without
// recomputing it.
class Transformer {
public:
    explicit Transformer(Checkpoint ckpt);

    // Feeds token at position pos and returns logits for the next token. pos
    // may be at most the number of positions already fed, so the caller can
    // continue or rewind and overwrite, but never skip ahead. The returned
    // reference is overwritten by the next call. Throws std::out_of_range for
    // a token outside the vocabulary, pos >= seq_len, or a skipped position.
    const std::vector<float>& forward(int token, std::size_t pos);

    const Config& config() const { return cfg_; }

private:
    // Causal attention of the current query against cached positions
    // [0, n) of layer l; writes one head_size slice per query head to out.
    void attention(std::size_t l, std::size_t n, const float* q, float* out);

    Config cfg_;
    Weights w_;
    std::vector<Tensor> key_cache_, value_cache_;  // per layer: seq_len x kv_dim
    std::vector<float> rope_cos_, rope_sin_;       // seq_len x head_size / 2
    std::size_t filled_ = 0;                       // positions in the cache

    // Per-call scratch, sized once.
    Tensor x_;      // 1 x dim residual stream
    Tensor att_;    // 1 x dim attention output, heads side by side
    Tensor hb_;     // 1 x hidden_dim gated feed-forward activation
    std::vector<float> scores_;  // seq_len
    std::vector<float> logits_;  // vocab_size
};
