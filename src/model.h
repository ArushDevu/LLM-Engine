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

// How the Transformer stores projection weights. q8 quantizes every
// projection and the classifier to int8 with a scale per output channel
// (quant.h), a quarter of the bytes to stream per token; the embedding stays
// fp32, since a token reads only its own row.
enum class WeightType { f32, q8 };

// Runs the model over positions fed in order. Keys and values of every
// position fed so far are cached, so new tokens attend to the past without
// recomputing it.
class Transformer {
public:
    explicit Transformer(Checkpoint ckpt, WeightType type = WeightType::f32);

    // Feeds token at position pos and returns logits for the next token. pos
    // may be at most the number of positions already fed, so the caller can
    // continue or rewind and overwrite, but never skip ahead. The returned
    // reference is overwritten by the next call. Throws std::out_of_range for
    // a token outside the vocabulary, pos >= seq_len, or a skipped position.
    const std::vector<float>& forward(int token, std::size_t pos);

    // Feeds tokens at positions [pos, pos + tokens.size()) in one pass and
    // returns logits for the token after the last; same result as calling
    // forward() on each in turn. Every projection becomes an m = tokens.size()
    // GEMM instead of m = 1 products that each stream all the weights, and the
    // classifier runs only for the last row. Same position rules and
    // reference lifetime as forward(); also throws std::invalid_argument for
    // an empty tokens.
    const std::vector<float>& prefill(const std::vector<int>& tokens, std::size_t pos);

    const Config& config() const { return cfg_; }

private:
    // Runs tokens[0:n] at positions [pos, pos + n).
    const std::vector<float>& run(const int* tokens, std::size_t n, std::size_t pos);

    // x * w, or x * the int8 copy of w when quantized.
    Tensor project(const Tensor& x, const Tensor& w, const QTensor& wq) const;

    // Causal attention of layer l for rows fed at positions [pos, pos +
    // qkv.rows): row i's queries (the q columns of qkv) attend to cached
    // positions [0, pos + i]. Writes one head_size slice per query head to
    // the matching row of out.
    void attention(std::size_t l, std::size_t pos, const Tensor& qkv, Tensor& out);

    Config cfg_;
    Weights w_;  // projections and classifier are emptied when quantized

    // int8 copies, populated only for WeightType::q8.
    struct QuantLayer {
        QTensor wqkv, wo, w13, w2;
    };
    bool quantized_ = false;
    std::vector<QuantLayer> qlayers_;
    QTensor qcls_;
    std::vector<Tensor> key_cache_, value_cache_;  // per layer: seq_len x kv_dim
    std::vector<float> rope_cos_, rope_sin_;       // seq_len x head_size / 2
    std::size_t filled_ = 0;                       // positions in the cache

    // Per-call scratch, one row per token fed; reallocated only when the
    // number of rows changes.
    Tensor x_;      // n x dim residual stream
    Tensor att_;    // n x dim attention output, heads side by side
    Tensor hb_;     // n x hidden_dim gated feed-forward activation
    std::vector<float> logits_;  // vocab_size
};
