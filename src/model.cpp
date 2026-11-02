#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "matmul.h"
#include "ops.h"

namespace {

class Reader {
public:
    explicit Reader(const std::string& path) : path_(path), in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("cannot open checkpoint " + path);
    }

    std::uint64_t size() {
        in_.seekg(0, std::ios::end);
        const auto end = static_cast<std::uint64_t>(in_.tellg());
        in_.seekg(0, std::ios::beg);
        return end;
    }

    template <class T>
    void read(T* dst, std::size_t n) {
        in_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n * sizeof(T)));
        if (!in_) throw std::runtime_error("checkpoint " + path_ + " ended early");
    }

    void read(std::vector<float>& dst, std::size_t n) {
        dst.resize(n);
        read(dst.data(), n);
    }

    void skip_floats(std::size_t n) {
        in_.seekg(static_cast<std::streamoff>(n * sizeof(float)), std::ios::cur);
    }

    // Reads an (out x in) row-major matrix and writes its transpose into
    // dst[0:in, col0:col0 + out].
    void read_transposed(std::size_t out, std::size_t in, Tensor& dst, std::size_t col0) {
        buf_.resize(out * in);
        read(buf_.data(), buf_.size());
        for (std::size_t o = 0; o < out; ++o) {
            const float* src = buf_.data() + o * in;
            for (std::size_t i = 0; i < in; ++i) dst.at(i, col0 + o) = src[i];
        }
    }

private:
    std::string path_;
    std::ifstream in_;
    std::vector<float> buf_;
};

Config parse_header(const std::int32_t h[7], bool& shared_classifier, const std::string& path) {
    shared_classifier = h[5] > 0;
    for (int i = 0; i < 7; ++i) {
        if (h[i] == 0 || (i != 5 && h[i] < 0)) {
            throw std::runtime_error("checkpoint " + path + ": bad header field " +
                                     std::to_string(i) + " = " + std::to_string(h[i]));
        }
    }
    auto u = [](std::int32_t v) { return static_cast<std::size_t>(v < 0 ? -std::int64_t{v} : v); };
    Config c{u(h[0]), u(h[1]), u(h[2]), u(h[3]), u(h[4]), u(h[5]), u(h[6])};
    if (c.dim % c.n_heads != 0 || c.head_size() % 2 != 0 || c.n_heads % c.n_kv_heads != 0) {
        throw std::runtime_error("checkpoint " + path +
                                 ": dim must split into even-sized heads, and query heads "
                                 "into whole groups per kv head");
    }
    return c;
}

// File size implied by the header, in bytes.
std::uint64_t expected_size(const Config& c, bool shared_classifier) {
    const std::uint64_t D = c.dim, H = c.hidden_dim, KV = c.kv_dim(), V = c.vocab_size;
    const std::uint64_t per_layer = 2 * D + 2 * D * D + 2 * D * KV + 3 * D * H;
    std::uint64_t floats = V * D + c.n_layers * per_layer + D;
    floats += std::uint64_t{c.seq_len} * c.head_size();  // legacy RoPE tables
    if (!shared_classifier) floats += V * D;
    return 7 * sizeof(std::int32_t) + floats * sizeof(float);
}

// Rotates consecutive pairs (v[2j], v[2j+1]) of each head by the position's
// angle for frequency j. llama2.c's export permutes q and k weights so this
// interleaved layout matches the reference half-split rotation.
void rope(float* v, std::size_t heads, std::size_t head_size, const float* cos,
          const float* sin) {
    for (std::size_t h = 0; h < heads; ++h) {
        float* p = v + h * head_size;
        for (std::size_t j = 0; j < head_size / 2; ++j) {
            const float a = p[2 * j], b = p[2 * j + 1];
            p[2 * j] = a * cos[j] - b * sin[j];
            p[2 * j + 1] = a * sin[j] + b * cos[j];
        }
    }
}

void add_into(Tensor& x, const Tensor& y) {
    for (std::size_t i = 0; i < x.data.size(); ++i) x.data[i] += y.data[i];
}

} // namespace

Checkpoint load_checkpoint(const std::string& path) {
    Reader r(path);
    const std::uint64_t file_size = r.size();
    std::int32_t header[7];
    r.read(header, 7);
    bool shared = false;
    const Config c = parse_header(header, shared, path);
    if (const std::uint64_t want = expected_size(c, shared); file_size != want) {
        throw std::runtime_error("checkpoint " + path + " is " + std::to_string(file_size) +
                                 " bytes, header implies " + std::to_string(want));
    }

    const std::size_t D = c.dim, H = c.hidden_dim, KV = c.kv_dim(), V = c.vocab_size;
    Checkpoint ckpt{c, {}};
    Weights& w = ckpt.weights;
    w.layers.resize(c.n_layers);
    for (auto& l : w.layers) {
        l.wqkv = Tensor(D, D + 2 * KV);
        l.wo = Tensor(D, D);
        l.w13 = Tensor(D, 2 * H);
        l.w2 = Tensor(H, D);
    }

    // The file groups each weight across all layers: every layer's wq, then
    // every layer's wk, and so on.
    w.token_embedding = Tensor(V, D);
    r.read(w.token_embedding.data.data(), V * D);
    for (auto& l : w.layers) r.read(l.rms_att, D);
    for (auto& l : w.layers) r.read_transposed(D, D, l.wqkv, 0);
    for (auto& l : w.layers) r.read_transposed(KV, D, l.wqkv, D);
    for (auto& l : w.layers) r.read_transposed(KV, D, l.wqkv, D + KV);
    for (auto& l : w.layers) r.read_transposed(D, D, l.wo, 0);
    for (auto& l : w.layers) r.read(l.rms_ffn, D);
    for (auto& l : w.layers) r.read_transposed(H, D, l.w13, 0);
    for (auto& l : w.layers) r.read_transposed(D, H, l.w2, 0);
    for (auto& l : w.layers) r.read_transposed(H, D, l.w13, H);
    r.read(w.rms_final, D);
    // RoPE tables: recomputed at load, as llama2.c does.
    r.skip_floats(c.seq_len * c.head_size());

    w.wcls = Tensor(D, V);
    if (shared) {
        for (std::size_t t = 0; t < V; ++t)
            for (std::size_t i = 0; i < D; ++i) w.wcls.at(i, t) = w.token_embedding.at(t, i);
    } else {
        r.read_transposed(V, D, w.wcls, 0);
    }
    return ckpt;
}

Transformer::Transformer(Checkpoint ckpt)
    : cfg_(ckpt.config), w_(std::move(ckpt.weights)),
      x_(1, cfg_.dim), att_(1, cfg_.dim), hb_(1, cfg_.hidden_dim),
      scores_(cfg_.seq_len), logits_(cfg_.vocab_size) {
    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        key_cache_.emplace_back(cfg_.seq_len, cfg_.kv_dim());
        value_cache_.emplace_back(cfg_.seq_len, cfg_.kv_dim());
    }
    // Same float expressions as llama2.c, so the angles match it bit for bit.
    const std::size_t half = cfg_.head_size() / 2;
    rope_cos_.resize(cfg_.seq_len * half);
    rope_sin_.resize(cfg_.seq_len * half);
    for (std::size_t pos = 0; pos < cfg_.seq_len; ++pos) {
        for (std::size_t j = 0; j < half; ++j) {
            const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * j) /
                                                          static_cast<float>(cfg_.head_size()));
            const float angle = static_cast<float>(pos) * freq;
            rope_cos_[pos * half + j] = std::cos(angle);
            rope_sin_[pos * half + j] = std::sin(angle);
        }
    }
}

void Transformer::attention(std::size_t l, std::size_t n, const float* q, float* out) {
    const std::size_t hs = cfg_.head_size(), kv_dim = cfg_.kv_dim();
    const std::size_t group = cfg_.n_heads / cfg_.n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    const float* K = key_cache_[l].data.data();
    const float* V = value_cache_[l].data.data();
    for (std::size_t h = 0; h < cfg_.n_heads; ++h) {
        const float* qh = q + h * hs;
        const std::size_t kv_off = (h / group) * hs;
        for (std::size_t t = 0; t < n; ++t) {
            const float* k = K + t * kv_dim + kv_off;
            float dot = 0.0f;
            for (std::size_t i = 0; i < hs; ++i) dot += qh[i] * k[i];
            scores_[t] = dot * scale;
        }
        softmax(scores_.data(), n);
        float* oh = out + h * hs;
        std::fill(oh, oh + hs, 0.0f);
        for (std::size_t t = 0; t < n; ++t) {
            const float* v = V + t * kv_dim + kv_off;
            const float a = scores_[t];
            for (std::size_t i = 0; i < hs; ++i) oh[i] += a * v[i];
        }
    }
}

const std::vector<float>& Transformer::forward(int token, std::size_t pos) {
    if (token < 0 || static_cast<std::size_t>(token) >= cfg_.vocab_size)
        throw std::out_of_range("forward: token " + std::to_string(token) + " outside vocab");
    if (pos >= cfg_.seq_len || pos > filled_)
        throw std::out_of_range("forward: position " + std::to_string(pos) + " with " +
                                std::to_string(filled_) + " cached of " +
                                std::to_string(cfg_.seq_len));

    const std::size_t D = cfg_.dim, H = cfg_.hidden_dim, KV = cfg_.kv_dim();
    const std::size_t half = cfg_.head_size() / 2;
    const float* cos = rope_cos_.data() + pos * half;
    const float* sin = rope_sin_.data() + pos * half;

    const float* emb = w_.token_embedding.data.data() + static_cast<std::size_t>(token) * D;
    std::copy(emb, emb + D, x_.data.begin());

    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        const LayerWeights& lw = w_.layers[l];

        Tensor qkv = matmul_parallel(rmsnorm(x_, lw.rms_att), lw.wqkv);
        float* q = qkv.data.data();
        float* k = q + D;
        float* v = k + KV;
        rope(q, cfg_.n_heads, cfg_.head_size(), cos, sin);
        rope(k, cfg_.n_kv_heads, cfg_.head_size(), cos, sin);
        std::copy(k, k + KV, key_cache_[l].data.begin() + static_cast<std::ptrdiff_t>(pos * KV));
        std::copy(v, v + KV, value_cache_[l].data.begin() + static_cast<std::ptrdiff_t>(pos * KV));

        attention(l, pos + 1, q, att_.data.data());
        add_into(x_, matmul_parallel(att_, lw.wo));

        // SwiGLU: w2(silu(w1 x) * w3 x).
        const Tensor gu = matmul_parallel(rmsnorm(x_, lw.rms_ffn), lw.w13);
        for (std::size_t i = 0; i < H; ++i) {
            const float g = gu.data[i];
            hb_.data[i] = g / (1.0f + std::exp(-g)) * gu.data[H + i];
        }
        add_into(x_, matmul_parallel(hb_, lw.w2));
    }
    filled_ = pos + 1;

    const Tensor logits = matmul_parallel(rmsnorm(x_, w_.rms_final), w_.wcls);
    std::copy(logits.data.begin(), logits.data.end(), logits_.begin());
    return logits_;
}
