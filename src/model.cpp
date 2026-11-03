#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "matmul.h"
#include "ops.h"
#include "quant.h"
#include "thread_pool.h"

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

void resize_rows(Tensor& t, std::size_t rows) {
    if (t.rows != rows) t = Tensor(rows, t.cols);
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

void check_feed(const Config& cfg, const int* tokens, std::size_t n, std::size_t pos,
                std::size_t filled) {
    for (std::size_t i = 0; i < n; ++i) {
        if (tokens[i] < 0 || static_cast<std::size_t>(tokens[i]) >= cfg.vocab_size)
            throw std::out_of_range("forward: token " + std::to_string(tokens[i]) +
                                    " outside vocab");
    }
    if (pos + n > cfg.seq_len || pos > filled)
        throw std::out_of_range("forward: positions " + std::to_string(pos) + ".." +
                                std::to_string(pos + n - 1) + " with " +
                                std::to_string(filled) + " cached of " +
                                std::to_string(cfg.seq_len));
}

void rope_tables(const Config& cfg, std::vector<float>& cos, std::vector<float>& sin) {
    const std::size_t half = cfg.head_size() / 2;
    cos.resize(cfg.seq_len * half);
    sin.resize(cfg.seq_len * half);
    for (std::size_t pos = 0; pos < cfg.seq_len; ++pos) {
        for (std::size_t j = 0; j < half; ++j) {
            const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * j) /
                                                          static_cast<float>(cfg.head_size()));
            const float angle = static_cast<float>(pos) * freq;
            cos[pos * half + j] = std::cos(angle);
            sin[pos * half + j] = std::sin(angle);
        }
    }
}

Transformer::Transformer(Checkpoint ckpt, WeightType type)
    : cfg_(ckpt.config), w_(std::move(ckpt.weights)), quantized_(type == WeightType::q8),
      x_(1, cfg_.dim), att_(1, cfg_.dim), hb_(1, cfg_.hidden_dim),
      logits_(cfg_.vocab_size) {
    if (quantized_) {
        // Quantize and free each fp32 matrix in turn, so peak memory stays
        // near the fp32 model's.
        auto take = [](Tensor& t) {
            QTensor q = quantize_q8(t);
            t = Tensor{};
            return q;
        };
        for (LayerWeights& lw : w_.layers)
            qlayers_.push_back({take(lw.wqkv), take(lw.wo), take(lw.w13), take(lw.w2)});
        qcls_ = take(w_.wcls);
    }
    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        key_cache_.emplace_back(cfg_.seq_len, cfg_.kv_dim());
        value_cache_.emplace_back(cfg_.seq_len, cfg_.kv_dim());
    }
    rope_tables(cfg_, rope_cos_, rope_sin_);
}

void Transformer::attention(std::size_t l, std::size_t pos, const Tensor& qkv, Tensor& out) {
    const std::size_t hs = cfg_.head_size(), kv_dim = cfg_.kv_dim();
    const std::size_t group = cfg_.n_heads / cfg_.n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    const float* K = key_cache_[l].data.data();
    const float* V = value_cache_[l].data.data();

    // One task per (row, head): they share only read-only cache rows, and
    // each writes its own slice of out.
    auto run_head = [&](std::size_t task) {
        const std::size_t i = task / cfg_.n_heads, h = task % cfg_.n_heads;
        const std::size_t n = pos + i + 1;
        thread_local std::vector<float> scores;
        if (scores.size() < n) scores.resize(cfg_.seq_len);
        const float* qh = qkv.data.data() + i * qkv.cols + h * hs;
        const std::size_t kv_off = (h / group) * hs;
        for (std::size_t t = 0; t < n; ++t) {
            const float* k = K + t * kv_dim + kv_off;
            float dot = 0.0f;
            for (std::size_t j = 0; j < hs; ++j) dot += qh[j] * k[j];
            scores[t] = dot * scale;
        }
        softmax(scores.data(), n);
        float* oh = out.data.data() + i * out.cols + h * hs;
        std::fill(oh, oh + hs, 0.0f);
        for (std::size_t t = 0; t < n; ++t) {
            const float* v = V + t * kv_dim + kv_off;
            const float a = scores[t];
            for (std::size_t j = 0; j < hs; ++j) oh[j] += a * v[j];
        }
    };
    const std::size_t tasks = qkv.rows * cfg_.n_heads;
    if (tasks == 1) {
        run_head(0);
    } else {
        ThreadPool::global().run(tasks, run_head);
    }
}

Tensor Transformer::project(const Tensor& x, const Tensor& w, const QTensor& wq) const {
    return quantized_ ? matmul_parallel(x, wq) : matmul_parallel(x, w);
}

const std::vector<float>& Transformer::forward(int token, std::size_t pos) {
    return run(&token, 1, pos);
}

const std::vector<float>& Transformer::prefill(const std::vector<int>& tokens, std::size_t pos) {
    if (tokens.empty()) throw std::invalid_argument("prefill: no tokens");
    return run(tokens.data(), tokens.size(), pos);
}

const std::vector<float>& Transformer::run(const int* tokens, std::size_t n, std::size_t pos) {
    check_feed(cfg_, tokens, n, pos, filled_);

    const std::size_t D = cfg_.dim, H = cfg_.hidden_dim, KV = cfg_.kv_dim();
    const std::size_t half = cfg_.head_size() / 2;
    resize_rows(x_, n);
    resize_rows(att_, n);
    resize_rows(hb_, n);

    for (std::size_t i = 0; i < n; ++i) {
        const float* emb = w_.token_embedding.data.data() + static_cast<std::size_t>(tokens[i]) * D;
        std::copy(emb, emb + D, x_.data.begin() + static_cast<std::ptrdiff_t>(i * D));
    }

    static const QuantLayer kNoQuant;
    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        const LayerWeights& lw = w_.layers[l];
        const QuantLayer& ql = quantized_ ? qlayers_[l] : kNoQuant;

        Tensor qkv = project(rmsnorm(x_, lw.rms_att), lw.wqkv, ql.wqkv);
        for (std::size_t i = 0; i < n; ++i) {
            float* q = qkv.data.data() + i * qkv.cols;
            float* k = q + D;
            float* v = k + KV;
            const float* cos = rope_cos_.data() + (pos + i) * half;
            const float* sin = rope_sin_.data() + (pos + i) * half;
            rope(q, cfg_.n_heads, cfg_.head_size(), cos, sin);
            rope(k, cfg_.n_kv_heads, cfg_.head_size(), cos, sin);
            const auto at = static_cast<std::ptrdiff_t>((pos + i) * KV);
            std::copy(k, k + KV, key_cache_[l].data.begin() + at);
            std::copy(v, v + KV, value_cache_[l].data.begin() + at);
        }

        attention(l, pos, qkv, att_);
        add_into(x_, project(att_, lw.wo, ql.wo));

        // SwiGLU: w2(silu(w1 x) * w3 x).
        const Tensor gu = project(rmsnorm(x_, lw.rms_ffn), lw.w13, ql.w13);
        for (std::size_t i = 0; i < n; ++i) {
            const float* g = gu.data.data() + i * 2 * H;
            float* hb = hb_.data.data() + i * H;
            for (std::size_t j = 0; j < H; ++j) hb[j] = g[j] / (1.0f + std::exp(-g[j])) * g[H + j];
        }
        add_into(x_, project(hb_, lw.w2, ql.w2));
    }
    filled_ = pos + n;

    // Only the last row's logits are returned.
    Tensor last(1, D);
    std::copy(x_.data.end() - static_cast<std::ptrdiff_t>(D), x_.data.end(), last.data.begin());
    const Tensor logits = project(rmsnorm(last, w_.rms_final), w_.wcls, qcls_);
    std::copy(logits.data.begin(), logits.data.end(), logits_.begin());
    return logits_;
}
