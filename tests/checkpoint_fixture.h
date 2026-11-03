#pragma once
// Small random llama2.c checkpoints for the model tests: the raw weights in
// the file's own layout, and a writer for them.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// Weights as laid out in the file, one entry per layer where it applies.
struct RawModel {
    int dim, hidden, layers, heads, kv_heads, vocab, seq_len;
    bool shared;
    std::vector<float> emb, rms_final, wcls;
    std::vector<std::vector<float>> rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3;

    int head_size() const { return dim / heads; }
    int kv_dim() const { return kv_heads * head_size(); }
};

inline std::vector<float> random_vec(std::size_t n, float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

inline RawModel make_model(int heads, int kv_heads, bool shared, std::uint32_t seed) {
    std::mt19937 rng(seed);
    RawModel m{16, 40, 2, heads, kv_heads, 29, 8, shared, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
    const auto D = std::size_t(m.dim), H = std::size_t(m.hidden), KV = std::size_t(m.kv_dim());
    const auto V = std::size_t(m.vocab);
    m.emb = random_vec(V * D, 1.0f, rng);
    for (int l = 0; l < m.layers; ++l) {
        std::vector<float> ones_ish = random_vec(D, 0.2f, rng);
        for (auto& x : ones_ish) x += 1.0f;
        m.rms_att.push_back(ones_ish);
        m.rms_ffn.push_back(random_vec(D, 0.2f, rng));
        m.wq.push_back(random_vec(D * D, 0.4f, rng));
        m.wk.push_back(random_vec(KV * D, 0.4f, rng));
        m.wv.push_back(random_vec(KV * D, 0.4f, rng));
        m.wo.push_back(random_vec(D * D, 0.3f, rng));
        m.w1.push_back(random_vec(H * D, 0.3f, rng));
        m.w2.push_back(random_vec(D * H, 0.3f, rng));
        m.w3.push_back(random_vec(H * D, 0.3f, rng));
    }
    m.rms_final = random_vec(D, 1.0f, rng);
    if (!shared) m.wcls = random_vec(V * D, 1.0f, rng);
    return m;
}

inline void write_checkpoint(const RawModel& m, const std::string& path, std::size_t drop_bytes = 0) {
    std::vector<char> bytes;
    auto put = [&](const void* p, std::size_t n) {
        const char* c = static_cast<const char*>(p);
        bytes.insert(bytes.end(), c, c + n);
    };
    const std::int32_t header[7] = {m.dim, m.hidden, m.layers, m.heads, m.kv_heads,
                                    m.shared ? m.vocab : -m.vocab, m.seq_len};
    put(header, sizeof header);
    auto put_vec = [&](const std::vector<float>& v) { put(v.data(), v.size() * sizeof(float)); };
    auto put_all = [&](const std::vector<std::vector<float>>& vs) {
        for (const auto& v : vs) put_vec(v);
    };
    put_vec(m.emb);
    put_all(m.rms_att);
    put_all(m.wq);
    put_all(m.wk);
    put_all(m.wv);
    put_all(m.wo);
    put_all(m.rms_ffn);
    put_all(m.w1);
    put_all(m.w2);
    put_all(m.w3);
    put_vec(m.rms_final);
    // Legacy RoPE tables, ignored by the loader: fill with garbage to prove it.
    put_vec(std::vector<float>(std::size_t(m.seq_len * m.head_size()), 123.0f));
    if (!m.shared) put_vec(m.wcls);
    bytes.resize(bytes.size() - drop_bytes);
    std::ofstream(path, std::ios::binary).write(bytes.data(), std::streamsize(bytes.size()));
}

inline std::string temp_path(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}
