// Checks the cached, transposed, fused forward pass against a reference that
// works straight from the checkpoint's layout: (out x in) weights, every
// position recomputed from scratch, double accumulation, and its own RoPE.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "checkpoint_fixture.h"
#include "model.h"
#include "quant.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                         #cond);                                                 \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using Vec = std::vector<double>;

// y = W x for W stored (out x in).
Vec matvec(const std::vector<float>& W, const Vec& x, std::size_t out) {
    const std::size_t in = x.size();
    Vec y(out, 0.0);
    for (std::size_t o = 0; o < out; ++o)
        for (std::size_t i = 0; i < in; ++i) y[o] += double(W[o * in + i]) * x[i];
    return y;
}

Vec rmsnorm_ref(const Vec& x, const std::vector<float>& w) {
    double ss = 0.0;
    for (double v : x) ss += v * v;
    const double s = 1.0 / std::sqrt(ss / double(x.size()) + 1e-5);
    Vec y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * s * w[i];
    return y;
}

void rope_ref(Vec& v, int pos, int head_size) {
    for (std::size_t i = 0; i + 1 < v.size(); i += 2) {
        const double j = double(i % std::size_t(head_size)) / head_size;
        const double angle = pos * std::pow(10000.0, -j);
        const double a = v[i], b = v[i + 1];
        v[i] = a * std::cos(angle) - b * std::sin(angle);
        v[i + 1] = a * std::sin(angle) + b * std::cos(angle);
    }
}

// Logits after the last of tokens, recomputing every position.
Vec reference_logits(const RawModel& m, const std::vector<int>& tokens) {
    const std::size_t D = std::size_t(m.dim), H = std::size_t(m.hidden);
    const std::size_t KV = std::size_t(m.kv_dim()), hs = std::size_t(m.head_size());
    const std::size_t T = tokens.size(), group = std::size_t(m.heads / m.kv_heads);

    std::vector<Vec> x(T);
    for (std::size_t t = 0; t < T; ++t)
        x[t].assign(m.emb.begin() + long(std::size_t(tokens[t]) * D),
                    m.emb.begin() + long(std::size_t(tokens[t] + 1) * D));

    for (std::size_t l = 0; l < std::size_t(m.layers); ++l) {
        std::vector<Vec> q(T), k(T), v(T);
        for (std::size_t t = 0; t < T; ++t) {
            const Vec xn = rmsnorm_ref(x[t], m.rms_att[l]);
            q[t] = matvec(m.wq[l], xn, D);
            k[t] = matvec(m.wk[l], xn, KV);
            v[t] = matvec(m.wv[l], xn, KV);
            rope_ref(q[t], int(t), int(hs));
            rope_ref(k[t], int(t), int(hs));
        }
        for (std::size_t t = 0; t < T; ++t) {
            Vec att(D, 0.0);
            for (std::size_t h = 0; h < std::size_t(m.heads); ++h) {
                const std::size_t kvo = (h / group) * hs;
                Vec s(t + 1);
                double max = -1e300;
                for (std::size_t u = 0; u <= t; ++u) {
                    double dot = 0.0;
                    for (std::size_t i = 0; i < hs; ++i) dot += q[t][h * hs + i] * k[u][kvo + i];
                    s[u] = dot / std::sqrt(double(hs));
                    max = std::max(max, s[u]);
                }
                double sum = 0.0;
                for (double& e : s) sum += (e = std::exp(e - max));
                for (std::size_t u = 0; u <= t; ++u)
                    for (std::size_t i = 0; i < hs; ++i)
                        att[h * hs + i] += s[u] / sum * v[u][kvo + i];
            }
            const Vec o = matvec(m.wo[l], att, D);
            for (std::size_t i = 0; i < D; ++i) x[t][i] += o[i];

            const Vec xn = rmsnorm_ref(x[t], m.rms_ffn[l]);
            const Vec g = matvec(m.w1[l], xn, H), u = matvec(m.w3[l], xn, H);
            Vec hb(H);
            for (std::size_t i = 0; i < H; ++i) hb[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
            const Vec f = matvec(m.w2[l], hb, D);
            for (std::size_t i = 0; i < D; ++i) x[t][i] += f[i];
        }
    }
    const Vec xn = rmsnorm_ref(x[T - 1], m.rms_final);
    return matvec(m.shared ? m.emb : m.wcls, xn, std::size_t(m.vocab));
}

double max_abs_diff(const std::vector<float>& got, const Vec& want) {
    double d = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) d = std::max(d, std::fabs(got[i] - want[i]));
    return d;
}

constexpr double kTol = 1e-4;

// Feeds a full context one token at a time and checks the logits after each.
void test_matches_reference(int heads, int kv_heads, bool shared) {
    const RawModel m = make_model(heads, kv_heads, shared, std::uint32_t(heads * 10 + kv_heads));
    const std::string path = temp_path("engine_test_model.bin");
    write_checkpoint(m, path);
    Transformer model(load_checkpoint(path));
    std::filesystem::remove(path);

    std::vector<int> tokens;
    for (int pos = 0; pos < m.seq_len; ++pos) {
        tokens.push_back((pos * 7 + 3) % m.vocab);
        const auto& logits = model.forward(tokens.back(), std::size_t(pos));
        const double diff = max_abs_diff(logits, reference_logits(m, tokens));
        if (diff > kTol) {
            std::fprintf(stderr, "heads %d/%d shared %d pos %d: max diff %g\n", heads,
                         kv_heads, int(shared), pos, diff);
        }
        CHECK(diff <= kTol);
    }
}

// Prefill in chunks of 1, 1 and 5 rows (the streaming path, then the packed
// GEMM above m = 4) matches the reference at each chunk's last position, and
// a decode step after it sees the cache prefill wrote.
void test_prefill(int heads, int kv_heads, bool shared) {
    const RawModel m = make_model(heads, kv_heads, shared, std::uint32_t(heads * 10 + kv_heads));
    const std::string path = temp_path("engine_test_prefill.bin");
    write_checkpoint(m, path);
    Transformer model(load_checkpoint(path));
    std::filesystem::remove(path);

    std::vector<int> tokens;
    for (int pos = 0; pos < m.seq_len; ++pos) tokens.push_back((pos * 5 + 2) % m.vocab);
    std::size_t pos = 0;
    for (std::size_t chunk : {1, 1, 5}) {
        const std::vector<int> part(tokens.begin() + long(pos), tokens.begin() + long(pos + chunk));
        const auto& logits = model.prefill(part, pos);
        pos += chunk;
        const std::vector<int> prefix(tokens.begin(), tokens.begin() + long(pos));
        CHECK(max_abs_diff(logits, reference_logits(m, prefix)) <= kTol);
    }
    const auto& logits = model.forward(tokens[pos], pos);
    const std::vector<int> prefix(tokens.begin(), tokens.begin() + long(pos + 1));
    CHECK(max_abs_diff(logits, reference_logits(m, prefix)) <= kTol);

    // Whole context in one pass, rewinding over what is cached.
    CHECK(max_abs_diff(model.prefill(tokens, 0), reference_logits(m, tokens)) <= kTol);
}

// W (out x in, as in the file) after an int8 round trip with one scale per
// output channel, the same scheme the Transformer applies to its (in x out)
// copy per column.
std::vector<float> q8_round_trip(const std::vector<float>& W, std::size_t out, std::size_t in) {
    Tensor t(in, out);
    for (std::size_t o = 0; o < out; ++o)
        for (std::size_t i = 0; i < in; ++i) t.at(i, o) = W[o * in + i];
    const Tensor d = dequantize(quantize_q8(t));
    std::vector<float> back(W.size());
    for (std::size_t o = 0; o < out; ++o)
        for (std::size_t i = 0; i < in; ++i) back[o * in + i] = d.at(i, o);
    return back;
}

// A q8 model computes exactly the fp32 model whose projections and
// classifier were replaced by their int8 round trips; the embedding lookup
// stays fp32, so a tied classifier becomes a separate, rounded one. Covers
// decode and prefill on both matmul paths.
void test_q8(int heads, int kv_heads, bool shared) {
    const RawModel m = make_model(heads, kv_heads, shared, std::uint32_t(heads * 10 + kv_heads));
    const std::string path = temp_path("engine_test_q8.bin");
    write_checkpoint(m, path);
    Transformer model(load_checkpoint(path), WeightType::q8);
    std::filesystem::remove(path);

    const auto D = std::size_t(m.dim), H = std::size_t(m.hidden), KV = std::size_t(m.kv_dim());
    RawModel ref = m;
    for (std::size_t l = 0; l < std::size_t(m.layers); ++l) {
        ref.wq[l] = q8_round_trip(m.wq[l], D, D);
        ref.wk[l] = q8_round_trip(m.wk[l], KV, D);
        ref.wv[l] = q8_round_trip(m.wv[l], KV, D);
        ref.wo[l] = q8_round_trip(m.wo[l], D, D);
        ref.w1[l] = q8_round_trip(m.w1[l], H, D);
        ref.w2[l] = q8_round_trip(m.w2[l], D, H);
        ref.w3[l] = q8_round_trip(m.w3[l], H, D);
    }
    ref.wcls = q8_round_trip(shared ? m.emb : m.wcls, std::size_t(m.vocab), D);
    ref.shared = false;

    std::vector<int> tokens;
    for (int pos = 0; pos < m.seq_len; ++pos) tokens.push_back((pos * 3 + 1) % m.vocab);
    for (std::size_t pos = 0; pos < 3; ++pos) {
        const std::vector<int> prefix(tokens.begin(), tokens.begin() + long(pos + 1));
        CHECK(max_abs_diff(model.forward(tokens[pos], pos), reference_logits(ref, prefix)) <=
              kTol);
    }
    CHECK(max_abs_diff(model.prefill(tokens, 0), reference_logits(ref, tokens)) <= kTol);
}

// Rewinding overwrites the cache from that position on.
void test_rewind() {
    const RawModel m = make_model(4, 2, true, 7);
    const std::string path = temp_path("engine_test_rewind.bin");
    write_checkpoint(m, path);
    Transformer model(load_checkpoint(path));
    std::filesystem::remove(path);

    for (int pos = 0; pos < 5; ++pos) model.forward(pos + 1, std::size_t(pos));
    std::vector<int> tokens = {1, 2, 20, 21};
    model.forward(20, 2);
    const auto& logits = model.forward(21, 3);
    CHECK(max_abs_diff(logits, reference_logits(m, tokens)) <= kTol);
}

template <class E, class F>
std::string error_of(F&& f) {
    try {
        f();
    } catch (const E& e) {
        return e.what();
    }
    return "";
}

void test_bad_inputs() {
    const RawModel m = make_model(4, 4, true, 1);
    const std::string path = temp_path("engine_test_bad.bin");

    write_checkpoint(m, path, 4);
    const std::string short_msg = error_of<std::runtime_error>([&] { load_checkpoint(path); });
    CHECK(short_msg.find("header implies") != std::string::npos);

    RawModel bad = m;
    bad.kv_heads = 3;  // 4 query heads don't split into groups of 3
    write_checkpoint(bad, path);
    CHECK(!error_of<std::runtime_error>([&] { load_checkpoint(path); }).empty());

    CHECK(!error_of<std::runtime_error>([&] { load_checkpoint(path + ".missing"); }).empty());

    write_checkpoint(m, path);
    Transformer model(load_checkpoint(path));
    std::filesystem::remove(path);
    CHECK(!error_of<std::out_of_range>([&] { model.forward(m.vocab, 0); }).empty());
    CHECK(!error_of<std::out_of_range>([&] { model.forward(-1, 0); }).empty());
    CHECK(!error_of<std::out_of_range>([&] { model.forward(0, 1); }).empty());  // skips pos 0
    for (int pos = 0; pos < m.seq_len; ++pos) model.forward(0, std::size_t(pos));
    CHECK(!error_of<std::out_of_range>([&] { model.forward(0, std::size_t(m.seq_len)); }).empty());
    CHECK(!error_of<std::invalid_argument>([&] { model.prefill({}, 0); }).empty());
    CHECK(!error_of<std::out_of_range>([&] { model.prefill({0, 0}, std::size_t(m.seq_len) - 1); })
               .empty());
    CHECK(!error_of<std::out_of_range>([&] { model.prefill({0, m.vocab}, 0); }).empty());
}

} // namespace

int main() {
    test_matches_reference(4, 4, true);   // multi-head, tied classifier
    test_matches_reference(4, 2, false);  // grouped-query, separate classifier
    test_matches_reference(4, 1, true);   // multi-query
    test_prefill(4, 4, true);
    test_prefill(4, 2, false);
    test_prefill(4, 1, true);
    test_q8(4, 2, false);
    test_q8(4, 1, true);
    test_rewind();
    test_bad_inputs();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
}
