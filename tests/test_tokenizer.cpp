#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "sampler.h"
#include "tokenizer.h"

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

// Vocabulary in llama2.c's order: <unk> <s> </s>, 256 byte tokens, then
// pieces with merge scores.
struct Vocab {
    std::vector<std::string> pieces;
    std::vector<float> scores;

    int add(const std::string& piece, float score) {
        pieces.push_back(piece);
        scores.push_back(score);
        return int(pieces.size()) - 1;
    }
};

Vocab base_vocab() {
    Vocab v;
    v.add("<unk>", 0.0f);
    v.add("\n<s>\n", 0.0f);
    v.add("\n</s>\n", 0.0f);
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "<0x%02X>", b);
        v.add(buf, 0.0f);
    }
    return v;
}

std::string write_vocab(const Vocab& v) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "engine_test_tokenizer.bin").string();
    std::ofstream out(path, std::ios::binary);
    std::int32_t max_len = 0;
    for (const auto& p : v.pieces) max_len = std::max(max_len, std::int32_t(p.size()));
    out.write(reinterpret_cast<const char*>(&max_len), sizeof max_len);
    for (std::size_t i = 0; i < v.pieces.size(); ++i) {
        const auto len = std::int32_t(v.pieces[i].size());
        out.write(reinterpret_cast<const char*>(&v.scores[i]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&len), sizeof len);
        out.write(v.pieces[i].data(), len);
    }
    return path;
}

void test_encode_merges_by_score() {
    Vocab v = base_vocab();
    const int sp = v.add(" ", -10.0f);
    const int a = v.add("a", -1.0f);
    v.add("b", -1.0f);
    const int c = v.add("c", -1.0f);
    const int ab = v.add("ab", -3.0f);
    const int bc = v.add("bc", -2.0f);  // beats "ab", so "abc" -> a, bc
    const int sp_a = v.add(" a", -4.0f);
    const std::string path = write_vocab(v);
    const Tokenizer tok(path, v.pieces.size());
    std::filesystem::remove(path);

    CHECK((tok.encode("abc", false, false) == std::vector<int>{sp_a, bc}));
    // "ab" outscores " a", and " ab" isn't a piece.
    CHECK((tok.encode("ab", true, true) ==
           std::vector<int>{Tokenizer::kBos, sp, ab, Tokenizer::kEos}));
    CHECK((tok.encode("ca", false, false) == std::vector<int>{sp, c, a}));
    CHECK(tok.encode("", true, false) == std::vector<int>{Tokenizer::kBos});
}

// Code points missing from the vocabulary fall back to one token per byte,
// and those decode back to the raw bytes.
void test_byte_fallback_round_trip() {
    Vocab v = base_vocab();
    v.add(" ", -10.0f);
    v.add("x", -1.0f);
    const std::string path = write_vocab(v);
    const Tokenizer tok(path, v.pieces.size());
    std::filesystem::remove(path);

    const std::string text = "x\xC3\xA9x";  // "xéx", é = C3 A9
    const std::vector<int> ids = tok.encode(text, false, false);
    CHECK(ids.size() == 5);  // space, x, two bytes, x
    CHECK(ids[2] == 3 + 0xC3 && ids[3] == 3 + 0xA9);
    std::string decoded;
    for (std::size_t i = 0; i < ids.size(); ++i) decoded += tok.decode(i ? ids[i - 1] : -1, ids[i]);
    CHECK(decoded == " " + text);
}

void test_decode_strips_space_after_bos() {
    Vocab v = base_vocab();
    const int once = v.add(" Once", -1.0f);
    const std::string path = write_vocab(v);
    const Tokenizer tok(path, v.pieces.size());
    std::filesystem::remove(path);
    CHECK(tok.decode(Tokenizer::kBos, once) == "Once");
    CHECK(tok.decode(once, once) == " Once");
}

void test_truncated_file() {
    Vocab v = base_vocab();
    const std::string path = write_vocab(v);
    bool threw = false;
    try {
        Tokenizer(path, v.pieces.size() + 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    CHECK(threw);
}

void test_greedy_picks_argmax() {
    Sampler s(0.0f, 0.9f, 1);
    std::vector<float> logits = {0.1f, 3.0f, -2.0f, 2.9f};
    CHECK(s.sample(logits) == 1);
}

// Draw frequencies follow softmax(logits / T).
void test_sampling_matches_distribution() {
    const std::vector<float> logits = {1.0f, 2.0f, 0.0f, 1.5f};
    const float T = 0.8f;
    std::vector<double> want(logits.size());
    double z = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) z += want[i] = std::exp(logits[i] / T);
    for (double& p : want) p /= z;

    Sampler s(T, 1.0f, 42);
    constexpr int kDraws = 200000;
    std::vector<int> counts(logits.size());
    for (int i = 0; i < kDraws; ++i) {
        std::vector<float> l = logits;
        ++counts[std::size_t(s.sample(l))];
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        // 5 standard deviations of a binomial count.
        const double sd = std::sqrt(kDraws * want[i] * (1 - want[i]));
        CHECK(std::fabs(counts[i] - kDraws * want[i]) < 5 * sd);
    }
}

// Probabilities 0.5, 0.3, 0.15, 0.05: top_p = 0.7 keeps the first two
// (0.5 alone isn't more than 0.7), in ratio 5:3.
void test_top_p_truncates() {
    const std::vector<float> probs = {0.15f, 0.5f, 0.05f, 0.3f};
    Sampler s(1.0f, 0.7f, 7);
    std::map<int, int> counts;
    constexpr int kDraws = 100000;
    for (int i = 0; i < kDraws; ++i) {
        std::vector<float> l(probs.size());
        for (std::size_t j = 0; j < probs.size(); ++j) l[j] = std::log(probs[j]);
        ++counts[s.sample(l)];
    }
    CHECK(counts.size() == 2 && counts.count(1) && counts.count(3));
    const double frac = double(counts[1]) / kDraws;
    CHECK(std::fabs(frac - 0.625) < 0.01);
}

void test_seed_reproduces() {
    Sampler a(1.0f, 0.9f, 123), b(1.0f, 0.9f, 123);
    const std::vector<float> logits = {0.3f, 0.1f, 0.5f, 0.2f, 0.4f};
    for (int i = 0; i < 100; ++i) {
        std::vector<float> la = logits, lb = logits;
        CHECK(a.sample(la) == b.sample(lb));
    }
}

} // namespace

int main() {
    test_encode_merges_by_score();
    test_byte_fallback_round_trip();
    test_decode_strips_space_after_bos();
    test_truncated_file();
    test_greedy_picks_argmax();
    test_sampling_matches_distribution();
    test_top_p_truncates();
    test_seed_reproduces();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all tests passed");
}
