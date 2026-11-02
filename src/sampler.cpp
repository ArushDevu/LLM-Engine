#include "sampler.h"

#include <algorithm>
#include <stdexcept>

#include "ops.h"

Sampler::Sampler(float temperature, float top_p, std::uint64_t seed)
    : temperature_(temperature), top_p_(top_p), state_(seed) {
    if (temperature < 0.0f) throw std::invalid_argument("sampler: negative temperature");
    // xorshift has a fixed point at zero.
    if (state_ == 0) state_ = 0x9E3779B97F4A7C15ull;
}

// xorshift64*, as in llama2.c.
std::uint32_t Sampler::next_u32() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return static_cast<std::uint32_t>((state_ * 0x2545F4914F6CDD1Dull) >> 32);
}

float Sampler::next_unit() { return static_cast<float>(next_u32() >> 8) / 16777216.0f; }

int Sampler::sample(std::vector<float>& logits) {
    if (logits.empty()) throw std::invalid_argument("sampler: empty logits");
    if (logits.size() == 1) return 0;
    if (temperature_ == 0.0f)
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());

    for (float& l : logits) l /= temperature_;
    softmax(logits.data(), logits.size());
    const float coin = next_unit();

    if (top_p_ <= 0.0f || top_p_ >= 1.0f) {
        float cdf = 0.0f;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            cdf += logits[i];
            if (coin < cdf) return static_cast<int>(i);
        }
        return static_cast<int>(logits.size() - 1);  // rounding left cdf < 1
    }

    // A token below this can't be in the nucleus: if every survivor were this
    // small, the n - 1 others would still leave more than top_p. Dropping
    // them first keeps the sort to a few hundred entries instead of the vocab.
    const float cutoff = (1.0f - top_p_) / static_cast<float>(logits.size() - 1);
    candidates_.clear();
    for (std::size_t i = 0; i < logits.size(); ++i)
        if (logits[i] >= cutoff) candidates_.emplace_back(logits[i], static_cast<int>(i));
    std::sort(candidates_.begin(), candidates_.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::size_t last = candidates_.size() - 1;
    float mass = 0.0f;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
        mass += candidates_[i].first;
        if (mass > top_p_) {
            last = i;
            break;
        }
    }
    const float r = coin * mass;
    float cdf = 0.0f;
    for (std::size_t i = 0; i <= last; ++i) {
        cdf += candidates_[i].first;
        if (r < cdf) return candidates_[i].second;
    }
    return candidates_[last].second;
}
