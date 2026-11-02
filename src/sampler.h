#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Picks the next token from logits. temperature == 0 is greedy (argmax).
// Otherwise logits are divided by the temperature and softmaxed; with
// 0 < top_p < 1 the draw is limited to the smallest set of most likely tokens
// whose probability adds up to more than top_p (nucleus sampling).
//
// The RNG (xorshift64*) and the top-p cut follow llama2.c, so a seed draws
// the same random numbers there and here.
class Sampler {
public:
    Sampler(float temperature, float top_p, std::uint64_t seed);

    // Overwrites logits with probabilities unless sampling greedily.
    int sample(std::vector<float>& logits);

private:
    std::uint32_t next_u32();
    float next_unit();  // uniform in [0, 1)

    float temperature_;
    float top_p_;
    std::uint64_t state_;
    std::vector<std::pair<float, int>> candidates_;  // top-p scratch
};
