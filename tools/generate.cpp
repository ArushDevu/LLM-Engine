// Generates text from a llama2.c checkpoint.
//
//   generate <checkpoint.bin> [options]
//     -z <path>   tokenizer (default: tokenizer.bin next to the checkpoint)
//     -i <text>   prompt (default: empty, i.e. start from BOS)
//     -n <int>    total positions to run, prompt included (default: seq_len)
//     -t <float>  temperature, 0 = greedy (default 1.0)
//     -p <float>  top-p (default 0.9)
//     -s <int>    RNG seed (default: time)
//
// Generation stops at seq_len or when the model emits BOS, which separates
// stories in the training data. Timing goes to stderr so stdout is only text.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: generate <checkpoint.bin> [-z tokenizer] [-i prompt] [-n steps]\n"
                 "                [-t temperature] [-p top_p] [-s seed]\n");
    std::exit(2);
}

// Skips lone control bytes from byte-fallback tokens, which would garble a
// terminal; whitespace and multi-byte pieces pass through.
void print_piece(const std::string& piece) {
    if (piece.size() == 1) {
        const auto c = static_cast<unsigned char>(piece[0]);
        if (c < 0x20 && c != '\n' && c != '\t' && c != '\r') return;
    }
    std::fwrite(piece.data(), 1, piece.size(), stdout);
}

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) usage();
    const std::string ckpt_path = argv[1];
    std::string tok_path, prompt;
    std::size_t steps = 0;
    float temperature = 1.0f, top_p = 0.9f;
    auto seed = static_cast<std::uint64_t>(std::time(nullptr));

    for (int i = 2; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc || flag.size() != 2 || flag[0] != '-') usage();
        const char* val = argv[i + 1];
        switch (flag[1]) {
            case 'z': tok_path = val; break;
            case 'i': prompt = val; break;
            case 'n': steps = std::strtoul(val, nullptr, 10); break;
            case 't': temperature = std::strtof(val, nullptr); break;
            case 'p': top_p = std::strtof(val, nullptr); break;
            case 's': seed = std::strtoull(val, nullptr, 10); break;
            default: usage();
        }
    }
    if (tok_path.empty())
        tok_path = (std::filesystem::path(ckpt_path).parent_path() / "tokenizer.bin").string();

    const auto t_load = Clock::now();
    Transformer model(load_checkpoint(ckpt_path));
    const Config& cfg = model.config();
    const Tokenizer tokenizer(tok_path, cfg.vocab_size);
    Sampler sampler(temperature, top_p, seed);
    if (steps == 0 || steps > cfg.seq_len) steps = cfg.seq_len;

    const std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
    if (prompt_tokens.size() > steps) {
        std::fprintf(stderr, "prompt is %zu tokens, more than %zu steps\n",
                     prompt_tokens.size(), steps);
        return 1;
    }

    // Prompt tokens are fed one at a time through the same path as generated
    // ones; their logits are discarded except the last.
    const auto t_start = Clock::now();
    Clock::time_point t_first{};
    int token = prompt_tokens[0];
    std::size_t pos = 0, generated = 0;
    while (pos < steps) {
        const std::vector<float>& logits = model.forward(token, pos);
        int next;
        if (pos + 1 < prompt_tokens.size()) {
            next = prompt_tokens[pos + 1];
        } else {
            std::vector<float> probs = logits;
            next = sampler.sample(probs);
            if (generated++ == 0) t_first = Clock::now();
        }
        ++pos;
        if (next == Tokenizer::kBos) break;
        print_piece(tokenizer.decode(token, next));
        std::fflush(stdout);
        token = next;
    }
    const auto t_end = Clock::now();
    std::printf("\n");

    // The first sampled token closes the prompt; decode speed is measured
    // over the tokens after it.
    std::fprintf(stderr, "load %.0f ms | prompt %zu tok, first token %.1f ms",
                 ms_between(t_load, t_start), prompt_tokens.size(),
                 ms_between(t_start, t_first));
    if (generated > 1) {
        const double ms = ms_between(t_first, t_end) / double(generated - 1);
        std::fprintf(stderr, " | decode %zu tok, %.3f ms/tok, %.0f tok/s", generated - 1, ms,
                     1e3 / ms);
    }
    std::fprintf(stderr, "\n");
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
