// Generates text from a llama2.c checkpoint.
//
//   generate <checkpoint.bin> [options]
//     -z <path>   tokenizer (default: tokenizer.bin next to the checkpoint)
//     -i <text>   prompt (default: empty, i.e. start from BOS)
//     -n <int>    total positions to run, prompt included (default: seq_len)
//     -t <float>  temperature, 0 = greedy (default 1.0)
//     -p <float>  top-p (default 0.9)
//     -s <int>    RNG seed (default: time)
//     -w <type>   weight storage: f32 or q8 (int8, default f32)
//     -d <device> cpu or gpu (default cpu; gpu needs an ENGINE_CUDA build)
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
#include <utility>
#include <vector>

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#ifdef ENGINE_HAVE_CUDA
#include "cuda_model.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: generate <checkpoint.bin> [-z tokenizer] [-i prompt] [-n steps]\n"
                 "                [-t temperature] [-p top_p] [-s seed] [-w f32|q8]\n"
                 "                [-d cpu|gpu]\n");
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

// Prefills the prompt, then samples until steps positions or BOS. Model is
// Transformer or CudaTransformer.
template <class Model>
void generate(Model& model, const Tokenizer& tokenizer, Sampler& sampler,
              const std::vector<int>& prompt_tokens, std::size_t steps,
              Clock::time_point t_load) {
    // The whole prompt goes through in one prefill pass; generated tokens are
    // then fed one at a time.
    const auto t_start = Clock::now();
    const std::vector<float>* logits = &model.prefill(prompt_tokens, 0);
    const auto t_prefill = Clock::now();
    for (std::size_t i = 0; i + 1 < prompt_tokens.size(); ++i)
        print_piece(tokenizer.decode(prompt_tokens[i], prompt_tokens[i + 1]));

    Clock::time_point t_first{};
    int token = prompt_tokens.back();
    std::size_t pos = prompt_tokens.size(), generated = 0;
    for (;;) {
        std::vector<float> probs = *logits;
        const int next = sampler.sample(probs);
        if (generated++ == 0) t_first = Clock::now();
        if (next == Tokenizer::kBos) break;
        print_piece(tokenizer.decode(token, next));
        std::fflush(stdout);
        token = next;
        if (pos >= steps) break;
        logits = &model.forward(token, pos++);
    }
    const auto t_end = Clock::now();
    std::printf("\n");

    // The first sampled token closes the prompt; decode speed is measured
    // over the tokens after it.
    const double prefill_ms = ms_between(t_start, t_prefill);
    std::fprintf(stderr, "load %.0f ms | prompt %zu tok, prefill %.1f ms (%.0f tok/s), "
                 "first token %.1f ms",
                 ms_between(t_load, t_start), prompt_tokens.size(), prefill_ms,
                 1e3 * double(prompt_tokens.size()) / prefill_ms, ms_between(t_start, t_first));
    if (generated > 1) {
        const double ms = ms_between(t_first, t_end) / double(generated - 1);
        std::fprintf(stderr, " | decode %zu tok, %.3f ms/tok, %.0f tok/s", generated - 1, ms,
                     1e3 / ms);
    }
    std::fprintf(stderr, "\n");
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) usage();
    const std::string ckpt_path = argv[1];
    std::string tok_path, prompt;
    std::size_t steps = 0;
    float temperature = 1.0f, top_p = 0.9f;
    auto seed = static_cast<std::uint64_t>(std::time(nullptr));
    WeightType weights = WeightType::f32;
    bool gpu = false;

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
            case 'w':
                if (std::string(val) == "q8") {
                    weights = WeightType::q8;
                } else if (std::string(val) != "f32") {
                    usage();
                }
                break;
            case 'd':
                if (std::string(val) == "gpu") {
                    gpu = true;
                } else if (std::string(val) != "cpu") {
                    usage();
                }
                break;
            default: usage();
        }
    }
    if (tok_path.empty())
        tok_path = (std::filesystem::path(ckpt_path).parent_path() / "tokenizer.bin").string();

#ifndef ENGINE_HAVE_CUDA
    if (gpu) {
        std::fprintf(stderr, "-d gpu needs a build configured with -DENGINE_CUDA=ON\n");
        return 2;
    }
#endif

    const auto t_load = Clock::now();
    Checkpoint ckpt = load_checkpoint(ckpt_path);
    const Config cfg = ckpt.config;
    const Tokenizer tokenizer(tok_path, cfg.vocab_size);
    Sampler sampler(temperature, top_p, seed);
    if (steps == 0 || steps > cfg.seq_len) steps = cfg.seq_len;

    const std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
    if (prompt_tokens.size() > steps) {
        std::fprintf(stderr, "prompt is %zu tokens, more than %zu steps\n",
                     prompt_tokens.size(), steps);
        return 1;
    }

#ifdef ENGINE_HAVE_CUDA
    if (gpu) {
        CudaTransformer model(ckpt, weights);
        generate(model, tokenizer, sampler, prompt_tokens, steps, t_load);
        return 0;
    }
#endif
    Transformer model(std::move(ckpt), weights);
    generate(model, tokenizer, sampler, prompt_tokens, steps, t_load);
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
