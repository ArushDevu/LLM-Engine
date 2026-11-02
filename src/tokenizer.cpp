#include "tokenizer.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {

// Byte-fallback tokens <0x00>..<0xFF> follow <unk>, <s>, </s>.
constexpr int kByteBase = 3;

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Byte value of a "<0xHH>" piece, or -1 if piece isn't one.
int byte_piece(const std::string& piece) {
    if (piece.size() != 6 || piece.compare(0, 3, "<0x") != 0 || piece[5] != '>') return -1;
    const int hi = hex_digit(piece[3]), lo = hex_digit(piece[4]);
    return hi < 0 || lo < 0 ? -1 : hi * 16 + lo;
}

bool is_continuation(unsigned char c) { return (c & 0xC0) == 0x80; }

} // namespace

Tokenizer::Tokenizer(const std::string& path, std::size_t vocab_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open tokenizer " + path);
    auto read = [&](void* dst, std::size_t n) {
        in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("tokenizer " + path + " ended early");
    };
    std::int32_t max_len = 0;
    read(&max_len, sizeof max_len);

    pieces_.resize(vocab_size);
    scores_.resize(vocab_size);
    ids_.reserve(vocab_size);
    for (std::size_t i = 0; i < vocab_size; ++i) {
        std::int32_t len = 0;
        read(&scores_[i], sizeof(float));
        read(&len, sizeof len);
        if (len < 0 || len > max_len)
            throw std::runtime_error("tokenizer " + path + ": bad piece length " +
                                     std::to_string(len) + " at token " + std::to_string(i));
        pieces_[i].resize(static_cast<std::size_t>(len));
        read(pieces_[i].data(), pieces_[i].size());
        // First id wins if a piece repeats.
        ids_.emplace(pieces_[i], static_cast<int>(i));
    }
}

int Tokenizer::lookup(const std::string& piece) const {
    const auto it = ids_.find(piece);
    return it == ids_.end() ? -1 : it->second;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) const {
    std::vector<int> tokens;
    if (bos) tokens.push_back(kBos);
    if (!text.empty()) {
        if (const int space = lookup(" "); space >= 0) tokens.push_back(space);
    }

    for (std::size_t i = 0; i < text.size();) {
        std::size_t end = i + 1;
        while (end < text.size() && end - i < 4 &&
               is_continuation(static_cast<unsigned char>(text[end])))
            ++end;
        const std::string cp = text.substr(i, end - i);
        if (const int id = lookup(cp); id >= 0) {
            tokens.push_back(id);
        } else {
            for (unsigned char c : cp) tokens.push_back(kByteBase + c);
        }
        i = end;
    }

    // Greedy merges, best score first. O(n^2) in the prompt length, which is
    // nothing next to running the model over the same tokens.
    for (;;) {
        float best_score = 0.0f;
        int best_id = -1;
        std::size_t best_at = 0;
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            const auto a = static_cast<std::size_t>(tokens[i]);
            const auto b = static_cast<std::size_t>(tokens[i + 1]);
            const int id = lookup(pieces_[a] + pieces_[b]);
            if (id >= 0 && (best_id < 0 || scores_[std::size_t(id)] > best_score)) {
                best_score = scores_[std::size_t(id)];
                best_id = id;
                best_at = i;
            }
        }
        if (best_id < 0) break;
        tokens[best_at] = best_id;
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_at) + 1);
    }

    if (eos) tokens.push_back(kEos);
    return tokens;
}

std::string Tokenizer::decode(int prev, int token) const {
    if (token < 0 || static_cast<std::size_t>(token) >= pieces_.size())
        throw std::out_of_range("decode: token " + std::to_string(token) + " outside vocab");
    const std::string& piece = pieces_[static_cast<std::size_t>(token)];
    if (const int byte = byte_piece(piece); byte >= 0)
        return std::string(1, static_cast<char>(byte));
    if (prev == kBos && !piece.empty() && piece[0] == ' ') return piece.substr(1);
    return piece;
}
