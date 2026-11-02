#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// SentencePiece-style BPE over a llama2.c tokenizer.bin vocabulary. Text is
// split into UTF-8 code points, each mapped to its piece or, if the vocabulary
// lacks it, to one byte-fallback token per byte; then the adjacent pair whose
// merge has the highest score is merged until no merge applies.
class Tokenizer {
public:
    static constexpr int kBos = 1;
    static constexpr int kEos = 2;

    // File layout: int32 max piece length, then per token a float32 merge
    // score, an int32 byte length and the piece's bytes. vocab_size comes from
    // the model, since the file doesn't store it. Throws std::runtime_error if
    // the file is short or unreadable.
    Tokenizer(const std::string& path, std::size_t vocab_size);

    // Like SentencePiece, non-empty text gets a leading space ("dummy
    // prefix"), so a word at the start encodes the same as mid-sentence.
    std::vector<int> encode(const std::string& text, bool bos, bool eos) const;

    // Text of token as it follows prev. Byte-fallback tokens decode to their
    // raw byte, and the leading space of the first piece after BOS is dropped.
    std::string decode(int prev, int token) const;

    std::size_t vocab_size() const { return pieces_.size(); }

private:
    int lookup(const std::string& piece) const;

    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int> ids_;
};
