#pragma once
// L01 BPE tokenizer: byte-level byte-pair encoding.
// Lesson page: README.md. Python reference: tokenizer.py. Test: tokenizer_test.cpp.
//
// Ids 0-255 are the bytes. Ids 256-271 are control tokens and ids 272-527 are action
// tokens: together, the special tokens. Ids from 528 are the merges, in the order of
// training. One tokenizer serves all blocks of the course.

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm {

inline constexpr int kFirstSpecial = 256;  ///< first control id
inline constexpr int kFirstAction = 272;   ///< first action id (block 3)
inline constexpr int kFirstMerge = 528;    ///< id of the first merge
inline constexpr int kVocabSize = 6400;    ///< ids of the model of block 1

inline constexpr int kEndOfText = 256;  ///< <|endoftext|>: padding
inline constexpr int kImStart = 257;    ///< <|im_start|>: start of a sample or of a message
inline constexpr int kImEnd = 258;      ///< <|im_end|>: end of a sample or of a message
inline constexpr int kThink = 259;      ///< <think>
inline constexpr int kEndThink = 260;   ///< </think>
inline constexpr int kImagePad = 261;   ///< <|image_pad|>: one image token (block 2)

/// Texts of the 16 control ids 256-271.
inline constexpr std::array<std::string_view, 16> kControlTexts = {
    "<|endoftext|>",  "<|im_start|>",   "<|im_end|>",     "<think>",
    "</think>",       "<|image_pad|>",  "<|reserved_0|>", "<|reserved_1|>",
    "<|reserved_2|>", "<|reserved_3|>", "<|reserved_4|>", "<|reserved_5|>",
    "<|reserved_6|>", "<|reserved_7|>", "<|reserved_8|>", "<|reserved_9|>"};

/// A merge: the ids of the left token and of the right token.
using Merge = std::pair<int, int>;

/// Text of special id `id`, 256 <= id < 528: kControlTexts[id - 256] for a control id,
/// "<|act_k|>" with k = id - 272 for an action id. Throws std::out_of_range for other ids.
std::string special_text(int id);

/// One piece of a text: plain text (special == -1), or one special token (text empty).
struct Piece {
    std::string text;
    int special = -1;
};

/// Cuts the special texts (all 272 of them) out of `text`, from left to right.
/// Returns the pieces in order. A plain piece is never empty.
/// No special text is the start of another, so at one position one special text agrees at most.
/// Example: "a<|im_end|>b" -> {"a", -1}, {"", 258}, {"b", -1}.
std::vector<Piece> split_special(std::string_view text);

/// Pre-tokens of a plain piece (a text with no special text in it). A pre-token is
///   - an optional space ' ' and a run of bytes that are not white space, or
///   - one white-space byte.
/// White space: ' ', '\t', '\n', '\r', '\f', '\v'. All other bytes are not white space,
/// also the bytes of UTF-8 characters. The pre-tokens joined give the text again.
/// Example: "Max  ran\tto" -> "Max", " ", " ran", "\t", "to".
std::vector<std::string> pretokenize(std::string_view text);

/// Learns vocab_size - 528 merges from texts (fewer when no pair is left):
///   1. Count each pre-token of the plain pieces of all texts. Special tokens do not count.
///   2. Write each pre-token as a sequence of byte ids.
///   3. Repeat: count each adjacent pair (a, b), weighted by the count of its pre-token.
///      Take the pair with the largest count. A tie goes to the pair with the larger
///      bytes: compare bytes(a) first, then bytes(b), as unsigned bytes (std::string
///      comparison does this). Give the pair the next id: 528, 529, ...
///      Replace the pair in each pre-token, from left to right.
/// Speed: the plain algorithm is fast enough in C++ (seconds for 6400 ids on TinyStories).
std::vector<Merge> train_bpe(const std::vector<std::string>& texts, int vocab_size);

/// A trained tokenizer. You can add private helper functions in tokenizer.cpp.
struct Tokenizer {
    /// Keeps the merges and builds vocab and rank from them.
    explicit Tokenizer(std::vector<Merge> merges);

    /// Ids of `text`. A special text gives its id. Each plain piece is cut into pre-tokens.
    /// A pre-token starts as its byte ids; then, while an adjacent pair has a rank, merge
    /// all occurrences of the pair of the lowest rank (from left to right).
    /// Uses `cache` for pre-tokens that it already encoded.
    std::vector<int> encode(std::string_view text) const;

    /// The bytes of the ids, joined. A special id gives its text.
    std::string decode(const std::vector<int>& ids) const;

    /// Writes one merge on each line: "<left> <right>\n". Makes the parent folder.
    void save(const std::filesystem::path& file) const;

    /// Reads a file of save(), or of the Python reference.
    static Tokenizer load(const std::filesystem::path& file);

    std::vector<Merge> merges;       ///< in training order: merge i has id 528 + i
    std::vector<std::string> vocab;  ///< bytes of each id; 528 + merges.size() entries
    std::map<Merge, int> rank;       ///< merge -> its index in merges
    mutable std::unordered_map<std::string, std::vector<int>> cache;  ///< pre-token -> ids
};

}  // namespace llm
