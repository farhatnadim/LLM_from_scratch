#pragma once
// L02 Token data: token files, pre-training samples, batches.
// Lesson page: README.md. Python reference: token_data.py. Test: token_data_test.cpp.
//
// tokens.bin   all token ids of all samples, uint16, little-endian
// offsets.bin  start of each sample in tokens.bin, uint64, little-endian; one more entry
//              at the end: the total count of tokens. Sample i is tokens[offsets[i], offsets[i+1]).

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <vector>

#include "L01_tokenizer/tokenizer.hpp"

namespace llm {

inline constexpr int64_t kPad = kEndOfText;  ///< id of the padding
inline constexpr int64_t kIgnore = -100;     ///< label that the loss ignores

/// Writes tokens.bin and offsets.bin of `samples` into `folder` (made when missing).
/// An empty sample is allowed: two equal offsets.
void write_token_files(const std::filesystem::path& folder, const std::vector<std::vector<int>>& samples);

/// Encodes the field "text" of each line of a JSON Lines file with `tok`, then writes the
/// token files into `folder`. Returns the count of samples. JSON: nlohmann/json
/// (#include <nlohmann/json.hpp>; nlohmann::json::parse(line)["text"].get<std::string>()).
int64_t tokenize_jsonl(const std::filesystem::path& jsonl, const Tokenizer& tok,
                       const std::filesystem::path& folder);

/// The two token files of a folder, in memory.
struct TokenFile {
    /// Reads tokens.bin and offsets.bin. Throws std::runtime_error when a file is missing.
    explicit TokenFile(const std::filesystem::path& folder);

    int64_t size() const;                     ///< count of samples: offsets.size() - 1
    std::vector<int> sample(int64_t i) const; ///< ids of sample i

    std::vector<uint16_t> tokens;
    std::vector<uint64_t> offsets;
};

/// One pre-training sample. Both tensors are int64 [max_len].
struct Sample {
    torch::Tensor input_ids;
    torch::Tensor labels;
};

/// [kImStart] + ids + [kImEnd]: ids is cut to max_len - 2 ids first. Then kPad up to max_len.
/// labels: input_ids, with kIgnore at the padding positions.
Sample pretrain_sample(const std::vector<int>& ids, int64_t max_len);

/// A batch. Both tensors are int64 [B, max_len].
struct Batch {
    torch::Tensor input_ids;
    torch::Tensor labels;
};

/// The pre-training samples of data at `indices` (B = indices.size()), stacked in that order.
Batch make_batch(const TokenFile& data, const std::vector<int64_t>& indices, int64_t max_len);

}  // namespace llm
