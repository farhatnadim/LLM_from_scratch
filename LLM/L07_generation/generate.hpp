#pragma once
// L07 Generation and KV cache.
// Lesson page: README.md. Python reference: generate.py. Test: generate_test.cpp.
//
// The KV cache is also the cache path of Attention::forward (L05) and the position
// offset of Model::forward (L06). Write them in this lesson.

#include <torch/torch.h>

#include <optional>

#include "L06_model/model.hpp"

namespace llm {

/// logits [B, V] -> probabilities [B, V], after three steps:
///   1. logits / temperature
///   2. top_k > 0: -infinity for each logit below the k-th largest of its row (ties stay).
///   3. top_p < 1: sort each row (largest first) and take p = softmax of the sorted logits.
///      Drop each token whose mass before it, cumsum(p) - p, is more than top_p: -infinity.
///      (torch::sort, then Tensor::scatter puts the sorted values back in place.)
///   Then softmax. top_k = 0 and top_p = 1 turn a filter off.
torch::Tensor filter_probs(const torch::Tensor& logits, double temperature, int64_t top_k, double top_p);

struct GenerateOptions {
    int64_t max_new = 64;       ///< largest count of new ids
    double temperature = 0.0;   ///< 0: greedy (argmax); above 0: sample from filter_probs
    int64_t top_k = 0;
    double top_p = 1.0;
    int64_t eos = -1;           ///< stop id; -1: none
};

/// ids: prompt, int64 [B, T] -> new ids, int64 [B, n] with n <= max_new. No gradients.
///   1. One std::vector<LayerCache> (one entry for each layer). Read the whole prompt with it.
///   2. Each step: the next id from the logits of the last position (greedy, or
///      torch::multinomial(filter_probs(...), 1, false, generator)). Then read only that id,
///      with the cache.
///   3. A row that wrote eos writes eos after it. Stop when all rows wrote eos.
/// The prompt and the new ids must fit in the RoPE tables (config.max_len).
torch::Tensor generate(Model& model, const torch::Tensor& ids, const GenerateOptions& options,
                       std::optional<at::Generator> generator = std::nullopt);

}  // namespace llm
