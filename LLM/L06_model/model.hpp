#pragma once
// L06 Block, model, loss.
// Lesson page: README.md. Python reference: model.py. Test: model_test.cpp.

#include <torch/torch.h>

#include <filesystem>
#include <vector>

#include "L05_attention/attention.hpp"

namespace llm {

/// Sizes of the model. The defaults give the model of block 1: 63 912 192 parameters.
struct Config {
    int64_t vocab_size = 6400;
    int64_t dim = 768;
    int64_t n_layers = 8;
    int64_t n_heads = 8;
    int64_t n_kv_heads = 4;
    int64_t head_dim = 96;
    int64_t hidden = 2432;  ///< feed-forward width: ceil(768 * pi / 64) * 64
    double eps = 1e-6;
    double rope_base = 1e6;
    int64_t max_len = 2048;  ///< rows of the RoPE tables: the longest sequence
};

/// One transformer block: x + attention(norm(x)), then x + feed-forward(norm(x)).
struct BlockImpl : torch::nn::Module {
    /// Registers "input_layernorm" (RMSNorm of dim), "self_attn" (Attention),
    /// "post_attention_layernorm" (RMSNorm of dim), "mlp" (FeedForward dim -> hidden).
    explicit BlockImpl(const Config& c);

    /// x: [B, T, dim]; cos, sin: [T, head_dim]; cache: for self_attn (lesson L07) -> [B, T, dim].
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin,
                          LayerCache* cache = nullptr);

    RMSNorm input_layernorm{nullptr};
    Attention self_attn{nullptr};
    RMSNorm post_attention_layernorm{nullptr};
    FeedForward mlp{nullptr};
};
TORCH_MODULE(Block);

/// The language model: embedding -> n_layers blocks -> RMSNorm -> logits.
struct ModelImpl : torch::nn::Module {
    /// Registers "embed_tokens" (Embedding vocab_size x dim), "layers" (a ModuleList of
    /// n_layers Blocks) and "norm" (RMSNorm of dim). Registers the RoPE tables
    /// (rope_tables(head_dim, max_len, rope_base)) as the buffers "cos" and "sin":
    /// buffers move with ->to(device), and they are not in the checkpoint.
    explicit ModelImpl(const Config& c);

    /// ids: int64 [B, T] -> logits [B, T, vocab_size].
    ///   The positions start at 0, or at the length of the cache (cache->at(0).k.size(2))
    ///   when the cache is not empty (lesson L07). cos, sin: rows start .. start+T-1.
    ///   The head uses the embedding weight (tied):
    ///   torch::nn::functional::linear(norm(x), embed_tokens->weight). No head parameter.
    torch::Tensor forward(const torch::Tensor& ids, std::vector<LayerCache>* cache = nullptr);

    Config config;
    torch::nn::Embedding embed_tokens{nullptr};
    torch::nn::ModuleList layers{nullptr};  ///< layers[i]->as<BlockImpl>() gives block i
    RMSNorm norm{nullptr};
    torch::Tensor cos, sin;
};
TORCH_MODULE(Model);

/// Next-token cross-entropy, a scalar: logits [B, T, V] at positions 0 .. T-2 against
/// labels [B, T] at positions 1 .. T-1. Labels -100 do not count (mean over the others).
torch::Tensor lm_loss(const torch::Tensor& logits, const torch::Tensor& labels);

/// Each parameter with 2 dimensions (embedding, projections) from N(0, std^2).
/// The RMSNorm weights stay 1. (torch::nn::init::normal_, under torch::NoGradGuard.)
void init_weights(Model& model, double std = 0.02);

/// Writes the parameters as a dictionary: parameter name -> tensor. Python reads it with
/// course.py.io.load. (course/golden.hpp has read_tensors and write_tensors.)
void save_checkpoint(const Model& model, const std::filesystem::path& file);

/// Reads a checkpoint (of save_checkpoint or of the Python reference) into the parameters,
/// by name. Throws std::runtime_error when a name is missing or extra, or a shape differs.
void load_checkpoint(Model& model, const std::filesystem::path& file);

}  // namespace llm
