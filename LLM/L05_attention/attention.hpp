#pragma once
// L05 Attention: causal grouped-query attention.
// Lesson page: README.md. Python reference: attention.py. Test: attention_test.cpp.
//
// Shapes: B batch, T queries, S keys (S >= T), H query heads, KV key-value heads, D head size.
// Query head h reads key-value head h / (H / KV).

#include <torch/torch.h>

#include "L03_rmsnorm_swiglu/rmsnorm_swiglu.hpp"
#include "L04_rope/rope.hpp"

namespace llm {

/// [T, S] bool: true where query t can see key s, that is s <= S - T + t.
/// (torch::ones({T, S}, torch::kBool).tril(S - T))
torch::Tensor causal_mask(int64_t T, int64_t S, torch::Device device = torch::kCPU);

/// Attention written out. q: [B, H, T, D]; k, v: [B, KV, S, D] -> [B, H, T, D].
///   1. Repeat each key-value head H / KV times: k.repeat_interleave(H / KV, 1). Same for v.
///   2. scores = q k^T / sqrt(D)                        [B, H, T, S]
///   3. -infinity where causal_mask(T, S) is false.
///   4. softmax over the keys, then times v.
torch::Tensor attend_manual(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);

/// Same result as attend_manual, with torch::scaled_dot_product_attention and enable_gqa = true.
/// T == S: is_causal = true (the fast path for training). T < S: attn_mask = causal_mask(T, S).
torch::Tensor attend_sdpa(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);

/// Keys and values of the earlier positions, for one layer (lesson L07).
/// Both tensors are undefined while the cache is empty; then [B, KV, S, D].
struct LayerCache {
    torch::Tensor k;
    torch::Tensor v;
};

struct AttentionImpl : torch::nn::Module {
    /// Registers, all linear layers without bias:
    ///   "q_proj": dim -> n_heads * head_dim       "k_proj": dim -> n_kv_heads * head_dim
    ///   "v_proj": dim -> n_kv_heads * head_dim    "o_proj": n_heads * head_dim -> dim
    ///   "q_norm", "k_norm": RMSNorm(head_dim, eps)
    AttentionImpl(int64_t dim, int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, double eps = 1e-6);

    /// x: [B, T, dim]; cos, sin: [T, D], the rows of the positions of x -> [B, T, dim].
    ///   q, k, v: project, view as [B, T, heads, D], transpose to [B, heads, T, D]
    ///   q = apply_rope(q_norm(q)), k = apply_rope(k_norm(k))
    ///   cache (lesson L07; nullptr before): if cache->k is defined, put k after cache->k on
    ///     dimension 2 (torch::cat), same for v. Then store k and v in the cache.
    ///   y = attend_sdpa(q, k, v), transpose to [B, T, H, D], reshape to [B, T, H * D], o_proj.
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin,
                          LayerCache* cache = nullptr);

    int64_t n_heads, n_kv_heads, head_dim;
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, o_proj{nullptr};
    RMSNorm q_norm{nullptr}, k_norm{nullptr};
};
TORCH_MODULE(Attention);

}  // namespace llm
