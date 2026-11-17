#pragma once
// L04 RoPE: rotary position embedding.
// Lesson page: README.md. Python reference: rope.py. Test: rope_test.cpp.

#include <torch/torch.h>

namespace llm {

/// cos and sin of the RoPE angles. Both float32 [max_len, head_dim].
struct RopeTables {
    torch::Tensor cos;
    torch::Tensor sin;
};

/// Tables for positions 0 .. max_len-1. With i = 0 .. head_dim/2 - 1:
///   inv_freq[i]  = 1 / base^(2i / head_dim)      (torch::pow(base, exponent), float32)
///   angle[p][i]  = p * inv_freq[i]                (torch::outer of positions and inv_freq)
///   angle row    = [angle[p], angle[p]]           (two copies, joined: head_dim values)
/// cos = cos(angle), sin = sin(angle).
RopeTables rope_tables(int64_t head_dim, int64_t max_len, double base);

/// (x1, x2) -> (-x2, x1), with x1 and x2 the two halves of the last dimension. Any shape.
torch::Tensor rotate_half(const torch::Tensor& x);

/// x: [B, H, T, head_dim]; cos, sin: [T, head_dim], the rows of the positions of x.
/// Returns x * cos + rotate_half(x) * sin, shape of x.
torch::Tensor apply_rope(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin);

}  // namespace llm
