#pragma once
// L03 RMSNorm and SwiGLU.
// Lesson page: README.md. Python reference: rmsnorm_swiglu.py. Test: rmsnorm_swiglu_test.cpp.

#include <torch/torch.h>

namespace llm {

/// y = weight * x / sqrt(mean(x^2) + eps), over the last dimension.
struct RMSNormImpl : torch::nn::Module {
    /// Registers the parameter "weight" [dim], all ones.
    RMSNormImpl(int64_t dim, double eps = 1e-6);

    /// x: [..., dim] -> [..., dim]. Computes in float32 (x.to(torch::kFloat32)) and returns
    /// the dtype of x, so that the norm stays exact under bfloat16 autocast.
    torch::Tensor forward(const torch::Tensor& x);

    double eps;
    torch::Tensor weight;
};
TORCH_MODULE(RMSNorm);

/// SwiGLU feed-forward: down_proj(silu(gate_proj(x)) * up_proj(x)). No bias.
struct FeedForwardImpl : torch::nn::Module {
    /// Registers three linear layers without bias, with these names:
    ///   "gate_proj": dim -> hidden, "up_proj": dim -> hidden, "down_proj": hidden -> dim.
    FeedForwardImpl(int64_t dim, int64_t hidden);

    /// x: [..., dim] -> [..., dim].
    torch::Tensor forward(const torch::Tensor& x);

    torch::nn::Linear gate_proj{nullptr}, up_proj{nullptr}, down_proj{nullptr};
};
TORCH_MODULE(FeedForward);

}  // namespace llm
