// Smoke test of milestone M0: LibTorch runs on the CPU and on the graphics
// processor, and tensors go from Python to C++ and from C++ to Python.
// The Python side is course/py/smoke.py.

#include <ATen/autocast_mode.h>

#include "course/testing.hpp"

namespace {
const std::filesystem::path kDir = course::golden_dir("smoke");

torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& w) {
    return w * (x * torch::rsqrt(x.pow(2).mean(-1, /*keepdim=*/true) + 1e-6));
}
}  // namespace

TEST_CASE("CPU: C++ reads the file of Python and gets the same numbers") {
    course::Golden g(kDir / "py_to_cpp.pt");
    torch::Tensor y = rms_norm(g.tensor("x"), g.tensor("weight"));
    CHECK_CLOSE(y, g.tensor("rmsnorm"), 1e-6);

    torch::Tensor q = g.tensor("q");
    torch::Tensor att = torch::scaled_dot_product_attention(q, q, q, {}, 0.0, /*is_causal=*/true);
    CHECK_CLOSE(att, g.tensor("attention"), 1e-6);

    // C++ to Python: course/py/smoke.py reads this file (CTest test smoke_py).
    course::write_tensors(kDir / "cpp_to_py.pt",
                          {{"rmsnorm", y}, {"attention", att}, {"step", torch::tensor(int64_t{1234})}});
}

TEST_CASE("CPU: one AdamW step with gradient clipping changes the weights") {
    torch::manual_seed(0);
    torch::nn::Linear layer(8, 4);
    torch::optim::AdamW opt(layer->parameters(), torch::optim::AdamWOptions(1e-2));
    torch::Tensor before = layer->weight.detach().clone();
    torch::Tensor loss = layer->forward(torch::randn({16, 8})).pow(2).mean();
    loss.backward();
    double norm = torch::nn::utils::clip_grad_norm_(layer->parameters(), 1.0);
    opt.step();
    CHECK(norm > 0.0);
    CHECK_FALSE(torch::equal(before, layer->weight.detach()));
}

TEST_CASE("GPU: attention on the graphics processor agrees with the CPU") {
    REQUIRE_MESSAGE(torch::cuda::is_available(),
                    "LibTorch sees no CUDA device. Check the NVIDIA driver (580 or above) and CUDA_VISIBLE_DEVICES.");
    const torch::Device gpu(torch::kCUDA, 0);
    course::Golden g(kDir / "py_to_cpp.pt");
    torch::Tensor q = g.tensor("q").to(gpu);
    torch::Tensor att = torch::scaled_dot_product_attention(q, q, q, {}, 0.0, /*is_causal=*/true);
    CHECK(att.is_cuda());
    CHECK_CLOSE(att.cpu(), g.tensor("attention"), 1e-5);
}

TEST_CASE("GPU: autocast gives bfloat16 for a linear layer") {
    REQUIRE(torch::cuda::is_available());
    const torch::Device gpu(torch::kCUDA, 0);
    torch::nn::Linear layer(8, 8);
    layer->to(gpu);
    at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    torch::Tensor out = layer->forward(torch::randn({4, 8}, gpu));
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    at::autocast::clear_cache();
    CHECK((out.scalar_type() == torch::kBFloat16));  // double parentheses: doctest cannot print a ScalarType
}
