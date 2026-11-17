// Test of L03. The golden file comes from golden.py (Python reference).
// RMSNorm limit 2e-6: its outputs reach 4, where 1e-6 is only two float32 steps, and two
// correct formulas (x * rsqrt(m) or x / sqrt(m)) differ by one step.

#include "L03_rmsnorm_swiglu/rmsnorm_swiglu.hpp"
#include "course/testing.hpp"

using namespace llm;

namespace {
const course::Golden& golden() {
    static const course::Golden g(course::golden_dir("LLM/L03_rmsnorm_swiglu") / "golden.pt");
    return g;
}
}  // namespace

TEST_CASE("RMSNorm has one parameter, weight [dim], all ones") {
    RMSNorm norm(64);
    CHECK(course::parameter_names(*norm) == std::vector<std::string>{"weight"});
    CHECK_TENSOR_EQUAL(norm->weight.detach(), torch::ones({64}));
    CHECK(norm->eps == doctest::Approx(1e-6));
}

TEST_CASE("RMSNorm gives the output of the reference") {
    const auto& g = golden();
    RMSNorm norm(g.integer("dim"), g.real("eps"));
    course::copy_parameters(*norm, g.tensors("norm_weights"));
    CHECK_CLOSE(norm->forward(g.tensor("x")), g.tensor("norm_out"), 2e-6);
}

TEST_CASE("RMSNorm normalizes over the last dimension only (4-D input of L05)") {
    const auto& g = golden();
    RMSNorm norm(g.integer("head_dim"), g.real("eps"));
    course::copy_parameters(*norm, g.tensors("norm4_weights"));
    CHECK_CLOSE(norm->forward(g.tensor("x4")), g.tensor("norm4_out"), 2e-6);
}

TEST_CASE("RMSNorm returns the dtype of its input") {
    RMSNorm norm(64);
    torch::Tensor x = torch::randn({2, 3, 64}).to(torch::kBFloat16);
    CHECK((norm->forward(x).scalar_type() == torch::kBFloat16));
}

TEST_CASE("FeedForward has the three projections of the reference") {
    const auto& g = golden();
    FeedForward ffn(g.integer("dim"), g.integer("hidden"));
    auto names = course::parameter_names(*ffn);
    std::sort(names.begin(), names.end());
    CHECK(names == std::vector<std::string>{"down_proj.weight", "gate_proj.weight", "up_proj.weight"});
    CHECK(ffn->gate_proj->weight.sizes() == torch::IntArrayRef{g.integer("hidden"), g.integer("dim")});
}

TEST_CASE("FeedForward gives the output of the reference") {
    const auto& g = golden();
    FeedForward ffn(g.integer("dim"), g.integer("hidden"));
    course::copy_parameters(*ffn, g.tensors("ffn_weights"));
    CHECK_CLOSE(ffn->forward(g.tensor("x")), g.tensor("ffn_out"), 1e-6);
}
