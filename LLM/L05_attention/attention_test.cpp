// Test of L05. The golden file comes from golden.py (Python reference).

#include "L05_attention/attention.hpp"
#include "course/testing.hpp"

using namespace llm;

namespace {
const course::Golden& golden() {
    static const course::Golden g(course::golden_dir("LLM/L05_attention") / "golden.pt");
    return g;
}

Attention reference_attention() {
    const auto& g = golden();
    Attention attn(g.integer("dim"), g.integer("n_heads"), g.integer("n_kv_heads"), g.integer("head_dim"),
                   g.real("eps"));
    course::copy_parameters(*attn, g.tensors("weights"));
    return attn;
}
}  // namespace

TEST_CASE("causal_mask: query t sees keys 0 .. S-T+t") {
    CHECK_TENSOR_EQUAL(causal_mask(3, 7), golden().tensor("mask_3_7"));
    CHECK_TENSOR_EQUAL(causal_mask(4, 4), torch::ones({4, 4}, torch::kBool).tril());
}

TEST_CASE("attend_manual gives the output of the reference, T == S and T < S") {
    const auto& g = golden();
    CHECK_CLOSE(attend_manual(g.tensor("q"), g.tensor("k"), g.tensor("v")), g.tensor("manual_out"), 1e-6);
    CHECK_CLOSE(attend_manual(g.tensor("q3"), g.tensor("k"), g.tensor("v")), g.tensor("manual_out_3"), 1e-6);
}

TEST_CASE("attend_sdpa gives the output of the reference, T == S and T < S") {
    const auto& g = golden();
    CHECK_CLOSE(attend_sdpa(g.tensor("q"), g.tensor("k"), g.tensor("v")), g.tensor("sdpa_out"), 1e-6);
    CHECK_CLOSE(attend_sdpa(g.tensor("q3"), g.tensor("k"), g.tensor("v")), g.tensor("sdpa_out_3"), 1e-6);
}

TEST_CASE("attend_sdpa and attend_manual agree") {
    const auto& g = golden();
    for (const char* q : {"q", "q3"}) {
        INFO("queries " << q);
        CHECK_CLOSE(attend_sdpa(g.tensor(q), g.tensor("k"), g.tensor("v")),
                    attend_manual(g.tensor(q), g.tensor("k"), g.tensor("v")), 1e-5);
    }
}

TEST_CASE("Attention has the parameters of the reference") {
    const auto& g = golden();
    Attention attn(g.integer("dim"), g.integer("n_heads"), g.integer("n_kv_heads"), g.integer("head_dim"));
    auto names = course::parameter_names(*attn);
    std::sort(names.begin(), names.end());
    std::vector<std::string> expected;
    for (const auto& [name, t] : g.tensors("weights")) expected.push_back(name);
    CHECK(names == expected);
    CHECK(attn->k_proj->weight.sizes() == torch::IntArrayRef{g.integer("n_kv_heads") * g.integer("head_dim"),
                                                             g.integer("dim")});
}

TEST_CASE("Attention gives the output of the reference") {
    const auto& g = golden();
    Attention attn = reference_attention();
    CHECK_CLOSE(attn->forward(g.tensor("x"), g.tensor("cos"), g.tensor("sin")), g.tensor("attention_out"), 1e-6);
}

TEST_CASE("Attention is causal: a change at position 5 does not change outputs 0 .. 4") {
    const auto& g = golden();
    Attention attn = reference_attention();
    torch::Tensor x = g.tensor("x");
    torch::Tensor x2 = x.clone();
    x2.select(1, 5).add_(1.0);
    torch::Tensor y = attn->forward(x, g.tensor("cos"), g.tensor("sin"));
    torch::Tensor y2 = attn->forward(x2, g.tensor("cos"), g.tensor("sin"));
    CHECK_CLOSE(y2.slice(1, 0, 5), y.slice(1, 0, 5), 1e-6);
    CHECK_FALSE(torch::allclose(y2.slice(1, 5), y.slice(1, 5)));
}
