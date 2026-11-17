// Test of L04. The golden file comes from golden.py (Python reference).
// Limit 1e-5: the angle grows with the position, so two correct formulas differ more than 1e-6.

#include "L04_rope/rope.hpp"
#include "course/testing.hpp"

using namespace llm;

namespace {
const course::Golden& golden() {
    static const course::Golden g(course::golden_dir("LLM/L04_rope") / "golden.pt");
    return g;
}

RopeTables tables() {
    const auto& g = golden();
    return rope_tables(g.integer("head_dim"), g.integer("max_len"), g.real("base"));
}
}  // namespace

TEST_CASE("rope_tables gives the cos and sin tables of the reference") {
    RopeTables t = tables();
    CHECK_CLOSE(t.cos, golden().tensor("cos"), 1e-5);
    CHECK_CLOSE(t.sin, golden().tensor("sin"), 1e-5);
}

TEST_CASE("rotate_half gives (-x2, x1)") {
    CHECK_TENSOR_EQUAL(rotate_half(golden().tensor("x")), golden().tensor("rotated_half"));
}

TEST_CASE("apply_rope at positions 0 .. 9") {
    RopeTables t = tables();
    torch::Tensor out = apply_rope(golden().tensor("x"), t.cos.slice(0, 0, 10), t.sin.slice(0, 0, 10));
    CHECK_CLOSE(out, golden().tensor("rope_out"), 1e-5);
}

TEST_CASE("apply_rope at positions 37 .. 46 (the KV cache of L07 needs an offset)") {
    RopeTables t = tables();
    int64_t s = golden().integer("offset");
    torch::Tensor out = apply_rope(golden().tensor("x"), t.cos.slice(0, s, s + 10), t.sin.slice(0, s, s + 10));
    CHECK_CLOSE(out, golden().tensor("rope_out_offset"), 1e-5);
}

TEST_CASE("RoPE keeps the length of each head vector") {
    RopeTables t = tables();
    torch::Tensor x = golden().tensor("x");
    torch::Tensor out = apply_rope(x, t.cos.slice(0, 0, 10), t.sin.slice(0, 0, 10));
    CHECK_CLOSE(out.norm(2, -1), x.norm(2, -1), 1e-4);
}

TEST_CASE("After RoPE, q.k depends only on the distance between the positions") {
    RopeTables t = rope_tables(96, 2048, 1e6);
    torch::manual_seed(0);
    torch::Tensor q = torch::randn({1, 1, 1, 96}), k = torch::randn({1, 1, 1, 96});
    auto dot = [&](int64_t m, int64_t n) {
        torch::Tensor qm = apply_rope(q, t.cos.slice(0, m, m + 1), t.sin.slice(0, m, m + 1));
        torch::Tensor kn = apply_rope(k, t.cos.slice(0, n, n + 1), t.sin.slice(0, n, n + 1));
        return (qm * kn).sum().item<double>();
    };
    CHECK(dot(5, 2) == doctest::Approx(dot(105, 102)).epsilon(1e-4));
    CHECK(dot(5, 2) == doctest::Approx(dot(505, 502)).epsilon(1e-4));
}
