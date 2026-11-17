// Test of L06. The golden files come from golden.py (Python reference).
// The test "save_checkpoint" writes cpp_model.pt; then CTest runs L06_model_py, in which
// Python reads that file (weights from C++ to Python).

#include "L06_model/model.hpp"
#include "course/testing.hpp"

using namespace llm;

namespace {
const std::filesystem::path kDir = course::golden_dir("LLM/L06_model");

const course::Golden& golden() {
    static const course::Golden g(kDir / "golden.pt");
    return g;
}

Config small_config() {  // the small model of the golden files
    const auto& g = golden();
    Config c;
    c.vocab_size = g.integer("vocab_size");
    c.dim = g.integer("dim");
    c.n_layers = g.integer("n_layers");
    c.n_heads = g.integer("n_heads");
    c.n_kv_heads = g.integer("n_kv_heads");
    c.head_dim = g.integer("head_dim");
    c.hidden = g.integer("hidden");
    c.eps = g.real("eps");
    c.rope_base = g.real("rope_base");
    c.max_len = g.integer("max_len");
    return c;
}

int64_t count_parameters(const torch::nn::Module& m) {
    int64_t n = 0;
    for (const auto& p : m.parameters()) n += p.numel();
    return n;
}
}  // namespace

TEST_CASE("Model of the default Config has 63 912 192 parameters") {
    Model model(Config{});
    CHECK(count_parameters(*model) == golden().integer("full_parameters"));
}

TEST_CASE("Model has the parameter names of the reference; the head has no parameter") {
    Model model(small_config());
    auto names = course::parameter_names(*model);
    std::sort(names.begin(), names.end());
    CHECK(names == golden().texts("names"));
}

TEST_CASE("Model keeps the RoPE tables as the buffers cos and sin") {
    Config c = small_config();
    Model model(c);
    auto buffers = model->named_buffers();
    REQUIRE(buffers.contains("cos"));
    REQUIRE(buffers.contains("sin"));
    CHECK(buffers["cos"].sizes() == torch::IntArrayRef{c.max_len, c.head_dim});
}

TEST_CASE("Block gives the output of the reference") {
    const auto& g = golden();
    Block block(small_config());
    course::copy_parameters(*block, g.tensors("block_weights"));
    torch::Tensor out = block->forward(g.tensor("block_x"), g.tensor("block_cos"), g.tensor("block_sin"));
    CHECK_CLOSE(out, g.tensor("block_out"), 1e-6);
}

TEST_CASE("load_checkpoint reads the reference; logits and loss agree") {
    const auto& g = golden();
    Model model(small_config());
    load_checkpoint(model, kDir / "model.pt");
    torch::Tensor logits = model->forward(g.tensor("ids"));
    CHECK_CLOSE(logits, g.tensor("logits"), 1e-5);
    CHECK_CLOSE(lm_loss(logits, g.tensor("labels")), g.tensor("loss"), 1e-6);
}

TEST_CASE("load_checkpoint stops on a checkpoint of another size") {
    Config c = small_config();
    c.hidden += 4;
    Model model(c);
    CHECK_THROWS(load_checkpoint(model, kDir / "model.pt"));
}

TEST_CASE("lm_loss ignores the labels -100") {
    const auto& g = golden();
    torch::Tensor logits = g.tensor("logits").clone();
    torch::Tensor labels = g.tensor("labels");  // sample 1 has -100 from position 6
    torch::Tensor loss = lm_loss(logits, labels);
    logits[1].slice(0, 5).add_(3.0);            // position t predicts label t+1: positions 5.. predict -100
    CHECK_CLOSE(lm_loss(logits, labels), loss, 1e-6);
}

TEST_CASE("init_weights: matrices from N(0, 0.02^2), RMSNorm weights 1") {
    torch::manual_seed(0);
    Model model(Config{});
    init_weights(model);
    for (const auto& item : model->named_parameters()) {
        INFO(item.key());
        torch::Tensor p = item.value().detach();
        if (p.dim() == 2) {
            CHECK(std::abs(p.mean().item<double>()) < 1e-3);
            CHECK(std::abs(p.std().item<double>() - 0.02) < 5e-4);
        } else {
            CHECK(torch::equal(p, torch::ones_like(p)));
        }
    }
}

TEST_CASE("save_checkpoint writes the parameters; load_checkpoint reads them back") {
    Model model(small_config());
    load_checkpoint(model, kDir / "model.pt");
    save_checkpoint(model, kDir / "cpp_model.pt");  // CTest test L06_model_py reads this file in Python
    Model again(small_config());
    load_checkpoint(again, kDir / "cpp_model.pt");
    for (const auto& item : model->named_parameters()) {
        INFO(item.key());
        CHECK(torch::equal(item.value(), again->named_parameters()[item.key()]));
    }
}
