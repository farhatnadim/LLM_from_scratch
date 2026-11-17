// Test of L07. The golden files come from golden.py (Python reference).
// Sampled ids are not compared (random streams); the probabilities after the filters are.

#include "L07_generation/generate.hpp"
#include "course/testing.hpp"

using namespace llm;

namespace {
const std::filesystem::path kDir = course::golden_dir("LLM/L07_generation");

const course::Golden& golden() {
    static const course::Golden g(kDir / "golden.pt");
    return g;
}

Model reference_model() {
    const auto& g = golden();
    Config c;
    c.vocab_size = g.integer("vocab_size");
    c.dim = g.integer("dim");
    c.n_layers = g.integer("n_layers");
    c.n_heads = g.integer("n_heads");
    c.n_kv_heads = g.integer("n_kv_heads");
    c.head_dim = g.integer("head_dim");
    c.hidden = g.integer("hidden");
    c.max_len = g.integer("max_len");
    Model model(c);
    load_checkpoint(model, kDir / "model.pt");
    model->eval();
    return model;
}
}  // namespace

TEST_CASE("filter_probs gives the probabilities of the reference") {
    const auto& g = golden();
    for (int64_t i = 0; i < g.integer("n_cases"); ++i) {
        std::string c = "case" + std::to_string(i) + "_";
        INFO(c << ": temperature " << g.real(c + "temperature") << ", top_k " << g.integer(c + "top_k")
               << ", top_p " << g.real(c + "top_p"));
        torch::Tensor p = filter_probs(g.tensor("logits"), g.real(c + "temperature"), g.integer(c + "top_k"),
                                       g.real(c + "top_p"));
        CHECK_CLOSE(p, g.tensor(c + "probs"), 1e-6);
    }
}

TEST_CASE("filter_probs with top_k = 1 keeps the largest logit only") {
    torch::Tensor logits = golden().tensor("logits");
    torch::Tensor p = filter_probs(logits, 1.0, 1, 1.0);
    CHECK_TENSOR_EQUAL(p.argmax(-1), logits.argmax(-1));
    CHECK_CLOSE(p.amax(-1), torch::ones({logits.size(0)}), 1e-6);
}

TEST_CASE("KV cache: logits step by step equal the logits of the whole sequence") {
    torch::NoGradGuard no_grad;
    Model model = reference_model();
    torch::Tensor ids = torch::cat({golden().tensor("prompt"), golden().tensor("greedy")}, 1);  // [2, 17]
    torch::Tensor full = model->forward(ids);                                                  // [2, 17, V]
    std::vector<LayerCache> cache(model->config.n_layers);
    int64_t T = golden().tensor("prompt").size(1);
    torch::Tensor first = model->forward(ids.slice(1, 0, T), &cache);                          // the prompt
    CHECK_CLOSE(first, full.slice(1, 0, T), 1e-5);
    for (int64_t t = T; t < ids.size(1); ++t) {
        INFO("position " << t);
        torch::Tensor step = model->forward(ids.slice(1, t, t + 1), &cache);                   // one id
        CHECK_CLOSE(step, full.slice(1, t, t + 1), 1e-5);
    }
    CHECK(cache[0].k.size(2) == ids.size(1));
}

TEST_CASE("generate, greedy: the ids of the reference") {
    Model model = reference_model();
    GenerateOptions opt;
    opt.max_new = golden().integer("max_new");
    CHECK_TENSOR_EQUAL(generate(model, golden().tensor("prompt"), opt), golden().tensor("greedy"));
}

TEST_CASE("generate with eos: a row writes eos after eos; the loop stops when all rows wrote it") {
    Model model = reference_model();
    GenerateOptions opt;
    opt.max_new = golden().integer("max_new");
    opt.eos = golden().integer("eos");
    CHECK_TENSOR_EQUAL(generate(model, golden().tensor("prompt"), opt), golden().tensor("greedy_eos"));
}

TEST_CASE("generate, sampled with top_k = 1: the greedy ids") {
    Model model = reference_model();
    GenerateOptions opt;
    opt.max_new = golden().integer("max_new");
    opt.temperature = 1.0;
    opt.top_k = 1;
    at::Generator gen = at::make_generator<at::CPUGeneratorImpl>(42);
    CHECK_TENSOR_EQUAL(generate(model, golden().tensor("prompt"), opt, gen), golden().tensor("greedy"));
}
