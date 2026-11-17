// L07 Generation and KV cache. Write the bodies: replace each COURSE_TODO line.
// Also write the cache path of Attention::forward (L05) and the position offset of
// Model::forward (L06). Read README.md and generate.py first. Check: ctest --preset clang -L L07

#include "L07_generation/generate.hpp"

#include "course/todo.hpp"

namespace llm {

torch::Tensor filter_probs(const torch::Tensor& logits, double temperature, int64_t top_k, double top_p) {
    COURSE_TODO("filter_probs");
}

torch::Tensor generate(Model& model, const torch::Tensor& ids, const GenerateOptions& options,
                       std::optional<at::Generator> generator) {
    COURSE_TODO("generate");
}

}  // namespace llm
