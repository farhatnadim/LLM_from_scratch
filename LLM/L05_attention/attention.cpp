// L05 Attention. Write the bodies: replace each COURSE_TODO line.
// Read README.md and attention.py first. Check: ctest --preset clang -L L05
// The cache path of Attention::forward is lesson L07: until then, you can write
//     if (cache) COURSE_TODO("Attention::forward with a cache (L07)");

#include "L05_attention/attention.hpp"

#include "course/todo.hpp"

namespace llm {

torch::Tensor causal_mask(int64_t T, int64_t S, torch::Device device) { COURSE_TODO("causal_mask"); }

torch::Tensor attend_manual(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    COURSE_TODO("attend_manual");
}

torch::Tensor attend_sdpa(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    COURSE_TODO("attend_sdpa");
}

AttentionImpl::AttentionImpl(int64_t dim, int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, double eps) {
    COURSE_TODO("Attention::Attention");
}

torch::Tensor AttentionImpl::forward(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin,
                                     LayerCache* cache) {
    COURSE_TODO("Attention::forward");
}

}  // namespace llm
