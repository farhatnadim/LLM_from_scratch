// L04 RoPE. Write the bodies: replace each COURSE_TODO line.
// Read README.md and rope.py first. Check: ctest --preset clang -L L04

#include "L04_rope/rope.hpp"

#include "course/todo.hpp"

namespace llm {

RopeTables rope_tables(int64_t head_dim, int64_t max_len, double base) { COURSE_TODO("rope_tables"); }

torch::Tensor rotate_half(const torch::Tensor& x) { COURSE_TODO("rotate_half"); }

torch::Tensor apply_rope(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin) {
    COURSE_TODO("apply_rope");
}

}  // namespace llm
