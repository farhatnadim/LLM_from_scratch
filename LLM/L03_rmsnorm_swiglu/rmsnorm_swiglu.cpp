// L03 RMSNorm and SwiGLU. Write the bodies: replace each COURSE_TODO line.
// Read README.md and rmsnorm_swiglu.py first. Check: ctest --preset clang -L L03

#include "L03_rmsnorm_swiglu/rmsnorm_swiglu.hpp"

#include "course/todo.hpp"

namespace llm {

RMSNormImpl::RMSNormImpl(int64_t dim, double eps) { COURSE_TODO("RMSNorm::RMSNorm"); }

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) { COURSE_TODO("RMSNorm::forward"); }

FeedForwardImpl::FeedForwardImpl(int64_t dim, int64_t hidden) { COURSE_TODO("FeedForward::FeedForward"); }

torch::Tensor FeedForwardImpl::forward(const torch::Tensor& x) { COURSE_TODO("FeedForward::forward"); }

}  // namespace llm
