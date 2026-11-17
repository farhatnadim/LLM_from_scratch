// L06 Block, model, loss. Write the bodies: replace each COURSE_TODO line.
// Read README.md and model.py first. Check: ctest --preset clang -L L06

#include "L06_model/model.hpp"

#include "course/golden.hpp"
#include "course/todo.hpp"

namespace llm {

BlockImpl::BlockImpl(const Config& c) { COURSE_TODO("Block::Block"); }

torch::Tensor BlockImpl::forward(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin,
                                 LayerCache* cache) {
    COURSE_TODO("Block::forward");
}

ModelImpl::ModelImpl(const Config& c) { COURSE_TODO("Model::Model"); }

torch::Tensor ModelImpl::forward(const torch::Tensor& ids, std::vector<LayerCache>* cache) {
    COURSE_TODO("Model::forward");
}

torch::Tensor lm_loss(const torch::Tensor& logits, const torch::Tensor& labels) { COURSE_TODO("lm_loss"); }

void init_weights(Model& model, double std) { COURSE_TODO("init_weights"); }

void save_checkpoint(const Model& model, const std::filesystem::path& file) { COURSE_TODO("save_checkpoint"); }

void load_checkpoint(Model& model, const std::filesystem::path& file) { COURSE_TODO("load_checkpoint"); }

}  // namespace llm
