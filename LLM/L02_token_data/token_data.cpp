// L02 Token data. Write the bodies: replace each COURSE_TODO line.
// Read README.md and token_data.py first. Check: ctest --preset clang -L L02

#include "L02_token_data/token_data.hpp"

#include "course/todo.hpp"

namespace llm {

void write_token_files(const std::filesystem::path& folder, const std::vector<std::vector<int>>& samples) {
    COURSE_TODO("write_token_files");
}

int64_t tokenize_jsonl(const std::filesystem::path& jsonl, const Tokenizer& tok,
                       const std::filesystem::path& folder) {
    COURSE_TODO("tokenize_jsonl");
}

TokenFile::TokenFile(const std::filesystem::path& folder) { COURSE_TODO("TokenFile::TokenFile"); }

int64_t TokenFile::size() const { COURSE_TODO("TokenFile::size"); }

std::vector<int> TokenFile::sample(int64_t i) const { COURSE_TODO("TokenFile::sample"); }

Sample pretrain_sample(const std::vector<int>& ids, int64_t max_len) { COURSE_TODO("pretrain_sample"); }

Batch make_batch(const TokenFile& data, const std::vector<int64_t>& indices, int64_t max_len) {
    COURSE_TODO("make_batch");
}

}  // namespace llm
