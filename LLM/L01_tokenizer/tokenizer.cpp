// L01 BPE tokenizer. Write the bodies: replace each COURSE_TODO line.
// Read README.md and tokenizer.py first. Check: ctest --preset clang -L L01
// You can add helper functions in an anonymous namespace in this file.

#include "L01_tokenizer/tokenizer.hpp"

#include "course/todo.hpp"

namespace llm {

std::string special_text(int id) { COURSE_TODO("special_text"); }

std::vector<Piece> split_special(std::string_view text) { COURSE_TODO("split_special"); }

std::vector<std::string> pretokenize(std::string_view text) { COURSE_TODO("pretokenize"); }

std::vector<Merge> train_bpe(const std::vector<std::string>& texts, int vocab_size) {
    COURSE_TODO("train_bpe");
}

Tokenizer::Tokenizer(std::vector<Merge> merges) { COURSE_TODO("Tokenizer::Tokenizer"); }

std::vector<int> Tokenizer::encode(std::string_view text) const { COURSE_TODO("Tokenizer::encode"); }

std::string Tokenizer::decode(const std::vector<int>& ids) const { COURSE_TODO("Tokenizer::decode"); }

void Tokenizer::save(const std::filesystem::path& file) const { COURSE_TODO("Tokenizer::save"); }

Tokenizer Tokenizer::load(const std::filesystem::path& file) { COURSE_TODO("Tokenizer::load"); }

}  // namespace llm
