// Test of L02. The golden files come from golden.py (Python reference).

#include <fstream>
#include <sstream>

#include "L02_token_data/token_data.hpp"
#include "course/testing.hpp"

using namespace llm;

namespace {
const std::filesystem::path kDir = course::golden_dir("LLM/L02_token_data");

const course::Golden& golden() {
    static const course::Golden g(kDir / "golden.pt");
    return g;
}

std::string read_file(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

std::filesystem::path scratch(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path() / "llm_from_scratch" / "L02" / name;
    std::filesystem::remove_all(dir);
    return dir;
}

std::vector<std::vector<int>> golden_samples() {  // the samples, from the golden tokens and offsets
    auto tokens = golden().integers("tokens");
    auto offsets = golden().integers("offsets");
    std::vector<std::vector<int>> samples;
    for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        samples.emplace_back(tokens.begin() + offsets[i], tokens.begin() + offsets[i + 1]);
    }
    return samples;
}
}  // namespace

TEST_CASE("write_token_files writes the bytes of the reference") {
    auto out = scratch("write");
    write_token_files(out, golden_samples());
    CHECK(read_file(out / "tokens.bin") == read_file(kDir / "tokens" / "tokens.bin"));
    CHECK(read_file(out / "offsets.bin") == read_file(kDir / "tokens" / "offsets.bin"));
}

TEST_CASE("TokenFile reads the files of the reference") {
    TokenFile data(kDir / "tokens");
    REQUIRE(data.size() == golden().integer("count"));
    auto samples = golden_samples();
    for (int64_t i = 0; i < data.size(); ++i) {
        INFO("sample " << i);
        CHECK(data.sample(i) == samples[i]);
    }
    CHECK(data.offsets.back() == data.tokens.size());
}

TEST_CASE("pretrain_sample pads a short sample") {
    TokenFile data(kDir / "tokens");
    Sample s = pretrain_sample(data.sample(golden().integer("short_index")), golden().integer("short_max_len"));
    CHECK_TENSOR_EQUAL(s.input_ids, golden().tensor("short_input_ids"));
    CHECK_TENSOR_EQUAL(s.labels, golden().tensor("short_labels"));
}

TEST_CASE("pretrain_sample cuts a long sample and keeps the end id") {
    TokenFile data(kDir / "tokens");
    Sample s = pretrain_sample(data.sample(golden().integer("long_index")), golden().integer("long_max_len"));
    CHECK_TENSOR_EQUAL(s.input_ids, golden().tensor("long_input_ids"));
    CHECK_TENSOR_EQUAL(s.labels, golden().tensor("long_labels"));
}

TEST_CASE("make_batch stacks the samples at the indices") {
    TokenFile data(kDir / "tokens");
    Batch b = make_batch(data, golden().integers("batch_indices"), golden().integer("batch_max_len"));
    CHECK_TENSOR_EQUAL(b.input_ids, golden().tensor("batch_input_ids"));
    CHECK_TENSOR_EQUAL(b.labels, golden().tensor("batch_labels"));
}

TEST_CASE("tokenize_jsonl gives the token files of the reference (uses the tokenizer of L01)") {
    Tokenizer tok = Tokenizer::load(kDir / "merges.txt");
    auto out = scratch("tokenize");
    CHECK(tokenize_jsonl(kDir / "corpus.jsonl", tok, out) == golden().integer("count"));
    CHECK(read_file(out / "tokens.bin") == read_file(kDir / "tokens" / "tokens.bin"));
    CHECK(read_file(out / "offsets.bin") == read_file(kDir / "tokens" / "offsets.bin"));
}
