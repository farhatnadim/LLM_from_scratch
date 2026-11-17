// Test of L01. The golden file comes from golden.py (Python reference).

#include <fstream>
#include <sstream>

#include "L01_tokenizer/tokenizer.hpp"
#include "course/testing.hpp"

using namespace llm;

namespace {
const std::filesystem::path kDir = course::golden_dir("LLM/L01_tokenizer");

const course::Golden& golden() {
    static const course::Golden g(kDir / "golden.pt");
    return g;
}

std::vector<Merge> to_merges(const torch::Tensor& t) {  // [M, 2] -> merges
    std::vector<Merge> out;
    for (int64_t i = 0; i < t.size(0); ++i) out.emplace_back(t[i][0].item<int>(), t[i][1].item<int>());
    return out;
}

torch::Tensor to_tensor(const std::vector<int>& ids) {
    return torch::tensor(std::vector<int64_t>(ids.begin(), ids.end()), torch::kInt64);
}

std::string read_file(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}
}  // namespace

TEST_CASE("special_text gives the text of each special id") {
    auto expected = golden().texts("special_texts");
    for (int id = kFirstSpecial; id < kFirstMerge; ++id) {
        INFO("id " << id);
        CHECK(special_text(id) == expected[id - kFirstSpecial]);
    }
    CHECK_THROWS_AS(special_text(kFirstMerge), std::out_of_range);
}

TEST_CASE("split_special cuts the special texts out of a chat text") {
    auto pieces = split_special(golden().text("sample"));
    auto texts = golden().texts("piece_texts");
    auto ids = golden().integers("piece_ids");
    REQUIRE(pieces.size() == texts.size());
    for (size_t i = 0; i < pieces.size(); ++i) {
        INFO("piece " << i);
        CHECK(pieces[i].text == texts[i]);
        CHECK(pieces[i].special == ids[i]);
    }
}

TEST_CASE("pretokenize cuts at white space") {
    auto got = pretokenize(golden().text("plain"));
    auto expected = golden().texts("pretokens");
    REQUIRE(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        INFO("pre-token " << i);
        CHECK(got[i] == expected[i]);
    }
}

TEST_CASE("train_bpe learns the merges of the reference, ties included") {
    auto merges = train_bpe(golden().texts("corpus"), static_cast<int>(golden().integer("vocab_size")));
    auto expected = to_merges(golden().tensor("merges"));
    REQUIRE(merges.size() == expected.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        INFO("merge " << i << " (id " << kFirstMerge + i << "): expected " << expected[i].first << " "
                      << expected[i].second);
        CHECK(merges[i].first == expected[i].first);
        CHECK(merges[i].second == expected[i].second);
    }
}

TEST_CASE("Tokenizer builds vocab and rank from the merges") {
    auto merges = to_merges(golden().tensor("merges"));
    Tokenizer tok(merges);
    REQUIRE(tok.vocab.size() == kFirstMerge + merges.size());
    CHECK(tok.vocab[65] == "A");
    CHECK(tok.vocab[kImStart] == "<|im_start|>");
    CHECK(tok.vocab[kFirstMerge] == tok.vocab[merges[0].first] + tok.vocab[merges[0].second]);
    CHECK(tok.rank.at(merges[5]) == 5);
}

TEST_CASE("Tokenizer encodes and decodes as the reference") {
    Tokenizer tok(to_merges(golden().tensor("merges")));
    std::vector<int> ids = tok.encode(golden().text("sample"));
    CHECK_TENSOR_EQUAL(to_tensor(ids), golden().tensor("ids"));
    CHECK(tok.decode(ids) == golden().text("decoded"));
    CHECK(tok.encode(golden().text("sample")) == ids);  // second time: from the cache
}

TEST_CASE("Tokenizer::load reads the file of the reference; save writes the same bytes") {
    Tokenizer tok = Tokenizer::load(kDir / "merges.txt");
    CHECK(tok.merges == to_merges(golden().tensor("merges")));
    auto out = std::filesystem::temp_directory_path() / "llm_from_scratch" / "L01" / "merges.txt";
    tok.save(out);
    CHECK(read_file(out) == read_file(kDir / "merges.txt"));
}
