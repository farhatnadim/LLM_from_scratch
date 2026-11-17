// Unit test of the shared helpers in course/cpp.

#include <fstream>
#include <sstream>

#include "course/args.hpp"
#include "course/csv_log.hpp"
#include "course/testing.hpp"

namespace {
std::filesystem::path scratch(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path() / "llm_from_scratch_course_test";
    std::filesystem::create_directories(dir);
    return dir / name;
}
}  // namespace

TEST_CASE("COURSE_TODO throws NotImplemented with the function name") {
    auto empty_body = [] { COURSE_TODO("Model::forward"); };
    CHECK_THROWS_WITH_AS(empty_body(), "not implemented: Model::forward", course::NotImplemented);
}

TEST_CASE("compare_close and compare_equal") {
    torch::Tensor a = torch::tensor({1.0f, 2.0f, 3.0f});
    CHECK(course::compare_close(a, a + 1e-7, 1e-6).ok);
    CHECK_FALSE(course::compare_close(a, a + 1e-3, 1e-6).ok);
    CHECK_FALSE(course::compare_close(a, a.to(torch::kFloat64), 1.0).ok);      // dtype differs
    CHECK_FALSE(course::compare_close(a.view({1, 3}), a, 1.0).ok);            // shape differs
    CHECK_FALSE(course::compare_close(torch::Tensor(), a, 1.0).ok);           // undefined
    CHECK_FALSE(course::compare_close(a / 0.0 * 0.0, a, 1.0).ok);             // NaN
    CHECK(course::compare_equal(torch::arange(5), torch::arange(5)).ok);
    CHECK_FALSE(course::compare_equal(torch::arange(5), torch::arange(5) * 2).ok);
}

TEST_CASE("write_tensors and read_tensors keep names and values") {
    auto file = scratch("tensors.pt");
    std::map<std::string, torch::Tensor> in{{"layers.0.weight", torch::randn({3, 4})},
                                            {"step", torch::tensor(int64_t{7})}};
    course::write_tensors(file, in);
    auto out = course::read_tensors(file);
    REQUIRE(out.size() == 2);
    CHECK_TENSOR_EQUAL(out.at("layers.0.weight"), in.at("layers.0.weight"));
    CHECK(out.at("step").item<int64_t>() == 7);
}

TEST_CASE("copy_parameters copies by name and rejects wrong names and shapes") {
    torch::nn::Linear layer(3, 2);
    torch::Tensor w = torch::ones({2, 3}), b = torch::zeros({2});
    course::copy_parameters(*layer, {{"weight", w}, {"bias", b}});
    CHECK_TENSOR_EQUAL(layer->weight.detach(), w);
    CHECK_THROWS(course::copy_parameters(*layer, {{"weight", w}}));                                  // missing
    CHECK_THROWS(course::copy_parameters(*layer, {{"weight", w}, {"bias", b}, {"extra", b}}));       // extra
    CHECK_THROWS(course::copy_parameters(*layer, {{"weight", torch::ones({3, 2})}, {"bias", b}}));  // shape
    CHECK(course::parameter_names(*layer) == std::vector<std::string>{"weight", "bias"});
}

TEST_CASE("Args reads values and flags, and finish() stops on unknown options") {
    const char* argv[] = {"prog", "--data", "dir/x", "--steps", "20", "--lr", "0.5", "--resume"};
    course::Args args(8, argv, "prog --data <dir>");
    CHECK(args.text("data") == "dir/x");
    CHECK(args.integer("steps", 1) == 20);
    CHECK(args.real("lr", 0.0) == doctest::Approx(0.5));
    CHECK(args.flag("resume"));
    CHECK(args.integer("batch", 32) == 32);
    CHECK_NOTHROW(args.finish());

    const char* bad[] = {"prog", "--dta", "x"};
    course::Args typo(3, bad, "prog --data <dir>");
    CHECK_THROWS_AS(typo.text("data"), std::invalid_argument);
    CHECK_THROWS_AS(typo.finish(), std::invalid_argument);
}

TEST_CASE("CsvLog writes a header and one line for each row") {
    auto file = scratch("curve.csv");
    {
        course::CsvLog log(file, {"step", "loss"});
        log.row({1, 2.5});
        log.row({2, 0.125});
        CHECK_THROWS(log.row({3}));
    }
    std::ifstream in(file);
    std::stringstream text;
    text << in.rdbuf();
    CHECK(text.str() == "step,loss\n1,2.5\n2,0.125\n");
}
