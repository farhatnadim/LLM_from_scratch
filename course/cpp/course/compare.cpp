#include "course/compare.hpp"

#include <format>

namespace course {
namespace {

std::string describe(const torch::Tensor& t) {
    if (!t.defined()) return "undefined tensor";
    return std::format("{} {}", c10::str(t.sizes()), c10::toString(t.scalar_type()));
}

// Shape and dtype. Returns an empty text when they agree.
std::string check_layout(const torch::Tensor& actual, const torch::Tensor& expected) {
    if (!actual.defined()) return "the result is an undefined tensor";
    if (actual.sizes() != expected.sizes() || actual.scalar_type() != expected.scalar_type()) {
        return std::format("result is {}, expected {}", describe(actual), describe(expected));
    }
    return {};
}

}  // namespace

Comparison compare_close(const torch::Tensor& actual, const torch::Tensor& expected, double tol) {
    Comparison c;
    if (std::string bad = check_layout(actual, expected); !bad.empty()) {
        c.report = bad;
        return c;
    }
    torch::Tensor a = actual.detach().cpu().to(torch::kFloat64);
    torch::Tensor e = expected.detach().cpu().to(torch::kFloat64);
    if (a.numel() > 0) {
        torch::Tensor diff = (a - e).abs();
        // NaN in the result is a failure: NaN compares false with everything.
        if (diff.isnan().any().item<bool>()) {
            c.report = std::format("result has NaN ({})", describe(actual));
            return c;
        }
        c.max_abs_diff = diff.max().item<double>();
    }
    c.ok = c.max_abs_diff <= tol;
    c.report = std::format("{}: largest absolute difference {:.3e}, limit {:.1e}", describe(actual),
                           c.max_abs_diff, tol);
    return c;
}

Comparison compare_equal(const torch::Tensor& actual, const torch::Tensor& expected) {
    Comparison c;
    if (std::string bad = check_layout(actual, expected); !bad.empty()) {
        c.report = bad;
        return c;
    }
    torch::Tensor a = actual.detach().cpu();
    torch::Tensor e = expected.detach().cpu();
    c.ok = torch::equal(a, e);
    if (c.ok) {
        c.report = std::format("{}: equal", describe(actual));
    } else {
        torch::Tensor wrong = (a != e).reshape(-1).nonzero();
        int64_t first = wrong.numel() > 0 ? wrong[0].item<int64_t>() : -1;
        c.report = std::format("{}: {} of {} values differ; first at flat index {} ({} != expected {})",
                               describe(actual), wrong.size(0), a.numel(), first,
                               first >= 0 ? c10::str(a.reshape(-1)[first].item()) : "?",
                               first >= 0 ? c10::str(e.reshape(-1)[first].item()) : "?");
    }
    return c;
}

}  // namespace course
