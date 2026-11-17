#pragma once
// Tensor comparison for the tests (docs/specification.md, S-4).

#include <torch/torch.h>

#include <string>

namespace course {

/// Result of a comparison: ok, and a one-line report for the test output.
struct Comparison {
    bool ok = false;
    double max_abs_diff = 0.0;
    std::string report;
};

/// Compares two floating-point tensors: shape, dtype, then the largest absolute difference.
/// ok is true when shape and dtype are equal and the difference is tol or less.
Comparison compare_close(const torch::Tensor& actual, const torch::Tensor& expected, double tol);

/// Compares two tensors for equal shape, dtype and values (token ids, masks, labels).
Comparison compare_equal(const torch::Tensor& actual, const torch::Tensor& expected);

}  // namespace course
