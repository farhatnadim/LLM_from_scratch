#pragma once
// The empty body of a function that the student writes.
//
//     torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) {
//         COURSE_TODO("RMSNorm::forward");
//     }
//
// The test of the lesson then fails with the text "not implemented: RMSNorm::forward".
// Replace the line with the body.

#include <stdexcept>
#include <string>
#include <string_view>

namespace course {

/// The exception of an empty function body. what() is "not implemented: <name>".
class NotImplemented : public std::logic_error {
public:
    explicit NotImplemented(std::string_view name)
        : std::logic_error("not implemented: " + std::string(name)) {}
};

}  // namespace course

#define COURSE_TODO(name) throw ::course::NotImplemented(name)
