#include "course/args.hpp"

#include <cstdlib>
#include <format>
#include <iostream>
#include <stdexcept>

namespace course {

Args::Args(int argc, const char* const* argv, std::string usage) : usage_(std::move(usage)) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << usage_ << '\n';
            std::exit(0);
        }
        if (!arg.starts_with("--") || arg.size() == 2) {
            throw std::invalid_argument(std::format("unexpected argument '{}'. usage: {}", arg, usage_));
        }
        std::string name = arg.substr(2);
        bool next_is_value = i + 1 < argc && !std::string(argv[i + 1]).starts_with("--");
        if (next_is_value) {
            values_[name] = argv[++i];
        } else {
            flags_.insert(name);
        }
    }
}

bool Args::has(const std::string& name) const {
    return values_.contains(name) || flags_.contains(name);
}

std::string Args::text(const std::string& name) const {
    used_.insert(name);
    auto it = values_.find(name);
    if (it == values_.end()) {
        throw std::invalid_argument(std::format("missing option --{} <value>. usage: {}", name, usage_));
    }
    return it->second;
}

std::string Args::text(const std::string& name, const std::string& fallback) const {
    used_.insert(name);
    auto it = values_.find(name);
    return it == values_.end() ? fallback : it->second;
}

int64_t Args::integer(const std::string& name, int64_t fallback) const {
    std::string v = text(name, "");
    if (v.empty()) return fallback;
    size_t end = 0;
    int64_t out = std::stoll(v, &end);
    if (end != v.size()) throw std::invalid_argument(std::format("--{} needs an integer, not '{}'", name, v));
    return out;
}

double Args::real(const std::string& name, double fallback) const {
    std::string v = text(name, "");
    if (v.empty()) return fallback;
    size_t end = 0;
    double out = std::stod(v, &end);
    if (end != v.size()) throw std::invalid_argument(std::format("--{} needs a number, not '{}'", name, v));
    return out;
}

bool Args::flag(const std::string& name) const {
    used_.insert(name);
    return flags_.contains(name);
}

void Args::finish() const {
    for (const auto& [name, value] : values_) {
        if (!used_.contains(name)) throw std::invalid_argument(std::format("unknown option --{}. usage: {}", name, usage_));
    }
    for (const auto& name : flags_) {
        if (!used_.contains(name)) throw std::invalid_argument(std::format("unknown option --{}. usage: {}", name, usage_));
    }
}

}  // namespace course
