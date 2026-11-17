#pragma once
// Command-line options of the programs: --name value, or --flag.
//
//     course::Args args(argc, argv, "train --data <dir> [--steps N]");
//     auto data  = args.text("data");             // required
//     auto steps = args.integer("steps", 1000);   // optional, with a default
//     args.finish();                              // stops on an unknown option
//
// --help prints the usage text and stops the program.

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace course {

class Args {
public:
    Args(int argc, const char* const* argv, std::string usage);

    bool has(const std::string& name) const;
    std::string text(const std::string& name) const;  ///< Required. Throws when missing.
    std::string text(const std::string& name, const std::string& fallback) const;
    int64_t integer(const std::string& name, int64_t fallback) const;
    double real(const std::string& name, double fallback) const;
    bool flag(const std::string& name) const;  ///< True when --name is given without a value.

    /// Throws std::invalid_argument when the command line has an option that no getter read.
    void finish() const;

    const std::string& usage() const { return usage_; }

private:
    std::string usage_;
    std::map<std::string, std::string> values_;
    std::set<std::string> flags_;
    mutable std::set<std::string> used_;
};

}  // namespace course
