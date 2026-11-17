#include "course/csv_log.hpp"

#include <format>
#include <stdexcept>

namespace course {

CsvLog::CsvLog(const std::filesystem::path& file, std::vector<std::string> columns)
    : columns_(std::move(columns)) {
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());
    out_.open(file, std::ios::trunc);
    if (!out_) throw std::runtime_error(std::format("cannot write {}", file.string()));
    for (size_t i = 0; i < columns_.size(); ++i) out_ << (i ? "," : "") << columns_[i];
    out_ << std::endl;
}

void CsvLog::row(const std::vector<double>& values) {
    if (values.size() != columns_.size()) {
        throw std::invalid_argument(
            std::format("CsvLog::row: {} values for {} columns", values.size(), columns_.size()));
    }
    // {} gives the shortest text that reads back to the same double: 100 -> "100", 0.1 -> "0.1".
    for (size_t i = 0; i < values.size(); ++i) out_ << (i ? "," : "") << std::format("{}", values[i]);
    out_ << std::endl;
}

}  // namespace course
