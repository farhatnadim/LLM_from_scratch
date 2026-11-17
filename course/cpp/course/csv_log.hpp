#pragma once
// A CSV file for training curves (docs/specification.md, S-4): one header line,
// then one line for each call of row(). Each line goes to the disk at once,
// so the file is complete if the run stops.
//
//     course::CsvLog log("runs/pretrain/curve.csv", {"step", "loss", "lr", "seconds"});
//     log.row({step, loss, lr, seconds});

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace course {

class CsvLog {
public:
    /// Makes the parent folders and writes the header. An old file is replaced.
    CsvLog(const std::filesystem::path& file, std::vector<std::string> columns);

    /// Writes one line. Throws std::invalid_argument when the count of values is not the count of columns.
    void row(const std::vector<double>& values);

private:
    std::ofstream out_;
    std::vector<std::string> columns_;
};

}  // namespace course
