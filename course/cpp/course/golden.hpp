#pragma once
// Tensor exchange with the Python reference (docs/specification.md, S-1).
//
// Python to C++: Python writes a dictionary with torch.save(dict).
//                C++ reads it with read_dict() or with the class Golden.
// C++ to Python: C++ writes a dictionary of tensors with write_tensors().
//                Python reads it with course.py.io.load().

#include <torch/torch.h>

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace course {

/// Root folder of the repository.
std::filesystem::path source_dir();

/// Folder of the golden files of one lesson: <root>/data/goldens/<lesson>.
/// Example: golden_dir("LLM/L03_rmsnorm_swiglu").
std::filesystem::path golden_dir(std::string_view lesson);

/// Reads a dictionary that Python wrote with torch.save(dict).
/// Throws std::runtime_error when the file is missing or is not a dictionary.
c10::impl::GenericDict read_dict(const std::filesystem::path& file);

/// Reads a dictionary of tensors: a checkpoint of the reference, or a file of write_tensors().
std::map<std::string, torch::Tensor> read_tensors(const std::filesystem::path& file);

/// Writes a dictionary of tensors. Python reads it with course.py.io.load().
/// Makes the parent folders.
void write_tensors(const std::filesystem::path& file,
                   const std::map<std::string, torch::Tensor>& tensors);

/// One golden file: a dictionary with text keys, written by the Python reference.
class Golden {
public:
    explicit Golden(const std::filesystem::path& file);

    bool has(const std::string& key) const;
    torch::Tensor tensor(const std::string& key) const;
    int64_t integer(const std::string& key) const;
    double real(const std::string& key) const;
    std::string text(const std::string& key) const;
    std::vector<std::string> texts(const std::string& key) const;
    /// From a list of int or from a 1-D integer tensor.
    std::vector<int64_t> integers(const std::string& key) const;
    /// The tensors of a nested dictionary, for example golden.tensors("weights").
    std::map<std::string, torch::Tensor> tensors(const std::string& key) const;

    const std::filesystem::path& file() const { return file_; }

private:
    c10::IValue at(const std::string& key) const;

    std::filesystem::path file_;
    c10::impl::GenericDict dict_;
};

/// Copies tensors into the parameters of a module, by parameter name.
/// Throws std::runtime_error when a name is missing, a name is extra, or a shape is not equal.
/// The tests use it to put the weights of the reference into the module of the student.
void copy_parameters(torch::nn::Module& module, const std::map<std::string, torch::Tensor>& weights);

/// Names of the parameters of a module, in registration order.
std::vector<std::string> parameter_names(const torch::nn::Module& module);

}  // namespace course
