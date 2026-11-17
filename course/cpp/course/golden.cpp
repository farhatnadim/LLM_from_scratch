#include "course/golden.hpp"

#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace course {

std::filesystem::path source_dir() { return COURSE_SOURCE_DIR; }

std::filesystem::path golden_dir(std::string_view lesson) {
    return source_dir() / "data" / "goldens" / lesson;
}

c10::impl::GenericDict read_dict(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::format(
            "cannot open {}. Make the golden files: uv run tools/make_goldens.py", file.string()));
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    c10::IValue value = torch::pickle_load(bytes);
    if (!value.isGenericDict()) {
        throw std::runtime_error(std::format("{} does not hold a dictionary", file.string()));
    }
    return value.toGenericDict();
}

std::map<std::string, torch::Tensor> read_tensors(const std::filesystem::path& file) {
    std::map<std::string, torch::Tensor> out;
    for (const auto& entry : read_dict(file)) {
        if (entry.value().isTensor()) out[entry.key().toStringRef()] = entry.value().toTensor();
    }
    return out;
}

void write_tensors(const std::filesystem::path& file,
                   const std::map<std::string, torch::Tensor>& tensors) {
    c10::Dict<std::string, torch::Tensor> dict;
    for (const auto& [name, tensor] : tensors) dict.insert(name, tensor.detach().cpu().contiguous());
    std::vector<char> bytes = torch::pickle_save(c10::IValue(dict));
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());
    std::ofstream out(file, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error(std::format("cannot write {}", file.string()));
}

Golden::Golden(const std::filesystem::path& file) : file_(file), dict_(read_dict(file)) {}

bool Golden::has(const std::string& key) const { return dict_.contains(key); }

c10::IValue Golden::at(const std::string& key) const {
    if (!dict_.contains(key)) {
        throw std::runtime_error(std::format("golden file {} has no key '{}'", file_.string(), key));
    }
    return dict_.at(key);
}

torch::Tensor Golden::tensor(const std::string& key) const { return at(key).toTensor(); }

int64_t Golden::integer(const std::string& key) const { return at(key).toInt(); }

double Golden::real(const std::string& key) const {
    c10::IValue v = at(key);
    return v.isInt() ? static_cast<double>(v.toInt()) : v.toDouble();
}

std::string Golden::text(const std::string& key) const { return at(key).toStringRef(); }

std::vector<std::string> Golden::texts(const std::string& key) const {
    std::vector<std::string> out;
    for (const c10::IValue& v : at(key).toListRef()) out.push_back(v.toStringRef());
    return out;
}

std::vector<int64_t> Golden::integers(const std::string& key) const {
    c10::IValue v = at(key);
    std::vector<int64_t> out;
    if (v.isTensor()) {
        torch::Tensor t = v.toTensor().to(torch::kInt64).contiguous().view(-1);
        out.assign(t.data_ptr<int64_t>(), t.data_ptr<int64_t>() + t.numel());
    } else {
        for (const c10::IValue& x : v.toListRef()) out.push_back(x.toInt());
    }
    return out;
}

std::map<std::string, torch::Tensor> Golden::tensors(const std::string& key) const {
    std::map<std::string, torch::Tensor> out;
    for (const auto& entry : at(key).toGenericDict()) {
        out[entry.key().toStringRef()] = entry.value().toTensor();
    }
    return out;
}

void copy_parameters(torch::nn::Module& module, const std::map<std::string, torch::Tensor>& weights) {
    torch::NoGradGuard no_grad;
    auto params = module.named_parameters(/*recurse=*/true);
    for (const auto& item : params) {
        auto it = weights.find(item.key());
        if (it == weights.end()) {
            throw std::runtime_error(std::format("no weight for parameter '{}'", item.key()));
        }
        if (it->second.sizes() != item.value().sizes()) {
            throw std::runtime_error(std::format("parameter '{}' has shape {}, the weight has shape {}",
                                                 item.key(), c10::str(item.value().sizes()),
                                                 c10::str(it->second.sizes())));
        }
        item.value().copy_(it->second);
    }
    for (const auto& [name, tensor] : weights) {
        if (params.find(name) == nullptr) {
            throw std::runtime_error(std::format("weight '{}' has no parameter with that name", name));
        }
    }
}

std::vector<std::string> parameter_names(const torch::nn::Module& module) {
    std::vector<std::string> names;
    for (const auto& item : module.named_parameters(/*recurse=*/true)) names.push_back(item.key());
    return names;
}

}  // namespace course
