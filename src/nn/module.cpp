#include "mtorch/nn/module.h"

namespace mtorch::nn {

std::vector<Tensor> Module::parameters() const {
    std::vector<Tensor> out;
    out.reserve(parameters_.size());
    for (const auto& kv : parameters_) out.push_back(kv.second);
    for (const auto& kv : submodules_) {
        auto child_params = kv.second->parameters();
        out.insert(out.end(), child_params.begin(), child_params.end());
    }
    return out;
}

void Module::train() { training_ = true; }
void Module::eval() { training_ = false; }
bool Module::is_training() const { return training_; }

void Module::register_parameter(const std::string& name, const Tensor& param) {
    parameters_[name] = param;
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    submodules_[name] = std::move(module);
}

} // namespace mtorch::nn
