#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "mtorch/core/tensor.h"

namespace mtorch::nn {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;

    std::vector<Tensor> parameters() const;
    void train();
    void eval();
    bool is_training() const;

protected:
    void register_parameter(const std::string& name, const Tensor& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

private:
    bool training_ = true;
    std::unordered_map<std::string, Tensor> parameters_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;
};

} // namespace mtorch::nn
