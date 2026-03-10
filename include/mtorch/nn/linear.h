#pragma once

#include "mtorch/nn/module.h"

namespace mtorch::nn {

class Linear : public Module {
public:
    Linear(std::int64_t in_features, std::int64_t out_features);
    Tensor forward(const Tensor& x) override;

    Tensor weight() const;
    Tensor bias() const;

private:
    Tensor weight_;
    Tensor bias_;
    std::int64_t in_features_;
    std::int64_t out_features_;
};

} // namespace mtorch::nn
