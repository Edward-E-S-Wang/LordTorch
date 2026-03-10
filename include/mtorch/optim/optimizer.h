#pragma once

#include <vector>
#include "mtorch/core/tensor.h"

namespace mtorch::optim {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> params);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad();

protected:
    std::vector<Tensor> params_;
};

} // namespace mtorch::optim
