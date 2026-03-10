#pragma once

#include "mtorch/optim/optimizer.h"

namespace mtorch::optim {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> params, float lr);
    void step() override;

private:
    float lr_ = 1e-3f;
};

} // namespace mtorch::optim
