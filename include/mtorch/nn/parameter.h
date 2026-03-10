#pragma once

#include "mtorch/core/tensor.h"

namespace mtorch::nn {

class Parameter : public Tensor {
public:
    Parameter();
    explicit Parameter(const Tensor& tensor);
};

} // namespace mtorch::nn
