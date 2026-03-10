#pragma once

#include "mtorch/core/tensor.h"

namespace mtorch {

Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);

} // namespace mtorch
