#pragma once

#include "mtorch/core/tensor.h"

namespace mtorch::nn {

Tensor mse_loss(const Tensor& pred, const Tensor& target);

} // namespace mtorch::nn
