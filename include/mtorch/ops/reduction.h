#pragma once

#include "mtorch/core/tensor.h"

namespace mtorch {

Tensor sum(const Tensor& x);
Tensor mean(const Tensor& x);

} // namespace mtorch
