#pragma once

#include <ostream>
#include "mtorch/core/tensor.h"

namespace mtorch {

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

} // namespace mtorch
