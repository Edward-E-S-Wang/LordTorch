#pragma once

#include <cstdint>
#include <vector>
#include "mtorch/core/tensor.h"
#include "mtorch/core/tensor_options.h"

namespace mtorch {

Tensor empty(const std::vector<std::int64_t>& sizes, const TensorOptions& options = {});
Tensor zeros(const std::vector<std::int64_t>& sizes, const TensorOptions& options = {});
Tensor ones(const std::vector<std::int64_t>& sizes, const TensorOptions& options = {});
Tensor randn(const std::vector<std::int64_t>& sizes, const TensorOptions& options = {});
Tensor tensor(const std::vector<float>& values, const std::vector<std::int64_t>& sizes, const TensorOptions& options = {});

} // namespace mtorch
