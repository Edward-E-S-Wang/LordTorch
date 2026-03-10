#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "mtorch/core/device.h"
#include "mtorch/core/scalar_type.h"
#include "mtorch/core/storage.h"
#include "mtorch/autograd/node.h"

namespace mtorch {

class Tensor;

class TensorImpl {
public:
    TensorImpl(std::shared_ptr<Storage> storage,
               std::vector<std::int64_t> sizes,
               std::vector<std::int64_t> strides,
               std::int64_t storage_offset,
               ScalarType dtype,
               Device device);

    std::shared_ptr<Storage> storage;
    std::vector<std::int64_t> sizes;
    std::vector<std::int64_t> strides;
    std::int64_t storage_offset = 0;
    ScalarType dtype = ScalarType::Float32;
    Device device{};
    bool requires_grad = false;
    bool is_leaf = true;
    std::shared_ptr<Tensor> grad;
    autograd::NodePtr grad_fn;
};

} // namespace mtorch
