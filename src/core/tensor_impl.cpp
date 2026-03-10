#include "mtorch/core/tensor_impl.h"

namespace mtorch {

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
                       std::vector<std::int64_t> sizes,
                       std::vector<std::int64_t> strides,
                       std::int64_t storage_offset,
                       ScalarType dtype,
                       Device device)
    : storage(std::move(storage)),
      sizes(std::move(sizes)),
      strides(std::move(strides)),
      storage_offset(storage_offset),
      dtype(dtype),
      device(device) {}

} // namespace mtorch
