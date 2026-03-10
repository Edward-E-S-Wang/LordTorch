#include "mtorch/core/tensor.h"
#include <algorithm>
#include <numeric>
#include "mtorch/core/exception.h"
#include "mtorch/core/storage.h"

namespace mtorch {

Tensor::Tensor() = default;
Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

bool Tensor::defined() const { return static_cast<bool>(impl_); }
std::int64_t Tensor::dim() const { return static_cast<std::int64_t>(sizes().size()); }
std::int64_t Tensor::numel() const { return compute_numel(sizes()); }
const std::vector<std::int64_t>& Tensor::sizes() const { return impl_->sizes; }
const std::vector<std::int64_t>& Tensor::strides() const { return impl_->strides; }
bool Tensor::requires_grad() const { return impl_ && impl_->requires_grad; }
void Tensor::set_requires_grad(bool flag) { if (impl_) impl_->requires_grad = flag; }

float* Tensor::data() { return impl_->storage->data() + impl_->storage_offset; }
const float* Tensor::data() const { return impl_->storage->data() + impl_->storage_offset; }

float Tensor::item() const {
    if (numel() != 1) {
        throw MtorchError("item() requires a tensor with exactly one element");
    }
    return data()[0];
}

float& Tensor::operator[](std::int64_t idx) {
    if (idx < 0 || idx >= numel()) {
        throw MtorchError("tensor index out of range");
    }
    return data()[idx];
}

const float& Tensor::operator[](std::int64_t idx) const {
    if (idx < 0 || idx >= numel()) {
        throw MtorchError("tensor index out of range");
    }
    return data()[idx];
}

Tensor Tensor::reshape(const std::vector<std::int64_t>& new_sizes) const {
    if (compute_numel(new_sizes) != numel()) {
        throw MtorchError("reshape changes total number of elements");
    }
    auto out = std::make_shared<TensorImpl>(impl_->storage, new_sizes, contiguous_strides(new_sizes), impl_->storage_offset, impl_->dtype, impl_->device);
    out->requires_grad = impl_->requires_grad;
    out->is_leaf = impl_->is_leaf;
    out->grad_fn = impl_->grad_fn;
    return Tensor(out);
}

Tensor Tensor::clone() const {
    auto new_storage = std::make_shared<Storage>(static_cast<std::size_t>(numel()), impl_->dtype);
    std::copy(data(), data() + numel(), new_storage->data());
    auto out = std::make_shared<TensorImpl>(new_storage, sizes(), contiguous_strides(sizes()), 0, impl_->dtype, impl_->device);
    out->requires_grad = impl_->requires_grad;
    return Tensor(out);
}

std::shared_ptr<TensorImpl> Tensor::impl() const { return impl_; }

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& sizes) {
    if (sizes.empty()) return {};
    std::vector<std::int64_t> strides(sizes.size(), 1);
    for (std::int64_t i = static_cast<std::int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[static_cast<std::size_t>(i)] = strides[static_cast<std::size_t>(i + 1)] * sizes[static_cast<std::size_t>(i + 1)];
    }
    return strides;
}

std::int64_t compute_numel(const std::vector<std::int64_t>& sizes) {
    if (sizes.empty()) return 0;
    return std::accumulate(sizes.begin(), sizes.end(), std::int64_t{1}, std::multiplies<>{});
}

} // namespace mtorch
