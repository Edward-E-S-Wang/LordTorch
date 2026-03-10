#include "mtorch/ops/creation.h"
#include <algorithm>
#include <random>
#include "mtorch/core/exception.h"

namespace mtorch {

namespace {
Tensor make_tensor(const std::vector<std::int64_t>& sizes, const TensorOptions& options) {
    auto storage = std::make_shared<Storage>(static_cast<std::size_t>(compute_numel(sizes)), options.dtype());
    auto impl = std::make_shared<TensorImpl>(storage, sizes, contiguous_strides(sizes), 0, options.dtype(), options.device());
    impl->requires_grad = options.requires_grad();
    return Tensor(impl);
}
} // namespace

Tensor empty(const std::vector<std::int64_t>& sizes, const TensorOptions& options) {
    return make_tensor(sizes, options);
}

Tensor zeros(const std::vector<std::int64_t>& sizes, const TensorOptions& options) {
    auto t = make_tensor(sizes, options);
    std::fill(t.data(), t.data() + t.numel(), 0.0f);
    return t;
}

Tensor ones(const std::vector<std::int64_t>& sizes, const TensorOptions& options) {
    auto t = make_tensor(sizes, options);
    std::fill(t.data(), t.data() + t.numel(), 1.0f);
    return t;
}

Tensor randn(const std::vector<std::int64_t>& sizes, const TensorOptions& options) {
    auto t = make_tensor(sizes, options);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::int64_t i = 0; i < t.numel(); ++i) t[i] = dist(gen);
    return t;
}

Tensor tensor(const std::vector<float>& values, const std::vector<std::int64_t>& sizes, const TensorOptions& options) {
    if (static_cast<std::int64_t>(values.size()) != compute_numel(sizes)) {
        throw MtorchError("tensor(values, sizes): values.size() does not match sizes product");
    }
    auto t = make_tensor(sizes, options);
    std::copy(values.begin(), values.end(), t.data());
    return t;
}

} // namespace mtorch
