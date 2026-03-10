#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "mtorch/core/tensor_impl.h"
#include "mtorch/core/tensor_options.h"

namespace mtorch {

class Tensor {
public:
    Tensor();
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    bool defined() const;
    std::int64_t dim() const;
    std::int64_t numel() const;
    const std::vector<std::int64_t>& sizes() const;
    const std::vector<std::int64_t>& strides() const;
    bool requires_grad() const;
    void set_requires_grad(bool flag);

    float* data();
    const float* data() const;

    float item() const;
    float& operator[](std::int64_t idx);
    const float& operator[](std::int64_t idx) const;

    Tensor reshape(const std::vector<std::int64_t>& new_sizes) const;
    Tensor clone() const;

    std::shared_ptr<TensorImpl> impl() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& sizes);
std::int64_t compute_numel(const std::vector<std::int64_t>& sizes);

} // namespace mtorch
