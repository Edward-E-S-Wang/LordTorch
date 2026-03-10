#pragma once

#include <cstddef>
#include <memory>
#include "mtorch/core/scalar_type.h"

namespace mtorch {

class Storage {
public:
    Storage(std::size_t numel, ScalarType dtype = ScalarType::Float32);

    float* data();
    const float* data() const;
    std::size_t numel() const;

private:
    std::shared_ptr<float[]> buffer_;
    std::size_t numel_ = 0;
    ScalarType dtype_ = ScalarType::Float32;
};

} // namespace mtorch
