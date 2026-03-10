#pragma once

#include "mtorch/core/device.h"
#include "mtorch/core/scalar_type.h"

namespace mtorch {

class TensorOptions {
public:
    TensorOptions& dtype(ScalarType value) { dtype_ = value; return *this; }
    TensorOptions& device(Device value) { device_ = value; return *this; }
    TensorOptions& requires_grad(bool value) { requires_grad_ = value; return *this; }

    ScalarType dtype() const { return dtype_; }
    Device device() const { return device_; }
    bool requires_grad() const { return requires_grad_; }

private:
    ScalarType dtype_ = ScalarType::Float32;
    Device device_{};
    bool requires_grad_ = false;
};

} // namespace mtorch
