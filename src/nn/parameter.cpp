#include "mtorch/nn/parameter.h"

namespace mtorch::nn {

Parameter::Parameter() = default;
Parameter::Parameter(const Tensor& tensor) : Tensor(tensor) {
    set_requires_grad(true);
}

} // namespace mtorch::nn
