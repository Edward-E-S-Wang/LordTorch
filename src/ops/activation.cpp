#include "mtorch/ops/activation.h"
#include <cmath>
#include "mtorch/ops/creation.h"

namespace mtorch {

Tensor relu(const Tensor& x) {
    Tensor out = zeros(x.sizes());
    for (std::int64_t i = 0; i < x.numel(); ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return out;
}

Tensor sigmoid(const Tensor& x) {
    Tensor out = zeros(x.sizes());
    for (std::int64_t i = 0; i < x.numel(); ++i) out[i] = 1.0f / (1.0f + std::exp(-x[i]));
    return out;
}

} // namespace mtorch
