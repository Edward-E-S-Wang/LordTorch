#include "mtorch/ops/reduction.h"
#include "mtorch/core/exception.h"
#include "mtorch/ops/creation.h"

namespace mtorch {

Tensor sum(const Tensor& x) {
    Tensor out = zeros({1});
    float acc = 0.0f;
    for (std::int64_t i = 0; i < x.numel(); ++i) acc += x[i];
    out[0] = acc;
    return out;
}

Tensor mean(const Tensor& x) {
    if (x.numel() == 0) {
        throw MtorchError("mean() on empty tensor");
    }
    Tensor out = zeros({1});
    out[0] = sum(x).item() / static_cast<float>(x.numel());
    return out;
}

} // namespace mtorch
