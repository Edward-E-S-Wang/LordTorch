#include "mtorch/nn/loss.h"
#include "mtorch/core/exception.h"
#include "mtorch/ops/creation.h"

namespace mtorch::nn {

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.sizes() != target.sizes()) {
        throw MtorchError("mse_loss requires tensors of the same shape");
    }
    Tensor out = zeros({1});
    float acc = 0.0f;
    for (std::int64_t i = 0; i < pred.numel(); ++i) {
        const float d = pred[i] - target[i];
        acc += d * d;
    }
    out[0] = acc / static_cast<float>(pred.numel());
    return out;
}

} // namespace mtorch::nn
