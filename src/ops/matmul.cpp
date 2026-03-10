#include "mtorch/ops/matmul.h"
#include "mtorch/core/exception.h"
#include "mtorch/ops/creation.h"

namespace mtorch {

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2) {
        throw MtorchError("matmul currently supports only 2D tensors");
    }
    const auto m = a.sizes()[0];
    const auto k = a.sizes()[1];
    const auto k2 = b.sizes()[0];
    const auto n = b.sizes()[1];
    if (k != k2) {
        throw MtorchError("matmul inner dimensions do not match");
    }
    Tensor out = zeros({m, n});
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (std::int64_t p = 0; p < k; ++p) {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

} // namespace mtorch
