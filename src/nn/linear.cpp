#include "mtorch/nn/linear.h"
#include "mtorch/ops/creation.h"
#include "mtorch/ops/matmul.h"

namespace mtorch::nn {

Linear::Linear(std::int64_t in_features, std::int64_t out_features)
    : in_features_(in_features), out_features_(out_features) {
    weight_ = randn({in_features_, out_features_}, TensorOptions().requires_grad(true));
    bias_ = zeros({1, out_features_}, TensorOptions().requires_grad(true));
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
}

Tensor Linear::forward(const Tensor& x) {
    Tensor out = matmul(x, weight_);
    if (x.dim() == 2) {
        for (std::int64_t i = 0; i < x.sizes()[0]; ++i) {
            for (std::int64_t j = 0; j < out_features_; ++j) {
                out[i * out_features_ + j] += bias_[j];
            }
        }
    }
    return out;
}

Tensor Linear::weight() const { return weight_; }
Tensor Linear::bias() const { return bias_; }

} // namespace mtorch::nn
