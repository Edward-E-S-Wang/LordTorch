#include "mtorch/optim/optimizer.h"

namespace mtorch::optim {

Optimizer::Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
void Optimizer::zero_grad() {}

} // namespace mtorch::optim
