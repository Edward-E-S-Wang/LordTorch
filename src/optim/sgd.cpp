#include "mtorch/optim/sgd.h"

namespace mtorch::optim {

SGD::SGD(std::vector<Tensor> params, float lr) : Optimizer(std::move(params)), lr_(lr) {}
void SGD::step() {
    (void)lr_;
    // Placeholder: autograd gradients are not wired yet.
}

} // namespace mtorch::optim
