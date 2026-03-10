#include "mtorch/autograd/engine.h"
#include <stdexcept>

namespace mtorch {
class Tensor;
namespace autograd {

void Engine::backward(const Tensor&) {
    throw std::runtime_error("autograd engine is not implemented yet in this scaffold");
}

} // namespace autograd
} // namespace mtorch
