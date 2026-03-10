#pragma once

namespace mtorch {
class Tensor;

namespace autograd {

class Engine {
public:
    static void backward(const Tensor& loss);
};

} // namespace autograd
} // namespace mtorch
