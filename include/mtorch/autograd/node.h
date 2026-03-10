#pragma once

#include <memory>
#include <vector>

namespace mtorch {
class Tensor;

namespace autograd {

class Node {
public:
    virtual ~Node() = default;
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
};

using NodePtr = std::shared_ptr<Node>;

} // namespace autograd
} // namespace mtorch
