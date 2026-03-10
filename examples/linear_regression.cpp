#include <iostream>
#include "mtorch/mtorch.h"

int main() {
    using namespace mtorch;
    using namespace mtorch::nn;

    Tensor x = tensor({1.f, 2.f, 3.f, 4.f}, {2, 2});
    Linear fc(2, 1);
    Tensor pred = fc.forward(x);

    std::cout << "pred = " << pred << "\n";
    std::cout << "This example only demonstrates the forward path.\n";
    std::cout << "Training is pending full autograd + optimizer implementation.\n";
    return 0;
}
