#include <iostream>
#include "mtorch/mtorch.h"

int main() {
    using namespace mtorch;

    Tensor a = tensor({1.f, 2.f, 3.f, 4.f}, {2, 2});
    Tensor b = ones({2, 2});
    Tensor c = a + b;

    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "c = a + b = " << c << "\n";
    std::cout << "sum(c) = " << sum(c).item() << "\n";
    std::cout << "mean(c) = " << mean(c).item() << "\n";
    return 0;
}
