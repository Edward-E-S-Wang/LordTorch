#include <cassert>
#include <cmath>
#include <iostream>
#include "mtorch/mtorch.h"

int main() {
    using namespace mtorch;

    Tensor x = tensor({1.f, 2.f, 3.f, 4.f}, {2, 2});
    assert(std::fabs(sum(x).item() - 10.0f) < 1e-6f);
    assert(std::fabs(mean(x).item() - 2.5f) < 1e-6f);

    std::cout << "test_reduction passed\n";
    return 0;
}
