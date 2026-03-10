#include <cassert>
#include <cmath>
#include <iostream>
#include "mtorch/mtorch.h"

int main() {
    using namespace mtorch;

    Tensor a = zeros({2, 3});
    assert(a.dim() == 2);
    assert(a.numel() == 6);

    for (std::int64_t i = 0; i < a.numel(); ++i) a[i] = static_cast<float>(i + 1);
    Tensor b = a.reshape({3, 2});
    assert(b.numel() == 6);
    assert(std::fabs(b[0] - 1.0f) < 1e-6f);

    Tensor c = a.clone();
    c[0] = 99.0f;
    assert(std::fabs(a[0] - 1.0f) < 1e-6f);
    assert(std::fabs(c[0] - 99.0f) < 1e-6f);

    std::cout << "test_tensor passed\n";
    return 0;
}
