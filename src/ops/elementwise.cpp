#include "mtorch/ops/elementwise.h"
#include "mtorch/core/exception.h"
#include "mtorch/ops/creation.h"

namespace mtorch {

namespace {
using BinaryFn = float(*)(float, float);

Tensor binary_op(const Tensor& a, const Tensor& b, BinaryFn fn) {
    if (a.sizes() != b.sizes()) {
        throw MtorchError("binary op requires same shape in this scaffold");
    }
    Tensor out = zeros(a.sizes());
    for (std::int64_t i = 0; i < a.numel(); ++i) {
        out[i] = fn(a[i], b[i]);
    }
    return out;
}

float addf(float x, float y) { return x + y; }
float subf(float x, float y) { return x - y; }
float mulf(float x, float y) { return x * y; }
float divf(float x, float y) { return x / y; }
} // namespace

Tensor add(const Tensor& a, const Tensor& b) { return binary_op(a, b, addf); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary_op(a, b, subf); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary_op(a, b, mulf); }
Tensor div(const Tensor& a, const Tensor& b) { return binary_op(a, b, divf); }

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }

} // namespace mtorch
