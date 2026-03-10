# API Examples

```cpp
using namespace mtorch;
Tensor x = randn({2, 3}, TensorOptions().requires_grad(true));
Tensor y = sum(x);
```
