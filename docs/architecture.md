# Architecture

- core: Tensor, TensorImpl, Storage
- ops: creation / elementwise / reduction / matmul / activation
- autograd: graph node and backward engine placeholders
- nn: Module / Parameter / Linear / Loss
- optim: Optimizer / SGD
