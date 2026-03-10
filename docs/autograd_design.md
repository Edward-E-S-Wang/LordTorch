# Autograd Design

This scaffold reserves Node / Engine APIs.
A future implementation should:
- attach grad_fn on non-leaf outputs
- topologically sort the graph
- accumulate grads into leaf tensors
