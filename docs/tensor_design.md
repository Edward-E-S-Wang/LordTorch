# Tensor Design

Tensor is a lightweight handle around shared TensorImpl.
TensorImpl owns metadata such as shape, strides, storage offset,
and optional autograd state.
Storage owns raw contiguous memory.
