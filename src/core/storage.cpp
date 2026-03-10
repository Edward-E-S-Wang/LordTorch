#include "mtorch/core/storage.h"

namespace mtorch {

Storage::Storage(std::size_t numel, ScalarType dtype)
    : buffer_(numel ? new float[numel]{} : nullptr), numel_(numel), dtype_(dtype) {}

float* Storage::data() { return buffer_.get(); }
const float* Storage::data() const { return buffer_.get(); }
std::size_t Storage::numel() const { return numel_; }

} // namespace mtorch
