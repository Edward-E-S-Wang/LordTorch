#include "mtorch/utils/print.h"

namespace mtorch {

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(shape=[";
    for (std::size_t i = 0; i < tensor.sizes().size(); ++i) {
        os << tensor.sizes()[i];
        if (i + 1 != tensor.sizes().size()) os << ", ";
    }
    os << "], values=[";
    for (std::int64_t i = 0; i < tensor.numel(); ++i) {
        os << tensor[i];
        if (i + 1 != tensor.numel()) os << ", ";
    }
    os << "])";
    return os;
}

} // namespace mtorch
