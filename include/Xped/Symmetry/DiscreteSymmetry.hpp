#include <array>

#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Scalar, typename InternalSymmetry>
class C4v
{
public:
    std::array<Tensor<Scalar, 4, 4, InternalSymmetry>, 8> RotOps{};
    std::size_t dimension();
    auto Projector();
};

} // namespace Xped
