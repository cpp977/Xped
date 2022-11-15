#ifndef XPED_SIMPLE_UPDATE_H_
#define XPED_SIMPLE_UPDATE_H_

#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Scalar_, typename Symmetry_>
class SimpleUpdate
{
public:
    using Scalar = Scalar_;
    using Symmetry = Symmetry_;

    SimpleUpdate() = default;

    std::tuple<Tensor<Scalar, 2, 1, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 1, 2, Symmetry>>
    renormalize(const Tensor<Scalar, 2, 2, Symmetry>& bond,
                const Tensor<Scalar, 3, 1, Symmetry>&,
                const Tensor<Scalar, 1, 3, Symmetry>&,
                std::size_t D) const;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/SimpleUpdate.cpp"
#endif

#endif
