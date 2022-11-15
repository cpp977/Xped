#include "Xped/PEPS/SimpleUpdate.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
std::tuple<Tensor<Scalar, 2, 1, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 1, 2, Symmetry>>
SimpleUpdate<Scalar, Symmetry>::renormalize(const Tensor<Scalar, 2, 2, Symmetry>& bond,
                                            const Tensor<Scalar, 3, 1, Symmetry>&,
                                            const Tensor<Scalar, 1, 3, Symmetry>&,
                                            std::size_t D) const
{
    double dummy;
    return bond.tSVD(D, 1.e-14, dummy, false);
}

} // namespace Xped
