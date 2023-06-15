#ifndef MPS_ALGEBRA_H_
#define MPS_ALGEBRA_H_

#include <cstddef>

#include <boost/describe.hpp>

#include "Xped/Util/Macros.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
class Mps;

namespace DMRG {
BOOST_DEFINE_ENUM_CLASS(DIRECTION, LEFT, RIGHT)
}

template <typename Scalar, typename Symmetry>
typename Symmetry::Scalar
dot(XPED_CONST Mps<Scalar, Symmetry>& Bra, XPED_CONST Mps<Scalar, Symmetry>& Ket, const DMRG::DIRECTION DIR = DMRG::DIRECTION::RIGHT);

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "MPS/MpsAlgebra.cpp"
#endif

#endif
