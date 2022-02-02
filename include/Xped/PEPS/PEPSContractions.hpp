#ifndef XPED_PEPS_CONTRACTIONS_H_
#define XPED_PEPS_CONTRACTIONS_H_

#include <tuple>

#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Derived1, typename Derived2>
std::pair<Tensor<typename TensorTraits<Derived1>::Scalar, 1, 3, typename TensorTraits<Derived1>::Symmetry>,
          Tensor<typename TensorTraits<Derived1>::Scalar, 3, 1, typename TensorTraits<Derived1>::Symmetry>>
decompose(const TensorBase<Derived1>& T1, const TensorBase<Derived2>& T2, const std::size_t max_nsv);
}

#ifndef XPED_COMPILED_LIB
#    include "PEPS/PEPSContractions.cpp"
#endif

#endif
