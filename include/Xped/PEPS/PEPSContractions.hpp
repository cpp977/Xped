#ifndef XPED_PEPS_CONTRACTIONS_H_
#define XPED_PEPS_CONTRACTIONS_H_

#include <tuple>

#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Derived1, typename Derived2>
auto
// std::pair<Tensor<typename TensorTraits<Derived1>::Scalar, 1, 3, typename TensorTraits<Derived1>::Symmetry>,
//           Tensor<typename TensorTraits<Derived1>::Scalar, 3, 1, typename TensorTraits<Derived1>::Symmetry>>
decompose(XPED_CONST TensorBase<Derived1>& T1, XPED_CONST TensorBase<Derived2>& T2, const std::size_t max_nsv);

template <typename Scalar, typename Symmetry, typename AllocationPolicy>
std::pair<Tensor<Scalar, 1, 3, Symmetry, true, AllocationPolicy>, Tensor<Scalar, 3, 1, Symmetry, true, AllocationPolicy>>
decompose(XPED_CONST Tensor<Scalar, 3, 3, Symmetry, true, AllocationPolicy>& T1,
          XPED_CONST Tensor<Scalar, 3, 3, Symmetry, true, AllocationPolicy>& T2,
          const std::size_t max_nsv);

template <typename Scalar, typename Symmetry, typename AllocationPolicy>
std::pair<Tensor<Scalar, 1, 2, Symmetry, true, AllocationPolicy>, Tensor<Scalar, 2, 1, Symmetry, true, AllocationPolicy>>
decompose(XPED_CONST Tensor<Scalar, 2, 2, Symmetry, true, AllocationPolicy>& T1,
          XPED_CONST Tensor<Scalar, 2, 2, Symmetry, true, AllocationPolicy>& T2,
          const std::size_t max_nsv);

template <typename Scalar, typename Symmetry, typename AllocationPolicy, typename DerivedL, typename DerivedT, typename DerivedR, typename DerivedB>
Tensor<Scalar, 2, 3, Symmetry, false, AllocationPolicy> applyWeights(XPED_CONST Tensor<Scalar, 2, 3, Symmetry, false, AllocationPolicy>& A,
                                                                     XPED_CONST TensorBase<DerivedL>& wL,
                                                                     XPED_CONST TensorBase<DerivedT>& wT,
                                                                     XPED_CONST TensorBase<DerivedR>& wR,
                                                                     XPED_CONST TensorBase<DerivedB>& wB);
} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/PEPSContractions.cpp"
#endif

#endif
