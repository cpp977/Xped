#ifndef OPTIM_CONTRACTIONS_H_
#define OPTIM_CONTRACTIONS_H_

#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, typename PlainLib>
void contract_L(XPED_CONST Tensor<Scalar, 1, 1, Symmetry, PlainLib>& Bold,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, PlainLib>& Bra,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, PlainLib>& Ket,
                Tensor<Scalar, 1, 1, Symmetry, PlainLib>& Bnew);

template <typename Scalar, typename Symmetry, typename PlainLib>
void contract_R(XPED_CONST Tensor<Scalar, 1, 1, Symmetry, PlainLib>& Bold,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, PlainLib>& Bra,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, PlainLib>& Ket,
                Tensor<Scalar, 1, 1, Symmetry, PlainLib>& Bnew);

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "MPS/MpsContractions.cpp"
#endif

#endif
