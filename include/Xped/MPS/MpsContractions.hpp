#ifndef OPTIM_CONTRACTIONS_H_
#define OPTIM_CONTRACTIONS_H_

#include "Xped/Core/Xped.hpp"

template <typename Scalar, typename Symmetry, typename PlainLib>
void contract_L(XPED_CONST Xped<Scalar, 1, 1, Symmetry, PlainLib>& Bold,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, PlainLib>& Bra,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, PlainLib>& Ket,
                Xped<Scalar, 1, 1, Symmetry, PlainLib>& Bnew);

template <typename Scalar, typename Symmetry, typename PlainLib>
void contract_R(XPED_CONST Xped<Scalar, 1, 1, Symmetry, PlainLib>& Bold,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, PlainLib>& Bra,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, PlainLib>& Ket,
                Xped<Scalar, 1, 1, Symmetry, PlainLib>& Bnew);

#ifndef XPED_COMPILED_LIB
#    include "MPS/MpsContractions.cpp"
#endif

#endif
