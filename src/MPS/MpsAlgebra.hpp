#ifndef MPS_ALGEBRA_H_
#define MPS_ALGEBRA_H_

#include "MPS/Mps.hpp"

namespace DMRG {
enum class DIRECTION
{
    LEFT = 0,
    RIGHT = 1
};
}

template <typename Symmetry>
typename Symmetry::Scalar dot(const Mps<Symmetry>& Bra, const Mps<Symmetry>& Ket, const DMRG::DIRECTION DIR = DMRG::DIRECTION::RIGHT)
{
    assert(Bra.length() == Ket.length());
    if(DIR == DMRG::DIRECTION::RIGHT) {
        Xped<1, 1, Symmetry> B({{Ket.inBasis(0)}}, {{Bra.inBasis(0)}});
        B.setIdentity();
        Xped<1, 1, Symmetry> Bnext;
        for(size_t l = 0; l < Bra.length(); l++) {
            // Bnext = (Bra.A.Ac[l].adjoint().template permute<-1, 0, 2, 1>() * B).template permute<+1, 0, 2, 1>() * Ket.A.Ac[l];
            Bnext = (Bra.A.Ac[l].template permute<+1, 0, 1, 2>().adjoint() * B).template permute<+1, 1, 2, 0>() * Ket.A.Ac[l];
            B = Bnext;
            // std::cout << B << std::endl;
            Bnext.clear();
        }
        return B.norm();
    } else {
        Xped<1, 1, Symmetry> B({{Ket.outBasis(Ket.length() - 1)}}, {{Bra.outBasis(Bra.length() - 1)}});
        B.setIdentity();
        Xped<1, 1, Symmetry> Bnext;
        for(size_t l = Bra.length() - 1; l > 0; l--) {
            // Bnext = (Ket.A.Ac[l] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[l].adjoint().template permute<-1, 2, 0, 1>());
            Bnext = (Ket.A.Ac[l] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[l].template permute<+1, 0, 1, 2>().adjoint());
            B = Bnext;
            // std::cout << B << std::endl;
            Bnext.clear();
        }
        // Bnext = (Ket.A.Ac[0] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[0].adjoint().template permute<-1, 2, 0, 1>());
        Bnext = (Ket.A.Ac[0] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[0].template permute<+1, 0, 1, 2>().adjoint());
        return Bnext.norm();
    }
}
#endif
