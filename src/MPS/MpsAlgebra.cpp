#include "Xped/MPS/MpsAlgebra.hpp"

#include "spdlog/spdlog.h"

#include "Xped/MPS/Mps.hpp"
#include "Xped/MPS/MpsContractions.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
typename Symmetry::Scalar dot(XPED_CONST Mps<Scalar, Symmetry>& Bra, XPED_CONST Mps<Scalar, Symmetry>& Ket, const DMRG::DIRECTION DIR)
{
    assert(Bra.length() == Ket.length());
    SPDLOG_INFO("Entering dot()");
    if(DIR == DMRG::DIRECTION::RIGHT) {
        Tensor<Scalar, 1, 1, Symmetry> B({{Ket.inBasis(0)}}, {{Bra.inBasis(0)}});
        B.setIdentity();
        Tensor<Scalar, 1, 1, Symmetry> Bnext;
        for(size_t l = 0; l < Bra.length(); l++) {
            // Bnext = (Bra.A.Ac[l].adjoint().eval().template permute<-1, 0, 2, 1>() * B).template permute<+1, 0, 2, 1>() * Ket.A.Ac[l];
            // Bnext = (Bra.A.Ac[l].template permute<+1, 0, 1, 2>().adjoint() * B).template permute<+1, 1, 2, 0>() * Ket.A.Ac[l];
            contract_L(B, Bra.A.Ac[l], Ket.A.Ac[l], Bnext);
            B = Bnext;
            // std::cout << B << std::endl;
            Bnext.clear();
        }
        SPDLOG_INFO("Leaving dot()");
        return B.norm();
    } else {
        Tensor<Scalar, 1, 1, Symmetry> B({{Ket.outBasis(Ket.length() - 1)}}, {{Bra.outBasis(Bra.length() - 1)}});
        B.setIdentity();
        Tensor<Scalar, 1, 1, Symmetry> Bnext;
        for(size_t l = Bra.length() - 1; l > 0; l--) {
            // Bnext = (Ket.A.Ac[l] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[l].adjoint().eval().template permute<-1, 2, 0, 1>());
            // Bnext = (Ket.A.Ac[l] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[l].template permute<+1, 0, 1, 2>().adjoint());
            contract_R(B, Bra.A.Ac[l], Ket.A.Ac[l], Bnext);
            B = Bnext;
            // std::cout << B << std::endl;
            Bnext.clear();
        }
        // Bnext = (Ket.A.Ac[0] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[0].adjoint().template permute<-1, 2, 0, 1>());
        // Bnext = (Ket.A.Ac[0] * B).template permute<+1, 0, 1, 2>() * (Bra.A.Ac[0].template permute<+1, 0, 1, 2>().adjoint());
        contract_R(B, Bra.A.Ac[0], Ket.A.Ac[0], Bnext);
        SPDLOG_INFO("Leaving dot()");
        return Bnext.norm();
    }
}

} // namespace Xped

#if __has_include("MpsAlgebra.gen.cpp")
#    include "MpsAlgebra.gen.cpp"
#endif
