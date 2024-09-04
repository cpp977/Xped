#include "Xped/MPS/MpsContractions.hpp"

#include "spdlog/spdlog.h"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, typename AllocationPolicy>
void contract_L(XPED_CONST Tensor<Scalar, 1, 1, Symmetry, false, AllocationPolicy>& Bold,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, false, AllocationPolicy>& Bra,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, false, AllocationPolicy>& Ket,
                Tensor<Scalar, 1, 1, Symmetry, false, AllocationPolicy>& Bnew)
{
    SPDLOG_INFO("Entering contract_L().");
    Bnew.clear();
    Bnew = Tensor<Scalar, 1, 1, Symmetry, false, AllocationPolicy>({{Bra.uncoupledCodomain()[0]}}, {{Ket.uncoupledCodomain()[0]}}, Bold.world());
    Bnew.setZero();

    for(std::size_t i = 0; i < Bra.sector().size(); i++) {
        std::size_t dimQ = PlainInterface::cols(Bra.block(i));
        typename Symmetry::qType Q = Bra.sector(i);
        auto itKet = Ket.dict().find(Q);
        if(itKet == Ket.dict().end()) { continue; }
        auto Mtmp = PlainInterface::construct_with_zero<Scalar>(dimQ, dimQ, Bold.world());
        for(const auto& domainTree : Bra.domainTrees(Q)) {
            FusionTree<1, Symmetry> trivial;
            trivial.q_coupled = Q;
            trivial.q_uncoupled[0] = Q;
            trivial.dims[0] = dimQ;
            trivial.dim = dimQ;
            auto Mbra = Bra.subMatrix(domainTree, trivial);
            auto Mket = Ket.subMatrix(domainTree, trivial);
            auto Qin = domainTree.q_uncoupled[0];
            auto itBold = Bold.dict().find(Qin);
            if(itBold == Bold.dict().end()) { continue; }
            for(std::size_t s = 0; s < domainTree.dims[1]; s++) {
                typename PlainInterface::MType<Scalar> Mbrablock = PlainInterface::block(Mbra, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ);
                PlainInterface::optimal_prod_add(1.,
                                                 PlainInterface::adjoint(Mbrablock),
                                                 Bold.block(itBold->second),
                                                 PlainInterface::block(Mket, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ),
                                                 Mtmp);
            }
        }
        auto it = Bnew.dict().find(Q);
        Bnew.block(it->second) = PlainInterface::add(Bnew.block(it->second), Mtmp);
    }
    SPDLOG_INFO("Leaving contract_L().");
}

template <typename Scalar, typename Symmetry, typename AllocationPolicy>
void contract_R(XPED_CONST Tensor<Scalar, 1, 1, Symmetry, false, AllocationPolicy>& Bold,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, false, AllocationPolicy>& Bra,
                XPED_CONST Tensor<Scalar, 2, 1, Symmetry, false, AllocationPolicy>& Ket,
                Tensor<Scalar, 1, 1, Symmetry, false, AllocationPolicy>& Bnew)
{
    SPDLOG_INFO("Entering contract_R().");
    Bnew.clear();
    Bnew = Tensor<Scalar, 1, 1, Symmetry, false, AllocationPolicy>({{Ket.uncoupledDomain()[0]}}, {{Bra.uncoupledDomain()[0]}}, Bold.world());
    Bnew.setZero();

    for(std::size_t i = 0; i < Ket.sector().size(); i++) {
        std::size_t dimQ = PlainInterface::cols(Ket.block(i));
        typename Symmetry::qType Q = Ket.sector(i);
        auto itBold = Bold.dict().find(Q);
        if(itBold == Bold.dict().end()) { continue; }
        auto itBra = Bra.dict().find(Q);
        if(itBra == Bra.dict().end()) { continue; }
        for(const auto& domainTree : Ket.domainTrees(Q)) {
            FusionTree<1, Symmetry> trivial;
            trivial.q_coupled = Q;
            trivial.q_uncoupled[0] = Q;
            trivial.dims[0] = dimQ;
            trivial.dim = dimQ;

            auto Qin = domainTree.q_uncoupled[0];
            auto Mtmp = PlainInterface::construct_with_zero<Scalar>(domainTree.dims[0], domainTree.dims[0], Bold.world());

            auto Mbra = Bra.subMatrix(domainTree, trivial);
            auto Mket = Ket.subMatrix(domainTree, trivial);
            for(std::size_t s = 0; s < domainTree.dims[1]; s++) {
                typename PlainInterface::MType<Scalar> Mbrablock = PlainInterface::block(Mbra, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ);
                PlainInterface::optimal_prod_add(1.,
                                                 PlainInterface::block(Mket, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ),
                                                 Bold.block(itBold->second),
                                                 PlainInterface::adjoint(Mbrablock),
                                                 Mtmp);
                // PlainInterface::optimal_prod_add(
                //     1.,
                //     Mket.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ),
                //     Bold.block_[itBold->second],
                //     Mbra.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint(),
                //     Mtmp);
                // Mtmp += Mket.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ) * Bold.block_[itBold->second] *
                //         Mbra.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint();
            }
            PlainInterface::scale(Mtmp, Symmetry::coeff_rightOrtho(Q, Qin));
            auto it = Bnew.dict().find(Qin);
            Bnew.block(it->second) = PlainInterface::add(Bnew.block(it->second), Mtmp);
        }
    }
    SPDLOG_INFO("Leaving contract_R().");
}

} // namespace Xped

#if __has_include("MpsContractions.gen.cpp")
#    include "MpsContractions.gen.cpp"
#endif
