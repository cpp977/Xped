#ifndef OPTIM_CONTRACTIONS_H_
#define OPTIM_CONTRACTIONS_H_

#include "Core/Xped.hpp"

template <typename Scalar, typename Symmetry, typename MatrixLib, typename TensorLib, typename VectorLib>
void contract_L(XPED_CONST Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Bold,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Bra,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Ket,
                Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Bnew)
{
    Bnew.clear();
    Bnew = Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib, VectorLib>({{Bra.uncoupledCodomain()[0]}}, {{Ket.uncoupledCodomain()[0]}});

    for(std::size_t i = 0; i < Bra.sector().size(); i++) {
        std::size_t dimQ = PlainInterface<MatrixLib, TensorLib, VectorLib>::template cols<Scalar>(Bra.block_[i]);
        typename Symmetry::qType Q = Bra.sector_[i];
        auto itKet = Ket.dict_.find(Q);
        if(itKet == Ket.dict_.end()) { continue; }
        auto Mtmp = PlainInterface<MatrixLib, TensorLib, VectorLib>::template construct_with_zero<Scalar>(dimQ, dimQ);
        // typename Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib>::MatrixType Mtmp(dimQ, dimQ);
        // Mtmp.setZero();
        for(const auto& domainTree : Bra.domainTrees(Q)) {
            FusionTree<1, Symmetry> trivial;
            trivial.q_coupled = Q;
            trivial.q_uncoupled[0] = Q;
            trivial.dims[0] = dimQ;
            trivial.dim = dimQ;
            auto Mbra = Bra.subMatrix(domainTree, trivial);
            auto Mket = Ket.subMatrix(domainTree, trivial);
            auto Qin = domainTree.q_uncoupled[0];
            auto itBold = Bold.dict_.find(Qin);
            if(itBold == Bold.dict_.end()) { continue; }
            for(std::size_t s = 0; s < domainTree.dims[1]; s++) {
                typename PlainInterface<MatrixLib, TensorLib, VectorLib>::template MType<Scalar> Mbrablock =
                    PlainInterface<MatrixLib, TensorLib, VectorLib>::block(Mbra, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ);
                PlainInterface<MatrixLib, TensorLib, VectorLib>::template optimal_prod_add<Scalar>(
                    1.,
                    PlainInterface<MatrixLib, TensorLib, VectorLib>::template adjoint<Scalar>(Mbrablock),
                    Bold.block_[itBold->second],
                    PlainInterface<MatrixLib, TensorLib, VectorLib>::block(Mket, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ),
                    Mtmp);
                // Mtmp += Mbra.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint() * Bold.block_[itBold->second] *
                //         Mket.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ);
            }
        }
        auto it = Bnew.dict_.find(Q);
        if(it == Bnew.dict_.end()) {
            Bnew.push_back(Q, Mtmp);
        } else {
            Bnew.block_[it->second] = PlainInterface<MatrixLib, TensorLib, VectorLib>::template add<Scalar>(Bnew.block_[it->second], Mtmp);
        }
    }
}

template <typename Scalar, typename Symmetry, typename MatrixLib, typename TensorLib, typename VectorLib>
void contract_R(XPED_CONST Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Bold,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Bra,
                XPED_CONST Xped<Scalar, 2, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Ket,
                Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib, VectorLib>& Bnew)
{
    Bnew.clear();
    Bnew = Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib, VectorLib>({{Ket.uncoupledDomain()[0]}}, {{Bra.uncoupledDomain()[0]}});

    for(std::size_t i = 0; i < Ket.sector().size(); i++) {
        std::size_t dimQ = PlainInterface<MatrixLib, TensorLib, VectorLib>::template cols<Scalar>(Ket.block_[i]);
        typename Symmetry::qType Q = Ket.sector_[i];
        auto itBold = Bold.dict_.find(Q);
        if(itBold == Bold.dict_.end()) { continue; }
        auto itBra = Bra.dict_.find(Q);
        if(itBra == Bra.dict_.end()) { continue; }
        for(const auto& domainTree : Ket.domainTrees(Q)) {
            FusionTree<1, Symmetry> trivial;
            trivial.q_coupled = Q;
            trivial.q_uncoupled[0] = Q;
            trivial.dims[0] = dimQ;
            trivial.dim = dimQ;

            auto Qin = domainTree.q_uncoupled[0];
            auto Mtmp = PlainInterface<MatrixLib, TensorLib, VectorLib>::template construct_with_zero<Scalar>(domainTree.dims[0], domainTree.dims[0]);
            // typename Xped<Scalar, 1, 1, Symmetry, MatrixLib, TensorLib>::MatrixType Mtmp(domainTree.dims[0], domainTree.dims[0]);
            // Mtmp.setZero();

            auto Mbra = Bra.subMatrix(domainTree, trivial);
            auto Mket = Ket.subMatrix(domainTree, trivial);
            for(std::size_t s = 0; s < domainTree.dims[1]; s++) {
                typename PlainInterface<MatrixLib, TensorLib, VectorLib>::template MType<Scalar> Mbrablock =
                    PlainInterface<MatrixLib, TensorLib, VectorLib>::block(Mbra, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ);
                PlainInterface<MatrixLib, TensorLib, VectorLib>::template optimal_prod_add<Scalar>(
                    1.,
                    PlainInterface<MatrixLib, TensorLib, VectorLib>::block(Mket, s * domainTree.dims[0], 0, domainTree.dims[0], dimQ),
                    Bold.block_[itBold->second],
                    PlainInterface<MatrixLib, TensorLib, VectorLib>::template adjoint<Scalar>(Mbrablock),
                    Mtmp);
                // PlainInterface<MatrixLib, TensorLib, VectorLib>::optimal_prod_add(
                //     1.,
                //     Mket.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ),
                //     Bold.block_[itBold->second],
                //     Mbra.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint(),
                //     Mtmp);
                // Mtmp += Mket.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ) * Bold.block_[itBold->second] *
                //         Mbra.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint();
            }
            // Mtmp *= Symmetry::coeff_rightOrtho(Q, Qin);
            PlainInterface<MatrixLib, TensorLib, VectorLib>::scale(Mtmp, Symmetry::coeff_rightOrtho(Q, Qin));
            auto it = Bnew.dict_.find(Qin);
            if(it == Bnew.dict_.end()) {
                Bnew.push_back(Qin, Mtmp);
            } else {
                Bnew.block_[it->second] = PlainInterface<MatrixLib, TensorLib, VectorLib>::template add<Scalar>(Bnew.block_[it->second], Mtmp);
                // Bnew.block_[it->second] += Mtmp;
            }
        }
    }
}

#endif
