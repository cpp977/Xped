#ifndef OPTIM_CONTRACTIONS_H_
#define OPTIM_CONTRACTIONS_H_

#include "Core/Xped.hpp"

template <typename Symmetry, typename MatrixType, typename TensorLib>
void contract_L(const Xped<1, 1, Symmetry, MatrixType, TensorLib>& Bold,
                const Xped<2, 1, Symmetry, MatrixType, TensorLib>& Bra,
                const Xped<2, 1, Symmetry, MatrixType, TensorLib>& Ket,
                Xped<1, 1, Symmetry, MatrixType, TensorLib>& Bnew)
{
    Bnew.clear();
    Bnew = Xped<1, 1, Symmetry, MatrixType, TensorLib>({{Bra.uncoupledCodomain()[0]}}, {{Ket.uncoupledCodomain()[0]}});

    for(std::size_t i = 0; i < Bra.sector().size(); i++) {
        std::size_t dimQ = Bra.block_[i].cols();
        typename Symmetry::qType Q = Bra.sector_[i];
        auto itKet = Ket.dict_.find(Q);
        if(itKet == Ket.dict_.end()) { continue; }
        MatrixType Mtmp(dimQ, dimQ);
        Mtmp.setZero();
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
                Mtmp += Mbra.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint() * Bold.block_[itBold->second] *
                        Mket.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ);
            }
        }
        auto it = Bnew.dict_.find(Q);
        if(it == Bnew.dict_.end()) {
            Bnew.push_back(Q, Mtmp);
        } else {
            Bnew.block_[it->second] += Mtmp;
        }
    }
}

template <typename Symmetry, typename MatrixType, typename TensorLib>
void contract_R(const Xped<1, 1, Symmetry, MatrixType, TensorLib>& Bold,
                const Xped<2, 1, Symmetry, MatrixType, TensorLib>& Bra,
                const Xped<2, 1, Symmetry, MatrixType, TensorLib>& Ket,
                Xped<1, 1, Symmetry, MatrixType, TensorLib>& Bnew)
{
    Bnew.clear();
    Bnew = Xped<1, 1, Symmetry, MatrixType, TensorLib>({{Ket.uncoupledDomain()[0]}}, {{Bra.uncoupledDomain()[0]}});

    for(std::size_t i = 0; i < Ket.sector().size(); i++) {
        std::size_t dimQ = Ket.block_[i].cols();
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
            MatrixType Mtmp(domainTree.dims[0], domainTree.dims[0]);
            Mtmp.setZero();

            auto Mbra = Bra.subMatrix(domainTree, trivial);
            auto Mket = Ket.subMatrix(domainTree, trivial);
            for(std::size_t s = 0; s < domainTree.dims[1]; s++) {
                Mtmp += Mket.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ) * Bold.block_[itBold->second] *
                        Mbra.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint();
            }
            Mtmp *= Symmetry::coeff_rightOrtho(Q, Qin);
            auto it = Bnew.dict_.find(Qin);
            if(it == Bnew.dict_.end()) {
                Bnew.push_back(Qin, Mtmp);
            } else {
                Bnew.block_[it->second] += Mtmp;
            }
        }
    }
}

template <typename Symmetry, typename MatrixType, typename TensorLib>
void contract_L_AA(const Xped<1, 1, Symmetry, MatrixType, TensorLib>& Bold,
                   const Xped<2, 1, Symmetry, MatrixType, TensorLib>& A,
                   Xped<1, 1, Symmetry, MatrixType, TensorLib>& Bnew)
{
    Bnew.clear();
    Bnew = Xped<1, 1, Symmetry, MatrixType, TensorLib>({{A.uncoupledCodomain()[0]}}, {{A.uncoupledCodomain()[0]}});

    for(std::size_t i = 0; i < A.sector().size(); i++) {
        std::size_t dimQ = A.block_[i].cols();
        typename Symmetry::qType Q = A.sector_[i];
        MatrixType Mtmp(dimQ, dimQ);
        Mtmp.setZero();
        for(const auto& domainTree : A.domainTrees(Q)) {
            FusionTree<1, Symmetry> trivial;
            trivial.q_coupled = Q;
            trivial.q_uncoupled[0] = Q;
            trivial.dims[0] = dimQ;
            trivial.dim = dimQ;
            auto M = A.subMatrix(domainTree, trivial);
            auto Qin = domainTree.q_uncoupled[0];
            auto itBold = Bold.dict_.find(Qin);
            if(itBold == Bold.dict_.end()) { continue; }
            for(std::size_t s = 0; s < domainTree.dims[1]; s++) {
                Mtmp += M.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ).adjoint() * Bold.block_[itBold->second] *
                        M.block(s * domainTree.dims[0], 0, domainTree.dims[0], dimQ);
            }
        }
        auto it = Bnew.dict_.find(Q);
        if(it == Bnew.dict_.end()) {
            Bnew.push_back(Q, Mtmp);
        } else {
            Bnew.block_[it->second] += Mtmp;
        }
    }
}

#endif
