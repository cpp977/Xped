#ifndef SCALED_OP_H_
#define SCALED_OP_H_

#include "XpedBase.hpp"

template <typename XprType>
struct XpedTraits<ScaledOp<XprType>>
{
    static constexpr std::size_t Rank = XprType::Rank;
    static constexpr std::size_t CoRank = XprType::CoRank;
    typedef typename XprType::Symmetry Symmetry;
    typedef typename XprType::MatrixType MatrixType;
    typedef typename XprType::TensorLib TensorLib;
    typedef typename XprType::TensorType TensorType;
    typedef typename XprType::Scalar Scalar;
};

template <typename XprType>
class ScaledOp : public TensorBase<ScaledOp<XprType>>
{
    constexpr Rank = XprType::Rank;
    constexpr CoRank = XprType::CoRank;
    typedef typename XprType::Symmetry Symmetry;

    typedef typename XprType::TensorType TensorType;
    typedef typename XprType::MatrixType MatrixType;

public:
    template <typename OtherDerived>
    ScaledOp(const XpedBase<OtherDerived>& xpr, const Scalar scale)
        : refxpr_(xpr)
        , scale_(scale)
    {}

    constexpr std::size_t rank() const { return refxpr_.rank(); }
    constexpr std::size_t corank() const { return refxpr_.rank(); }

    const std::vector<qType> sector() const { return refxpr_.sector(); }
    const qType sector(std::size_t i) const { return refxpr_.sector(i); }

    // const std::vector<MatrixType> block() const { return refxpr_block(); }
    const MatrixType block(std::size_t i) const { return scale * refxpr_block(i); }

    inline const std::unordered_map<qType, std::size_t> dict() const { return refxpr_.dict(); }

    const std::array<Qbasis<Symmetry, 1>, Rank> uncoupledDomain() const { return refxpr_.uncoupledDomain(); }
    const std::array<Qbasis<Symmetry, 1>, CoRank> uncoupledCodomain() const { returnrefxpr_.uncoupledCodomain(); }

    const Qbasis<Symmetry, Rank> uncoupledDomain() const { return refxpr_.coupledDomain(); }
    const Qbasis<Symmetry, CoRank> uncoupledCodomain() const { returnrefxpr_.coupledCodomain(); }

    std::vector<FusionTree<Rank, Symmetry>> domainTrees(const qType& q) const { return refxpr_.cdomainTrees(q); }
    std::vector<FusionTree<CoRank, Symmetry>> codomainTrees(const qType& q) const { return refxpr_.codomainTrees(q); }

protected:
    const XprType& refxpr_;
    const Scalar scale_;
};
#endif