#ifndef ADJOINT_OP_H_
#define ADJOINT_OP_H_

#include "XpedBase.hpp"

template <typename XprType>
class AdjointOp;

template <typename XprType>
struct XpedTraits<AdjointOp<XprType>>
{
    static constexpr std::size_t Rank = XprType::corank();
    static constexpr std::size_t CoRank = XprType::rank();
    typedef typename XprType::Symmetry Symmetry;
    typedef typename XprType::MatrixType MatrixType;
    typedef typename XprType::PlainLib PlainLib;
    typedef typename XprType::TensorType TensorType;
    typedef typename XprType::VectorType VectorType;
    typedef typename XprType::Scalar Scalar;
};

template <typename XprType>
class AdjointOp : public XpedBase<AdjointOp<XprType>>
{
public:
    static constexpr std::size_t Rank = XprType::corank();
    static constexpr std::size_t CoRank = XprType::rank();
    typedef typename XprType::Scalar Scalar;
    typedef typename XprType::Symmetry Symmetry;
    typedef typename Symmetry::qType qType;

    typedef typename XprType::PlainLib PlainLib;
    typedef typename XprType::MatrixType MatrixType;
    typedef typename XprType::TensorType TensorType;

    AdjointOp(XPED_CONST XprType& xpr)
        : refxpr_(xpr)
    {}

    const std::string name() const { return "AdjointOp"; }
    constexpr std::size_t rank() const { return refxpr_.corank(); }
    constexpr std::size_t corank() const { return refxpr_.rank(); }

    const std::vector<qType> sector() const { return refxpr_.sector(); }
    const qType sector(std::size_t i) const { return refxpr_.sector(i); }

    // const std::vector<MatrixType> block() const { return refxpr_block(); }
    // const MatrixType block(std::size_t i) const { return Plain::template adjoint(refxpr_.block(i)); }
    const auto block(std::size_t i) const { return PlainLib::template adjoint<Scalar>(refxpr_.block(i)); }
    auto block(std::size_t i) { return PlainLib::template adjoint<Scalar>(refxpr_.block(i)); }

    inline const std::unordered_map<qType, std::size_t> dict() const { return refxpr_.dict(); }

    const std::shared_ptr<util::mpi::XpedWorld> world() const { return refxpr_.world(); }

    const std::array<Qbasis<Symmetry, 1>, Rank> uncoupledDomain() const { return refxpr_.uncoupledCodomain(); }
    const std::array<Qbasis<Symmetry, 1>, CoRank> uncoupledCodomain() const { return refxpr_.uncoupledDomain(); }

    const Qbasis<Symmetry, Rank> coupledDomain() const { return refxpr_.coupledCodomain(); }
    const Qbasis<Symmetry, CoRank> coupledCodomain() const { return refxpr_.coupledDomain(); }

    std::vector<FusionTree<Rank, Symmetry>> domainTrees(const qType& q) const { return refxpr_.codomainTrees(q); }
    std::vector<FusionTree<CoRank, Symmetry>> codomainTrees(const qType& q) const { return refxpr_.domainTrees(q); }

protected:
    XPED_CONST XprType& refxpr_;
};
#endif
