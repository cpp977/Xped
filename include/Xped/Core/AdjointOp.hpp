#ifndef ADJOINT_OP_H_
#define ADJOINT_OP_H_

#include "Xped/Core/Qbasis.hpp"

#include "TensorBase.hpp"

namespace Xped {

template <typename XprType>
class AdjointOp;

template <typename XprType>
struct TensorTraits<AdjointOp<XprType>>
{
    static constexpr std::size_t Rank = XprType::corank();
    static constexpr std::size_t CoRank = XprType::rank();
    typedef typename XprType::Symmetry Symmetry;
    using AllocationPolicy = typename XprType::AllocationPolicy;
    typedef typename XprType::Scalar Scalar;
};

template <typename XprType>
class AdjointOp : public TensorBase<AdjointOp<XprType>>
{
public:
    static inline constexpr std::size_t Rank = XprType::corank();
    static inline constexpr std::size_t CoRank = XprType::rank();
    typedef typename XprType::Scalar Scalar;
    typedef typename XprType::Symmetry Symmetry;
    using AllocationPolicy = typename XprType::AllocationPolicy;
    typedef typename Symmetry::qType qType;

    AdjointOp(XPED_CONST XprType& xpr)
        : refxpr_(xpr)
    {}

    inline const std::string name() const { return "AdjointOp"; }
    constexpr std::size_t rank() const { return refxpr_.corank(); }
    constexpr std::size_t corank() const { return refxpr_.rank(); }

    inline const auto sector() const { return refxpr_.sector(); }
    inline const qType sector(std::size_t i) const { return refxpr_.sector(i); }

    // const std::vector<MatrixType> block() const { return refxpr_block(); }
    // const MatrixType block(std::size_t i) const { return Plain::template adjoint(refxpr_.block(i)); }
    inline const auto block(std::size_t i) const { return PlainInterface::adjoint<Scalar>(refxpr_.block(i)); }
    inline auto block(std::size_t i) { return PlainInterface::adjoint<Scalar>(refxpr_.block(i)); }

    inline const auto dict() const { return refxpr_.dict(); }

    inline const std::shared_ptr<mpi::XpedWorld> world() const { return refxpr_.world(); }

    // inline const std::array<Qbasis<Symmetry, 1, Allocator>, XprType::CoRank> uncoupledDomain() const { return refxpr_.uncoupledCodomain(); }
    // inline const std::array<Qbasis<Symmetry, 1, Allocator>, XprType::Rank> uncoupledCodomain() const { return refxpr_.uncoupledDomain(); }
    inline const auto& uncoupledDomain() const { return refxpr_.uncoupledCodomain(); }
    inline const auto& uncoupledCodomain() const { return refxpr_.uncoupledDomain(); }

    inline const auto& coupledDomain() const { return refxpr_.coupledCodomain(); }
    inline const auto& coupledCodomain() const { return refxpr_.coupledDomain(); }

    inline auto domainTrees(const qType& q) const { return refxpr_.codomainTrees(q); }
    inline auto codomainTrees(const qType& q) const { return refxpr_.domainTrees(q); }

protected:
    XPED_CONST XprType& refxpr_;
};

} // namespace Xped
#endif
