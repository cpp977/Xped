#ifndef XPED_BLOCK_UNARY_OP_H_
#define XPED_BLOCK_UNARY_OP_H_

#include "Xped/Core/Qbasis.hpp"

#include "TensorBase.hpp"

namespace Xped {

template <typename XprType>
class BlockUnaryOp;

template <typename XprType>
struct TensorTraits<BlockUnaryOp<XprType>>
{
    static constexpr std::size_t Rank = XprType::Rank;
    static constexpr std::size_t CoRank = XprType::CoRank;
    typedef typename XprType::Scalar Scalar;
    typedef typename XprType::Symmetry Symmetry;
    using AllocationPolicy = typename XprType::AllocationPolicy;
};

template <typename XprType>
class BlockUnaryOp : public TensorBase<BlockUnaryOp<XprType>>
{
public:
    static inline constexpr std::size_t Rank = XprType::Rank;
    static inline constexpr std::size_t CoRank = XprType::CoRank;
    typedef typename XprType::Scalar Scalar;
    typedef typename XprType::Symmetry Symmetry;
    using AllocationPolicy = typename XprType::AllocationPolicy;
    typedef typename Symmetry::qType qType;

    BlockUnaryOp(XPED_CONST XprType& xpr, const std::function<MatrixType(const MatrixType&)>& coeff_func)
        : refxpr_(xpr)
        , coeff_func_(coeff_func)
    {}

    inline const std::string name() const { return "BlockUnaryOp"; }
    constexpr std::size_t rank() const { return refxpr_.rank(); }
    constexpr std::size_t corank() const { return refxpr_.corank(); }

    inline const auto sector() const { return refxpr_.sector(); }
    inline const qType sector(std::size_t i) const { return refxpr_.sector(i); }

    inline const auto block(std::size_t i) const { return coeff_func_(refxpr_.block(i)); }
    inline auto block(std::size_t i) { return coeff_func_(refxpr_.block(i)); }

    inline const auto dict() const { return refxpr_.dict(); }

    inline const mpi::XpedWorld& world() const { return refxpr_.world(); }

    inline const auto uncoupledDomain() const { return refxpr_.uncoupledDomain(); }
    inline const auto uncoupledCodomain() const { return refxpr_.uncoupledCodomain(); }

    inline const auto coupledDomain() const { return refxpr_.coupledDomain(); }
    inline const auto coupledCodomain() const { return refxpr_.coupledCodomain(); }

    inline auto domainTrees(const qType& q) const { return refxpr_.cdomainTrees(q); }
    inline auto codomainTrees(const qType& q) const { return refxpr_.codomainTrees(q); }

protected:
    XPED_CONST XprType& refxpr_;
    const std::function<MatrixType(const MatrixType&)> coeff_func_;
};

} // namespace Xped
#endif
