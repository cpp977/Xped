#ifndef XPED_DIAG_COEFF_BINARY_OP_H_
#define XPED_DIAG_COEFF_BINARY_OP_H_

#include "Xped/Core/Qbasis.hpp"

#include "TensorBase.hpp"

namespace Xped {

template <typename XprTypeLeft, typename XprTypeRight>
class DiagCoeffBinaryOp;

template <typename XprTypeLeft, typename XprTypeRight>
struct TensorTraits<DiagCoeffBinaryOp<XprTypeLeft, XprTypeRight>>
{
    static constexpr std::size_t Rank = XprTypeLeft::Rank;
    static constexpr std::size_t CoRank = XprTypeLeft::CoRank;
    typedef typename XprTypeLeft::Scalar Scalar;
    typedef typename XprTypeLeft::Symmetry Symmetry;
    using AllocationPolicy = typename XprTypeLeft::AllocationPolicy;
};

template <typename XprTypeLeft, typename XprTypeRight>
class DiagCoeffBinaryOp : public TensorBase<DiagCoeffBinaryOp<XprTypeLeft, XprTypeRight>>
{
public:
    static inline constexpr std::size_t Rank = XprTypeLeft::Rank;
    static inline constexpr std::size_t CoRank = XprTypeLeft::CoRank;
    typedef typename XprTypeLeft::Scalar Scalar;
    typedef typename XprTypeLeft::Symmetry Symmetry;
    using AllocationPolicy = typename XprTypeLeft::AllocationPolicy;
    typedef typename Symmetry::qType qType;

    DiagCoeffBinaryOp(XPED_CONST XprTypeLeft& xpr_l, XPED_CONST XprTypeRight& xpr_r, const std::function<Scalar(Scalar, Scalar)>& coeff_func)
        : refxpr_l_(xpr_l)
        , refxpr_r_(xpr_r)
        , coeff_func_(coeff_func)
    {
        static_assert(XprTypeLeft::Rank == XprTypeRight::Rank);
        static_assert(XprTypeLeft::CoRank == XprTypeRight::CoRank);
        static_assert(std::is_same<typename XprTypeLeft::Symmetry, typename XprTypeLeft::Symmetry>::value);
        assert(refxpr_l_.sector() == refxpr_r_.sector());
        assert(refxpr_l_.dict() == refxpr_r_.dict());
        assert(refxpr_l_.world() == refxpr_r_.world());
        assert(refxpr_l_.uncoupledDomain() == refxpr_r_.uncoupledDomain());
        assert(refxpr_l_.uncoupledCodomain() == refxpr_r_.uncoupledCodomain());
        assert(refxpr_l_.coupledDomain() == refxpr_r_.coupledDomain());
        assert(refxpr_l_.coupledCodomain() == refxpr_r_.coupledCodomain());
    }

    inline const std::string name() const { return "CoeffBinaryOp"; }
    constexpr std::size_t rank() const { return refxpr_l_.rank(); }
    constexpr std::size_t corank() const { return refxpr_l_.corank(); }

    inline const auto sector() const { return refxpr_l_.sector(); }
    inline const qType sector(std::size_t i) const { return refxpr_l_.sector(i); }

    inline const auto block(std::size_t i) const { return PlainInterface::diagBinaryFunc(refxpr_l_.block(i), refxpr_r_.block(i), coeff_func_); }
    inline auto block(std::size_t i) { return PlainInterface::diagBinaryFunc(refxpr_l_.block(i), refxpr_r_.block(i), coeff_func_); }

    inline const auto dict() const { return refxpr_l_.dict(); }

    inline const mpi::XpedWorld& world() const { return refxpr_l_.world(); }

    inline const auto uncoupledDomain() const { return refxpr_l_.uncoupledDomain(); }
    inline const auto uncoupledCodomain() const { return refxpr_l_.uncoupledCodomain(); }

    inline const auto coupledDomain() const { return refxpr_l_.coupledDomain(); }
    inline const auto coupledCodomain() const { return refxpr_l_.coupledCodomain(); }

    inline auto domainTrees(const qType& q) const { return refxpr_l_.cdomainTrees(q); }
    inline auto codomainTrees(const qType& q) const { return refxpr_l_.codomainTrees(q); }

protected:
    XPED_CONST XprTypeLeft& refxpr_l_;
    XPED_CONST XprTypeRight& refxpr_r_;
    const std::function<Scalar(Scalar, Scalar)> coeff_func_;
};

} // namespace Xped
#endif
