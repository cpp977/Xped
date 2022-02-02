#ifndef XPED_COEFF_UNARY_OP_H_
#define XPED_COEFF_UNARY_OP_H_

#include "Xped/Core/Qbasis.hpp"

#include "TensorBase.hpp"

namespace Xped {

template <typename XprType>
class CoeffUnaryOp;

template <typename XprType>
struct TensorTraits<CoeffUnaryOp<XprType>>
{
    static constexpr std::size_t Rank = XprType::Rank;
    static constexpr std::size_t CoRank = XprType::CoRank;
    typedef typename XprType::Scalar Scalar;
    typedef typename XprType::Symmetry Symmetry;
    typedef typename XprType::PlainLib PlainLib;
    typedef typename XprType::MatrixType MatrixType;
    typedef typename XprType::TensorType TensorType;
    typedef typename XprType::VectorType VectorType;
};

template <typename XprType>
class CoeffUnaryOp : public TensorBase<CoeffUnaryOp<XprType>>
{
public:
    static inline constexpr std::size_t Rank = XprType::Rank;
    static inline constexpr std::size_t CoRank = XprType::CoRank;
    typedef typename XprType::Scalar Scalar;
    typedef typename XprType::Symmetry Symmetry;
    typedef typename Symmetry::qType qType;

    typedef typename XprType::PlainLib PlainLib;
    typedef typename XprType::MatrixType MatrixType;
    typedef typename XprType::TensorType TensorType;
    typedef typename XprType::VectorType VectorType;

    CoeffUnaryOp(XPED_CONST XprType& xpr, const std::function<Scalar(Scalar)>& coeff_func)
        : refxpr_(xpr)
        , coeff_func_(coeff_func)
    {}

    inline const std::string name() const { return "CoeffUnaryOp"; }
    constexpr std::size_t rank() const { return refxpr_.rank(); }
    constexpr std::size_t corank() const { return refxpr_.rank(); }

    inline const std::vector<qType> sector() const { return refxpr_.sector(); }
    inline const qType sector(std::size_t i) const { return refxpr_.sector(i); }

    // const std::vector<MatrixType> block() const { return refxpr_block(); }
    inline const auto block(std::size_t i) const { return refxpr_.block(i).unaryExpr(coeff_func_); }

    inline auto block(std::size_t i) { return refxpr_.block(i).unaryExpr(coeff_func_); }

    inline const std::unordered_map<qType, std::size_t> dict() const { return refxpr_.dict(); }

    inline const std::shared_ptr<mpi::XpedWorld> world() const { return refxpr_.world(); }

    inline const std::array<Qbasis<Symmetry, 1>, Rank> uncoupledDomain() const { return refxpr_.uncoupledDomain(); }
    inline const std::array<Qbasis<Symmetry, 1>, CoRank> uncoupledCodomain() const { return refxpr_.uncoupledCodomain(); }

    inline const Qbasis<Symmetry, Rank> coupledDomain() const { return refxpr_.coupledDomain(); }
    inline const Qbasis<Symmetry, CoRank> coupledCodomain() const { return refxpr_.coupledCodomain(); }

    inline std::vector<FusionTree<Rank, Symmetry>> domainTrees(const qType& q) const { return refxpr_.cdomainTrees(q); }
    inline std::vector<FusionTree<CoRank, Symmetry>> codomainTrees(const qType& q) const { return refxpr_.codomainTrees(q); }

protected:
    XPED_CONST XprType& refxpr_;
    const std::function<Scalar(Scalar)> coeff_func_;
};

} // namespace Xped
#endif
