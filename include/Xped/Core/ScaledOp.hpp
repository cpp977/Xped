#ifndef SCALED_OP_H_
#define SCALED_OP_H_

#include "Xped/Core/Qbasis.hpp"

#include "TensorBase.hpp"

template <typename XprType>
class ScaledOp;

template <typename XprType>
struct TensorTraits<ScaledOp<XprType>>
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
class ScaledOp : public TensorBase<ScaledOp<XprType>>
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

    ScaledOp(XPED_CONST XprType& xpr, const Scalar scale)
        : refxpr_(xpr)
        , scale_(scale)
    {}

    inline const std::string name() const { return "ScaledOp"; }
    constexpr std::size_t rank() const { return refxpr_.rank(); }
    constexpr std::size_t corank() const { return refxpr_.rank(); }

    inline const std::vector<qType> sector() const { return refxpr_.sector(); }
    inline const qType sector(std::size_t i) const { return refxpr_.sector(i); }

    // const std::vector<MatrixType> block() const { return refxpr_block(); }
    inline const auto block(std::size_t i) const
    {
        auto res = refxpr_.block(i);
        PlainLib::template scale<Scalar>(res, scale_);
        return res;
    }

    inline auto block(std::size_t i)
    {
        auto res = refxpr_.block(i);
        PlainLib::template scale<Scalar>(res, scale_);
        return res;
    }

    inline const std::unordered_map<qType, std::size_t> dict() const { return refxpr_.dict(); }

    inline const std::shared_ptr<util::mpi::XpedWorld> world() const { return refxpr_.world(); }

    inline const std::array<Qbasis<Symmetry, 1>, Rank> uncoupledDomain() const { return refxpr_.uncoupledDomain(); }
    inline const std::array<Qbasis<Symmetry, 1>, CoRank> uncoupledCodomain() const { return refxpr_.uncoupledCodomain(); }

    inline const Qbasis<Symmetry, Rank> coupledDomain() const { return refxpr_.coupledDomain(); }
    inline const Qbasis<Symmetry, CoRank> coupledCodomain() const { return refxpr_.coupledCodomain(); }

    inline std::vector<FusionTree<Rank, Symmetry>> domainTrees(const qType& q) const { return refxpr_.cdomainTrees(q); }
    inline std::vector<FusionTree<CoRank, Symmetry>> codomainTrees(const qType& q) const { return refxpr_.codomainTrees(q); }

protected:
    XPED_CONST XprType& refxpr_;
    const Scalar scale_;
};
#endif
