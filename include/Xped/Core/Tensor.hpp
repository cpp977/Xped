#ifndef XPED_H_
#define XPED_H_

#include <string>
#include <vector>

#include "seq/seq.h"

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"

#include "Xped/Core/FusionTree.hpp"
#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/ScalarTraits.hpp"
#include "Xped/Core/TensorTypedefs.hpp"
#include "Xped/Core/treepair.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/TensorBase.hpp"
#include "Xped/Core/TensorHelper.hpp"

namespace Xped {

template <typename Scalar_, std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename PlainLib_>
struct TensorTraits<Tensor<Scalar_, Rank_, CoRank_, Symmetry_, PlainLib_>>
{
    static constexpr std::size_t Rank = Rank_;
    static constexpr std::size_t CoRank = CoRank_;
    typedef Scalar_ Scalar;
    typedef Symmetry_ Symmetry;
    typedef typename Symmetry::qType qType;
    typedef PlainLib_ PlainLib;
    typedef typename PlainLib::template MType<Scalar> MatrixType;
    typedef typename PlainLib::template TType<Scalar, Rank + CoRank> TensorType;
    typedef typename PlainLib::template VType<Scalar> VectorType;
};

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry_, typename PlainLib_ = XPED_DEFAULT_PLAININTERFACE>
class Tensor : public TensorBase<Tensor<Scalar_, Rank, CoRank, Symmetry_, PlainLib_>>
{
    template <typename Derived>
    friend class TensorBase;

    template <typename Scalar__, std::size_t Rank_, std::size_t CoRank_, typename Symmetry__, typename PlainLib__>
    friend Tensor<Scalar__, Rank_, CoRank_, Symmetry__, PlainLib__>
    operator+(XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, PlainLib__>& T1,
              XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, PlainLib__>& T2);

    template <typename Scalar__, std::size_t Rank_, std::size_t CoRank_, typename Symmetry__, typename PlainLib__>
    friend Tensor<Scalar__, Rank_, CoRank_, Symmetry__, PlainLib__>
    operator-(XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, PlainLib__>& T1,
              XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, PlainLib__>& T2);

public:
    typedef Scalar_ Scalar;
    typedef typename ScalarTraits<Scalar>::Real RealScalar;

    typedef Symmetry_ Symmetry;
    typedef typename Symmetry::qType qType;

    typedef PlainLib_ PlainLib;

    typedef PlainLib Plain;
    typedef typename Plain::Indextype IndexType;
    typedef typename Plain::template VType<Scalar> VectorType;
    typedef typename Plain::template MType<Scalar> MatrixType;
    typedef typename Plain::template MapMType<Scalar> MatrixMapType;
    typedef typename Plain::template cMapMType<Scalar> MatrixcMapType;
    typedef typename Plain::template TType<Scalar, Rank + CoRank> TensorType;
    typedef typename Plain::template MapTType<Scalar, Rank + CoRank> TensorMapType;
    typedef typename Plain::template cMapTType<Scalar, Rank + CoRank> TensorcMapType;

    typedef Tensor<Scalar, Rank, CoRank, Symmetry, PlainLib> Self;

    /**Does nothing.*/
    Tensor(){};

    // Xped(const Xped& other) = default;
    // Xped(Xped&& other) = default;

    Tensor(const std::array<Qbasis<Symmetry, 1>, Rank> basis_domain,
           const std::array<Qbasis<Symmetry, 1>, CoRank> basis_codomain,
           mpi::XpedWorld& world = mpi::getUniverse());

    template <typename OtherDerived>
    inline Tensor(const TensorBase<OtherDerived>& other);

    static constexpr std::size_t rank() { return Rank; }
    static constexpr std::size_t corank() { return CoRank; }

    inline const std::vector<qType> sector() const { return sector_; }
    inline const qType sector(std::size_t i) const { return sector_[i]; }

    // inline const std::vector<MatrixType> block() const { return block_; }
    const MatrixType& block(std::size_t i) const { return block_[i]; }
    MatrixType& block(std::size_t i) { return block_[i]; }

    const std::unordered_map<qType, std::size_t> dict() const { return dict_; }

    const std::shared_ptr<mpi::XpedWorld> world() const { return world_; }

    const std::array<Qbasis<Symmetry, 1>, Rank> uncoupledDomain() const { return uncoupled_domain; }
    const std::array<Qbasis<Symmetry, 1>, CoRank> uncoupledCodomain() const { return uncoupled_codomain; }

    const Qbasis<Symmetry, Rank> coupledDomain() const { return domain; }
    const Qbasis<Symmetry, CoRank> coupledCodomain() const { return codomain; }

    const MatrixType operator()(const qType& q_coupled) const
    {
        auto it = dict_.find(q_coupled);
        assert(it != dict_.end());
        return block_[it->second];
    }
    const std::string name() const { return "Xped"; }
    // Eigen::TensorMap<TensorType> operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);
    // Eigen::TensorMap<TensorType> operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;

    // Eigen::TensorMap<TensorType> view(const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);
    // auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;

    // auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2, std::size_t block_number);

    auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2);
    auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2, std::size_t block_number);

    auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;
    auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2, std::size_t block_number) const;

    TensorType subBlock(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;
    TensorType subBlock(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2, std::size_t block_number) const;

    MatrixType subMatrix(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;
    // MatrixType& operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);

    void print(std::ostream& o, bool PRINT_MATRICES = true) XPED_CONST;

    void setRandom();
    void setZero();
    void setIdentity();
    void setConstant(const Scalar& val);

    void clear()
    {
        block_.clear();
        dict_.clear();
        sector_.clear();
    }

    // Apply the basis transformation of domain and codomain to the block matrices to get a plain array/tensor
    TensorType plainTensor() const;
    // MatrixType plainMatrix() const;

    // Tensor<CoRank, Rank, Symmetry, MatrixLib_, TensorLib_> adjoint() const;
    // AdjointTensor<self> adjoint() const;

    // self conjugate() const;
    // Tensor<CoRank, Rank, Symmetry, MatrixLib_, TensorLib_> transpose() const;

    // Scalar trace() const;

    // Scalar squaredNorm() const { return (*this * this->adjoint()).trace(); }

    // Scalar norm() const { return std::sqrt(squaredNorm()); }

    template <int shift, std::size_t...>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, PlainLib_> permute() const;

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, PlainLib_>,
               Tensor<RealScalar, 1, 1, Symmetry, PlainLib_>,
               Tensor<Scalar, 1, CoRank, Symmetry, PlainLib_>>
    tSVD(size_t maxKeep,
         RealScalar eps_svd,
         RealScalar& truncWeight,
         RealScalar& entropy,
         std::map<qarray<Symmetry::Nq>, VectorType>& SVspec,
         bool PRESERVE_MULTIPLETS = true,
         bool RETURN_SPEC = true) XPED_CONST;

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, PlainLib_>,
               Tensor<RealScalar, 1, 1, Symmetry, PlainLib_>,
               Tensor<Scalar, 1, CoRank, Symmetry, PlainLib_>>
    tSVD(size_t maxKeep, RealScalar eps_svd, RealScalar& truncWeight, bool PRESERVE_MULTIPLETS = true) XPED_CONST
    {
        RealScalar S_dumb;
        std::map<qarray<Symmetry::Nq>, VectorType> SVspec_dumb;
        return tSVD(maxKeep, eps_svd, truncWeight, S_dumb, SVspec_dumb, PRESERVE_MULTIPLETS, false); // false: Dont return singular value spectrum
    }

    std::vector<FusionTree<Rank, Symmetry>> domainTrees(const qType& q) const { return domain.tree(q); }
    std::vector<FusionTree<CoRank, Symmetry>> codomainTrees(const qType& q) const { return codomain.tree(q); }

    // private:
    std::vector<MatrixType> block_;

    std::unordered_map<qType, std::size_t> dict_; // sector --> number
    std::vector<qType> sector_;

    std::array<Qbasis<Symmetry, 1>, Rank> uncoupled_domain;
    std::array<Qbasis<Symmetry, 1>, CoRank> uncoupled_codomain;
    Qbasis<Symmetry, Rank> domain;
    Qbasis<Symmetry, CoRank> codomain;

    std::shared_ptr<mpi::XpedWorld> world_;

    void push_back(const qType& q, const MatrixType& M)
    {
        block_.push_back(M);
        sector_.push_back(q);
        dict_.insert(std::make_pair(q, sector_.size() - 1));
    }

    template <std::size_t... p_domain, std::size_t... p_codomain>
    Self permute_impl(seq::iseq<std::size_t, p_domain...> pd, seq::iseq<std::size_t, p_codomain...> pc) const;

    template <int shift, std::size_t... ps>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, PlainLib_> permute_impl(seq::iseq<std::size_t, ps...> per) const;
};

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
template <typename OtherDerived>
Tensor<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::Tensor(const TensorBase<OtherDerived>& other)
{
    sector_ = other.derived().sector();
    block_.resize(sector_.size());
    for(std::size_t i = 0; i < sector_.size(); i++) { block_[i] = other.derived().block(i); }
    dict_ = other.derived().dict();
    world_ = other.derived().world();
    uncoupled_domain = other.derived().uncoupledDomain();
    uncoupled_codomain = other.derived().uncoupledCodomain();
    domain = other.derived().coupledDomain();
    codomain = other.derived().coupledCodomain();
}

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/Tensor.cpp"
#endif

#endif
