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
#include "Xped/Core/storage/StorageType.hpp"
#include "Xped/Core/treepair.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/allocators/HeapPolicy.hpp"
#include "Xped/Core/allocators/PmrPolicy.hpp"
#ifdef XPED_USE_AD
#    include "Xped/Core/allocators/StanArenaPolicy.hpp"
#endif

#include "Xped/Core/TensorBase.hpp"
#include "Xped/Core/TensorHelper.hpp"

namespace Xped {

template <typename Scalar_, std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename AllocationPolicy_>
struct TensorTraits<Tensor<Scalar_, Rank_, CoRank_, Symmetry_, AllocationPolicy_>>
{
    static constexpr std::size_t Rank = Rank_;
    static constexpr std::size_t CoRank = CoRank_;
    typedef Scalar_ Scalar;
    typedef Symmetry_ Symmetry;
    typedef typename Symmetry::qType qType;
    using AllocationPolicy = AllocationPolicy_;
};

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry_, typename AllocationPolicy_ = HeapPolicy>
class Tensor : public TensorBase<Tensor<Scalar_, Rank, CoRank, Symmetry_, AllocationPolicy_>>
{
    template <typename Derived>
    friend class TensorBase;

    template <typename Scalar__, std::size_t Rank_, std::size_t CoRank_, typename Symmetry__, typename AllocationPolicy__>
    friend Tensor<Scalar__, Rank_, CoRank_, Symmetry__, AllocationPolicy__>
    operator+(XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, AllocationPolicy__>& T1,
              XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, AllocationPolicy__>& T2);

    template <typename Scalar__, std::size_t Rank_, std::size_t CoRank_, typename Symmetry__, typename AllocationPolicy__>
    friend Tensor<Scalar__, Rank_, CoRank_, Symmetry__, AllocationPolicy__>
    operator-(XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, AllocationPolicy__>& T1,
              XPED_CONST Tensor<Scalar__, Rank_, CoRank_, Symmetry__, AllocationPolicy__>& T2);

public:
    using Scalar = Scalar_;
    using RealScalar = typename ScalarTraits<Scalar>::Real;

    using Symmetry = Symmetry_;
    using qType = typename Symmetry::qType;

    using AllocationPolicy = AllocationPolicy_;

    using IndexType = PlainInterface::Indextype;
    using VectorType = PlainInterface::VType<Scalar>;
    using MatrixType = PlainInterface::MType<Scalar>;
    using MatrixMapType = PlainInterface::MapMType<Scalar>;
    using MatrixcMapType = PlainInterface::cMapMType<Scalar>;
    using TensorType = PlainInterface::TType<Scalar, Rank + CoRank>;
    using TensorMapType = PlainInterface::MapTType<Scalar, Rank + CoRank>;
    using TensorcMapType = PlainInterface::cMapTType<Scalar, Rank + CoRank>;

private:
    using Storage = StorageType<Scalar, Rank, CoRank, Symmetry, AllocationPolicy>;

    using Self = Tensor<Scalar, Rank, CoRank, Symmetry, AllocationPolicy>;

    using DictType = std::unordered_map<qType,
                                        std::size_t,
                                        std::hash<qType>,
                                        std::equal_to<qType>,
                                        typename AllocationPolicy::template Allocator<std::pair<const qType, std::size_t>>>;

public:
    /**Does nothing.*/
    Tensor(){};

    // Xped(const Xped& other) = default;
    // Xped(Xped&& other) = default;

    Tensor(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& basis_domain,
           const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& basis_codomain,
           mpi::XpedWorld& world = mpi::getUniverse())
        : storage_(basis_domain, basis_codomain, world)
        , world_(&world, mpi::TrivialDeleter<mpi::XpedWorld>{})
    {}

    Tensor(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& basis_domain,
           const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& basis_codomain,
           const std::shared_ptr<mpi::XpedWorld>& world)
        : storage_(basis_domain, basis_codomain, *world)
        , world_(world)
    {}

    template <typename OtherDerived>
    inline Tensor(const TensorBase<OtherDerived>& other);

    static constexpr std::size_t rank() { return Rank; }
    static constexpr std::size_t corank() { return CoRank; }

    inline const auto& sector() const { return storage_.sector(); }
    inline const qType sector(std::size_t i) const { return storage_.sector(i); }

    // inline const std::vector<MatrixType> block() const { return block_; }
    typename Storage::ConstMatrixReturnType block(std::size_t i) const { return storage_.block(i); }
    typename Storage::MatrixReturnType block(std::size_t i) { return storage_.block(i); }

    const std::size_t plainSize() const { return storage_.data().size(); }

    const DictType& dict() const { return storage_.dict(); }

    const Storage& storage() const { return storage_; }

    const std::shared_ptr<mpi::XpedWorld> world() const { return world_; }

    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& uncoupledDomain() const { return storage_.uncoupledDomain(); }
    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& uncoupledCodomain() const { return storage_.uncoupledCodomain(); }

    const Qbasis<Symmetry, Rank, AllocationPolicy>& coupledDomain() const { return storage_.coupledDomain(); }
    const Qbasis<Symmetry, CoRank, AllocationPolicy>& coupledCodomain() const { return storage_.coupledCodomain(); }

    typename Storage::ConstMatrixReturnType operator()(const qType& q_coupled) const { return storage_.block(q_coupled); }

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

    const MatrixType subMatrix(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;
    // MatrixType& operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);

    void print(std::ostream& o, bool PRINT_MATRICES = true) const;

    void setRandom();
    void setZero();
    void setIdentity();
    void setConstant(const Scalar& val);

    void clear() { storage_.clear(); }

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
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, AllocationPolicy> permute() const;

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, AllocationPolicy>,
               Tensor<RealScalar, 1, 1, Symmetry, AllocationPolicy>,
               Tensor<Scalar, 1, CoRank, Symmetry, AllocationPolicy>>
    tSVD(std::size_t maxKeep,
         RealScalar eps_svd,
         RealScalar& truncWeight,
         RealScalar& entropy,
         std::map<qarray<Symmetry::Nq>, VectorType>& SVspec,
         bool PRESERVE_MULTIPLETS = true,
         bool RETURN_SPEC = true) XPED_CONST;

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, AllocationPolicy>,
               Tensor<RealScalar, 1, 1, Symmetry, AllocationPolicy>,
               Tensor<Scalar, 1, CoRank, Symmetry, AllocationPolicy>>
    tSVD(std::size_t maxKeep, RealScalar eps_svd, RealScalar& truncWeight, bool PRESERVE_MULTIPLETS = true) XPED_CONST
    {
        RealScalar S_dumb;
        std::map<qarray<Symmetry::Nq>, VectorType> SVspec_dumb;
        return tSVD(maxKeep, eps_svd, truncWeight, S_dumb, SVspec_dumb, PRESERVE_MULTIPLETS, false); // false: Dont return singular value spectrum
    }

    const auto& domainTrees(const qType& q) const { return coupledDomain().tree(q); }
    const auto& codomainTrees(const qType& q) const { return coupledCodomain().tree(q); }

    void push_back(const qType& q, const MatrixType& M) { storage_.push_back(q, M); }

private:
    Storage storage_;

    std::shared_ptr<mpi::XpedWorld> world_;

    template <std::size_t... p_domain, std::size_t... p_codomain>
    Self permute_impl(seq::iseq<std::size_t, p_domain...> pd, seq::iseq<std::size_t, p_codomain...> pc) const;

    template <int shift, std::size_t... ps>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, AllocationPolicy> permute_impl(seq::iseq<std::size_t, ps...> per) const;
};

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <typename OtherDerived>
Tensor<Scalar_, Rank, CoRank, Symmetry, AllocationPolicy>::Tensor(const TensorBase<OtherDerived>& other)
{
    storage_ = Storage(other.derived().uncoupledDomain(), other.derived().uncoupledCodomain(), *other.derived().world());
    storage_.reserve(other.derived().sector().size());
    for(std::size_t i = 0; i < other.derived().sector().size(); ++i) { storage_.push_back(other.derived().sector(i), other.derived().block(i)); }
    world_ = other.derived().world();
}

#ifdef XPED_USE_AD
template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
using ArenaTensor = Tensor<Scalar, Rank, CoRank, Symmetry, StanArenaPolicy>;
#endif

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/Tensor.cpp"
#endif

#endif
