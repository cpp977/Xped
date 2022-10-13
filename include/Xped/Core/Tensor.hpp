#ifndef XPED_TENSOR_HPP_
#define XPED_TENSOR_HPP_

#include <string>
#include <vector>

#include "seq/seq.h"

#include "spdlog/spdlog.h"

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/Util/Macros.hpp"

#include "Xped/Util/Bool.hpp"
#include "Xped/Util/Constfct.hpp"
#include "Xped/Util/Mpi.hpp"

#include "Xped/Core/FusionTree.hpp"
#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/ScalarTraits.hpp"
#include "Xped/Core/TensorTypedefs.hpp"
#include "Xped/Core/storage/StorageType.hpp"
#include "Xped/Core/treepair.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/allocators/HeapPolicy.hpp"
#ifdef XPED_USE_AD
#    include "Xped/Core/allocators/StanArenaPolicy.hpp"
#endif

#include "Xped/Core/TensorBase.hpp"
#include "Xped/Core/TensorHelper.hpp"

namespace Xped {

template <typename, std::size_t, std::size_t, typename, bool = false, typename = HeapPolicy>
class Tensor;

template <typename Scalar_, std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename AllocationPolicy_>
struct TensorTraits<Tensor<Scalar_, Rank_, CoRank_, Symmetry_, false, AllocationPolicy_>>
{
    static constexpr std::size_t Rank = Rank_;
    static constexpr std::size_t CoRank = CoRank_;
    typedef Scalar_ Scalar;
    typedef Symmetry_ Symmetry;
    typedef typename Symmetry::qType qType;
    using AllocationPolicy = AllocationPolicy_;
};

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry_, typename AllocationPolicy_>
class Tensor<Scalar_, Rank, CoRank, Symmetry_, false, AllocationPolicy_>
    : public TensorBase<Tensor<Scalar_, Rank, CoRank, Symmetry_, false, AllocationPolicy_>>
{
    template <typename Derived>
    friend class TensorBase;

    friend class Tensor<Scalar_, Rank, CoRank, Symmetry_, true, AllocationPolicy_>;

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

    using Self = Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>;

    using DictType = std::unordered_map<qType,
                                        std::size_t,
                                        std::hash<qType>,
                                        std::equal_to<qType>,
                                        typename AllocationPolicy::template Allocator<std::pair<const qType, std::size_t>>>;

public:
    /**Does nothing.*/
    Tensor() = default;

    // Tensor(const Tensor& other) = default;
    // Tensor(Tensor&& other) = default;

    Tensor(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& basis_domain,
           const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& basis_codomain,
           const mpi::XpedWorld& world = mpi::getUniverse())
        : storage_(basis_domain, basis_codomain, world)
    {}

    // Tensor(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& basis_domain,
    //        const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& basis_codomain,
    //        mpi::XpedWorld& world)
    //     : storage_(basis_domain, basis_codomain, world)
    // {}

    Tensor(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& basis_domain,
           const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& basis_codomain,
           const Scalar* data,
           std::size_t size,
           const mpi::XpedWorld& world)
        : storage_(basis_domain, basis_codomain, data, size, world)
    {}

    template <typename OtherDerived>
    inline Tensor(const TensorBase<OtherDerived>& other);

    constexpr bool CONTIGUOUS_STORAGE() { return Storage::IS_CONTIGUOUS(); }
    constexpr bool AD_TENSOR() { return false; }

    static constexpr std::size_t rank() { return Rank; }
    static constexpr std::size_t corank() { return CoRank; }

    inline void set_data(const Scalar* data, std::size_t size) { storage_.set_data(data, size); }

    inline const auto& sector() const { return storage_.sector(); }
    inline const qType sector(std::size_t i) const { return storage_.sector(i); }

    typename Storage::ConstMatrixReturnType block(std::size_t i) const { return storage_.block(i); }
    typename Storage::MatrixReturnType block(std::size_t i) { return storage_.block(i); }

    const Scalar* data() const
    {
        static_assert(Storage::IS_CONTIGUOUS());
        return storage_.data();
    }
    Scalar* data()
    {
        static_assert(Storage::IS_CONTIGUOUS());
        return storage_.data();
    }

    const std::size_t plainSize() const { return storage_.plainSize(); }

    const DictType& dict() const { return storage_.dict(); }

    const Storage& storage() const { return storage_; }
    Storage& storage() { return storage_; }

    const mpi::XpedWorld& world() const { return storage_.world(); }

    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& uncoupledDomain() const { return storage_.uncoupledDomain(); }
    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& uncoupledCodomain() const { return storage_.uncoupledCodomain(); }

    const Qbasis<Symmetry, Rank, AllocationPolicy>& coupledDomain() const { return storage_.coupledDomain(); }
    const Qbasis<Symmetry, CoRank, AllocationPolicy>& coupledCodomain() const { return storage_.coupledCodomain(); }

    typename Storage::ConstMatrixReturnType operator()(const qType& q_coupled) const { return storage_.block(q_coupled); }

    const std::string name() const { return "Xped(" + std::to_string(rank()) + "," + std::to_string(corank()) + ")"; }
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

    const auto subMatrix(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
    {
        if(f1.q_coupled != f2.q_coupled) { assert(false); }

        const auto left_offset_domain = coupledDomain().leftOffset(f1);
        const auto left_offset_codomain = coupledCodomain().leftOffset(f2);
        const auto it = dict().find(f1.q_coupled);
        assert(it != dict().end());

        auto submatrix = PlainInterface::block(block(it->second), left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
        // auto submatrix = block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
        return submatrix;
    }

    auto subMatrix(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2)
    {
        if(f1.q_coupled != f2.q_coupled) { assert(false); }

        const auto left_offset_domain = coupledDomain().leftOffset(f1);
        const auto left_offset_codomain = coupledCodomain().leftOffset(f2);
        const auto it = dict().find(f1.q_coupled);
        assert(it != dict().end());

        auto submatrix = PlainInterface::block(block(it->second), left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
        // auto submatrix = block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
        return submatrix;
    }

    // MatrixType& operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);

    void print(std::ostream& o, bool PRINT_MATRICES = false) const;

    void setRandom();
    void setZero();
    void setIdentity();
    void setConstant(Scalar val);

    void clear() { storage_.clear(); }

    static Self Identity(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& basis_domain,
                         const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& basis_codomain,
                         const mpi::XpedWorld& world = mpi::getUniverse())
    {
        Self out(basis_domain, basis_codomain, world);
        out.setIdentity();
        return out;
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

    // template <bool, int shift, std::size_t...>
    // Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute() const;

    // template <bool TRACK, int shift, std::size_t... p>
    // Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute(seq::iseq<std::size_t, p...>) const
    // {
    //     return permute<TRACK, shift, p...>();
    // }

    // template <int shift, std::size_t... p>
    // Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute() const
    // {
    //     return permute<false, shift, p...>();
    // }

    template <int shift, std::size_t... p>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute() const;

    template <int shift, std::size_t... p>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute(seq::iseq<std::size_t, p...>) const
    {
        return permute<shift, p...>();
    }

    template <int shift, std::size_t... p, bool b>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute(Bool<b>) const
    {
        return permute<shift, p...>();
    }

    template <int shift, std::size_t... p, bool TRACK>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute(seq::iseq<std::size_t, p...>, Bool<TRACK>) const
    {
        return permute<shift, p...>(Bool<TRACK>{});
    }

    template <std::size_t leg>
    Tensor<Scalar, util::constFct::trimDim<Rank>(leg), Rank + CoRank - 1 - util::constFct::trimDim<Rank>(leg), Symmetry, false, AllocationPolicy>
    trim() const;

    template <bool = false>
    Self twist(std::size_t leg) const;

    template <std::size_t... legs>
    Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy> shiftQN(qType charge) const;

    // template <std::size_t leg>
    // Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy> shiftQN(qType charge) const;

#if XPED_HAS_NTTP
    template <auto a1, auto a2, std::size_t ResRank, bool TRACK = false, std::size_t OtherRank, std::size_t OtherCoRank, bool ENABLE_AD>
    auto contract(const Tensor<Scalar, OtherRank, OtherCoRank, Symmetry, ENABLE_AD, AllocationPolicy>& other) XPED_CONST
    {
        constexpr auto perms = util::constFct::get_permutations<a1, Rank, a2, OtherRank, ResRank>();
        constexpr auto p1 = std::get<0>(perms);
        constexpr auto shift1 = std::get<1>(perms);
        SPDLOG_INFO("shift1={}, p1={}", shift1, p1);
        constexpr auto p2 = std::get<2>(perms);
        constexpr auto shift2 = std::get<3>(perms);
        SPDLOG_INFO("shift2={}, p2={}", shift2, p2);
        constexpr auto pres = std::get<4>(perms);
        constexpr auto shiftres = std::get<5>(perms);
        SPDLOG_INFO("shiftres={}, pres={}", shiftres, pres);
        return operator*<TRACK>(this->template permute<shift1>(util::constFct::as_sequence<p1>(), Bool<TRACK>{}),
                                other.template permute<shift2>(util::constFct::as_sequence<p2>(), Bool<TRACK>{}))
            .template permute<shiftres>(util::constFct::as_sequence<pres>(), Bool<TRACK>{});
    }
#endif

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, false, AllocationPolicy>,
               Tensor<RealScalar, 1, 1, Symmetry, false, AllocationPolicy>,
               Tensor<Scalar, 1, CoRank, Symmetry, false, AllocationPolicy>>
    tSVD(std::size_t maxKeep,
         RealScalar eps_svd,
         RealScalar& truncWeight,
         RealScalar& entropy,
         std::map<qarray<Symmetry::Nq>, VectorType>& SVspec,
         bool PRESERVE_MULTIPLETS = true,
         bool RETURN_SPEC = true) XPED_CONST;

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, false, AllocationPolicy>,
               Tensor<RealScalar, 1, 1, Symmetry, false, AllocationPolicy>,
               Tensor<Scalar, 1, CoRank, Symmetry, false, AllocationPolicy>>
    tSVD(std::size_t maxKeep, RealScalar eps_svd, RealScalar& truncWeight, bool PRESERVE_MULTIPLETS = true) XPED_CONST
    {
        RealScalar S_dumb;
        std::map<qarray<Symmetry::Nq>, VectorType> SVspec_dumb;
        return tSVD(maxKeep, eps_svd, truncWeight, S_dumb, SVspec_dumb, PRESERVE_MULTIPLETS, false); // false: Dont return singular value spectrum
    }

    std::pair<Tensor<RealScalar, 1, 1, Symmetry, false, AllocationPolicy>, Tensor<RealScalar, Rank, 1, Symmetry, false, AllocationPolicy>>
    eigh() XPED_CONST;

    const auto& domainTrees(const qType& q) const { return coupledDomain().tree(q); }
    const auto& codomainTrees(const qType& q) const { return coupledCodomain().tree(q); }

    void push_back(const qType& q, const MatrixType& M) { storage_.push_back(q, M); }

    auto begin() { return storage_.begin(); }
    auto end() { return storage_.end(); }

    const auto cbegin() const { return storage_.cbegin(); }
    const auto cend() const { return storage_.cend(); }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("Tensor", ("storage", storage_));
    }

private:
    Storage storage_;

    template <std::size_t... p_domain, std::size_t... p_codomain>
    Self permute_impl(seq::iseq<std::size_t, p_domain...> pd, seq::iseq<std::size_t, p_codomain...> pc) const;

    template <int shift, std::size_t... ps>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy> permute_impl(seq::iseq<std::size_t, ps...> per) const;
};

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <typename OtherDerived>
Tensor<Scalar_, Rank, CoRank, Symmetry, false, AllocationPolicy>::Tensor(const TensorBase<OtherDerived>& other)
{
    storage_ = Storage(other.derived().uncoupledDomain(), other.derived().uncoupledCodomain(), other.derived().world());
    storage_.reserve(other.derived().sector().size());
    for(std::size_t i = 0; i < other.derived().sector().size(); ++i) { storage_.push_back(other.derived().sector(i), other.derived().block(i)); }
}

template <bool TRACK = false, typename Scalar, std::size_t Rank, std::size_t MiddleRank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, false> operator*(const Tensor<Scalar, Rank, MiddleRank, Symmetry, false>& left,
                                                        const Tensor<Scalar, MiddleRank, CoRank, Symmetry, false>& right)
{
    return left.template operator*<TRACK>(right);
}

#ifdef XPED_USE_AD
template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, bool ENABLE_AD = false>
using ArenaTensor = Tensor<Scalar, Rank, CoRank, Symmetry, ENABLE_AD, StanArenaPolicy>;
#endif

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/Tensor.cpp"
#endif

#endif
