#ifndef XPED_H_
#define XPED_H_

#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include "seq/seq.h"

#include "Eigen/SVD"
#include "unsupported/Eigen/KroneckerProduct"

#include "Core/FusionTree.hpp"
#include "Core/Qbasis.hpp"
#include "Core/XpedTypedefs.hpp"
#include "Interfaces/tensor_traits.hpp"
#include "Util/Constfct.hpp"
#include "Util/Macros.hpp"
#include "Util/Random.hpp"

#include "Core/XpedBase.hpp"

template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType_, typename TensorLib_>
struct XpedTraits<Xped<Rank_, CoRank_, Symmetry_, MatrixType_, TensorLib_>>
{
    static constexpr std::size_t Rank = Rank_;
    static constexpr std::size_t CoRank = CoRank_;
    typedef MatrixType_ MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef Symmetry_ Symmetry;
    typedef typename Symmetry::qType qType;
    typedef TensorLib_ TensorLib;
    typedef typename tensortraits<TensorLib>::template Ttype<Scalar, Rank + CoRank> TensorType;
};

template <std::size_t Rank, std::size_t CoRank, typename Symmetry_, typename MatrixType_ = Eigen::MatrixXd, typename TensorLib_ = M_TENSORLIB>
class Xped : public XpedBase<Xped<Rank, CoRank, Symmetry_, MatrixType_, TensorLib_>>
{
    template <typename Derived>
    friend class XpedBase;
    // template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    // friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    // operator*(const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T, const typename MatrixType__::Scalar& s);

    // template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    // friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    // operator*(const typename MatrixType__::Scalar& s, const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T);

    // template <std::size_t Rank_, std::size_t CoRank_, std::size_t MiddleRank, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    // friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    // operator*(const Tensor<Rank_, MiddleRank, Symmetry_, MatrixType__, TensorLib__>& T1,
    //           const Tensor<MiddleRank, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T2);

    template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry__, typename MatrixType__, typename TensorLib__>
    friend Xped<Rank_, CoRank_, Symmetry__, MatrixType__, TensorLib__>
    operator+(const Xped<Rank_, CoRank_, Symmetry__, MatrixType__, TensorLib__>& T1,
              const Xped<Rank_, CoRank_, Symmetry__, MatrixType__, TensorLib__>& T2);

    template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry__, typename MatrixType__, typename TensorLib__>
    friend Xped<Rank_, CoRank_, Symmetry__, MatrixType__, TensorLib__>
    operator-(const Xped<Rank_, CoRank_, Symmetry__, MatrixType__, TensorLib__>& T1,
              const Xped<Rank_, CoRank_, Symmetry__, MatrixType__, TensorLib__>& T2);

    // template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    // friend class Tensor;
    // typedef Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> self;

public:
    typedef MatrixType_ MatrixType;
    typedef Symmetry_ Symmetry;
    typedef typename MatrixType::Scalar Scalar;
    typedef TensorLib_ TensorLib;
    typedef tensortraits<TensorLib> Ttraits;
    typedef typename Ttraits::template Ttype<Scalar, Rank + CoRank> TensorType;
    typedef typename Ttraits::template Maptype<Scalar, Rank + CoRank> TensorMapType;
    typedef typename Ttraits::template cMaptype<Scalar, Rank + CoRank> TensorcMapType;
    typedef typename Ttraits::Indextype IndexType;
    typedef typename Symmetry::qType qType;

    typedef Xped<Rank, CoRank, Symmetry, MatrixType, TensorLib> Self;

    /**Does nothing.*/
    Xped(){};

    Xped(const std::array<Qbasis<Symmetry, 1>, Rank> basis_domain, const std::array<Qbasis<Symmetry, 1>, CoRank> basis_codomain);

    template <typename OtherDerived>
    Xped(const XpedBase<OtherDerived>& other);

    static constexpr std::size_t rank() { return Rank; }
    static constexpr std::size_t corank() { return CoRank; }

    inline const std::vector<qType> sector() const { return sector_; }
    inline const qType sector(std::size_t i) const { return sector_[i]; }

    // inline const std::vector<MatrixType> block() const { return block_; }
    inline const MatrixType block(std::size_t i) const { return block_[i]; }

    inline const std::unordered_map<qType, std::size_t> dict() const { return dict_; }

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

    std::string print(bool PRINT_MATRICES = true) const;

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

    // Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> adjoint() const;
    // AdjointTensor<self> adjoint() const;

    // self conjugate() const;
    // Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> transpose() const;

    // Scalar trace() const;

    // Scalar squaredNorm() const { return (*this * this->adjoint()).trace(); }

    // Scalar norm() const { return std::sqrt(squaredNorm()); }

    template <int shift, std::size_t...>
    Xped<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_> permute() const;

    template <typename EpsScalar>
    std::tuple<Xped<Rank, 1, Symmetry, MatrixType_, TensorLib_>,
               Xped<1, 1, Symmetry, MatrixType_, TensorLib_>,
               Xped<1, CoRank, Symmetry, MatrixType_, TensorLib_>>
    tSVD(size_t maxKeep,
         EpsScalar eps_svd,
         double& truncWeight,
         double& entropy,
         std::map<qarray<Symmetry::Nq>, Eigen::ArrayXd>& SVspec,
         bool PRESERVE_MULTIPLETS = true,
         bool RETURN_SPEC = true) const;

    template <typename EpsScalar>
    std::tuple<Xped<Rank, 1, Symmetry, MatrixType_, TensorLib_>,
               Xped<1, 1, Symmetry, MatrixType_, TensorLib_>,
               Xped<1, CoRank, Symmetry, MatrixType_, TensorLib_>>
    tSVD(size_t maxKeep, EpsScalar eps_svd, double& truncWeight, bool PRESERVE_MULTIPLETS = true) const
    {
        double S_dumb;
        std::map<qarray<Symmetry::Nq>, Eigen::ArrayXd> SVspec_dumb;
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

    void push_back(const qType& q, const MatrixType& M)
    {
        block_.push_back(M);
        sector_.push_back(q);
        dict_.insert(std::make_pair(q, sector_.size() - 1));
    }

    template <std::size_t... p_domain, std::size_t... p_codomain>
    Self permute_impl(seq::iseq<std::size_t, p_domain...> pd, seq::iseq<std::size_t, p_codomain...> pc) const;

    template <int shift, std::size_t... ps>
    Xped<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_> permute_impl(seq::iseq<std::size_t, ps...> per) const;
};

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::Xped(const std::array<Qbasis<Symmetry, 1>, Rank> basis_domain,
                                                            const std::array<Qbasis<Symmetry, 1>, CoRank> basis_codomain)
    : uncoupled_domain(basis_domain)
    , uncoupled_codomain(basis_codomain)
{
    domain = util::build_FusionTree(basis_domain);
    codomain = util::build_FusionTree(basis_codomain);
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <typename OtherDerived>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::Xped(const XpedBase<OtherDerived>& other)
{
    sector_ = other.derived().sector();
    block_.resize(sector_.size());
    for(std::size_t i = 0; i < sector_.size(); i++) { block_[i] = other.derived().block(i); }
    dict_ = other.derived().dict();
    uncoupled_domain = other.derived().uncoupledDomain();
    uncoupled_codomain = other.derived().uncoupledCodomain();
    domain = other.derived().coupledDomain();
    codomain = other.derived().coupledCodomain();
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setRandom()
{
    if(domain.dim() < codomain.dim()) {
        for(const auto& [q, dim, plain] : domain) {
            if(codomain.IS_PRESENT(q)) {
                sector_.push_back(q);
                dict_.insert(std::make_pair(q, sector_.size() - 1));
            }
        }
    } else {
        for(const auto& [q, dim, plain] : codomain) {
            if(domain.IS_PRESENT(q)) {
                sector_.push_back(q);
                dict_.insert(std::make_pair(q, sector_.size() - 1));
            }
        }
    }
    block_.resize(sector_.size());
    for(size_t i = 0; i < sector_.size(); i++) {
        block_[i].resize(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]));
        block_[i].setRandom();
        // for (IndexType row=0; row<block_[i].rows(); row++)
        //         for (IndexType col=0; col<block_[i].cols(); col++) {
        // 		block_[i](row,col) = util::random::threadSafeRandUniform<Scalar>(-1.,1.,true);
        //         }
    }
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setZero()
{
    std::unordered_set<qType> uniqueController;
    for(const auto& [q, dim, plain] : domain) {
        if(auto it = uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
            sector_.push_back(q);
            uniqueController.insert(q);
            dict_.insert(std::make_pair(q, sector_.size() - 1));
        }
    }
    block_.resize(sector_.size());
    for(size_t i = 0; i < sector_.size(); i++) {
        MatrixType mat(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]));
        mat.setZero();
        block_[i] = mat;
    }
}
template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setIdentity()
{
    std::unordered_set<qType> uniqueController;
    for(const auto& [q, dim, plain] : domain) {
        if(auto it = uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
            sector_.push_back(q);
            uniqueController.insert(q);
            dict_.insert(std::make_pair(q, sector_.size() - 1));
        }
    }
    block_.resize(sector_.size());
    for(size_t i = 0; i < sector_.size(); i++) {
        MatrixType mat(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]));
        mat.setIdentity();
        block_[i] = mat;
    }
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setConstant(const Scalar& val)
{
    std::unordered_set<qType> uniqueController;
    for(const auto& [q, dim, plain] : domain) {
        if(auto it = uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
            sector_.push_back(q);
            uniqueController.insert(q);
            dict_.insert(std::make_pair(q, sector_.size() - 1));
        }
    }
    block_.resize(sector_.size());
    for(size_t i = 0; i < sector_.size(); i++) {
        MatrixType mat(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]));
        mat.setConstant(val);
        block_[i] = mat;
    }
}

// template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::adjoint() const
// {
//     Xped<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> T;
//     T.domain = codomain;
//     T.codomain = domain;
//     T.uncoupled_domain = uncoupled_codomain;
//     T.uncoupled_codomain = uncoupled_domain;
//     T.sector = sector;
//     T.dict = dict;
//     T.block_.resize(T.sector_.size());
//     for(size_t i = 0; i < sector_.size(); i++) { T.block_[i] = block_[i].adjoint(); }
//     return T;
// }

// template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::transpose() const
// {
//     Xped<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> T;
//     T.domain = codomain;
//     T.codomain = domain;
//     T.uncoupled_domain = uncoupled_codomain;
//     T.uncoupled_codomain = uncoupled_domain;
//     T.sector = sector;
//     T.dict = dict;
//     T.block_.resize(T.sector_.size());
//     for(size_t i = 0; i < sector_.size(); i++) { T.block_[i] = block_[i].transpose(); }
//     return T;
// }

// template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::conjugate() const
// {
//     self T;
//     T.domain = domain;
//     T.codomain = codomain;
//     T.uncoupled_domain = uncoupled_domain;
//     T.uncoupled_codomain = uncoupled_codomain;
//     T.sector = sector;
//     T.dict = dict;
//     T.block_.resize(T.sector_.size());
//     for(size_t i = 0; i < sector_.size(); i++) { T.block_[i] = block_[i].conjugate(); }
//     return T;
// }

// template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// typename MatrixType_::Scalar Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::trace() const
// {
//     assert(domain == codomain);
//     Scalar out = 0.;
//     for(size_t i = 0; i < sector_.size(); i++) { out += block_[i].trace() * Symmetry::degeneracy(sector_[i]); }
//     return out;
// }

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <std::size_t... pds, std::size_t... pcs>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::permute_impl(seq::iseq<std::size_t, pds...> pd, seq::iseq<std::size_t, pcs...> pc) const
{
    std::array<std::size_t, Rank> pdomain_ = {pds...};
    std::array<std::size_t, CoRank> pcodomain_ = {(pcs - Rank)...};
    Permutation<Rank> p_domain(pdomain_);
    Permutation<CoRank> p_codomain(pcodomain_);

    std::array<IndexType, Rank + CoRank> total_p;
    auto it_total = std::copy(p_domain.pi.begin(), p_domain.pi.end(), total_p.begin());
    auto pi_codomain_shifted = p_codomain.pi;
    std::for_each(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), [](std::size_t& elem) { return elem += Rank; });
    std::copy(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), it_total);
    Self out;
    out.uncoupled_codomain = uncoupled_codomain;
    p_codomain.apply(out.uncoupled_codomain);

    out.uncoupled_domain = uncoupled_domain;
    p_domain.apply(out.uncoupled_domain);

    out.domain = util::build_FusionTree(out.uncoupled_domain);
    out.codomain = util::build_FusionTree(out.uncoupled_codomain);

    for(size_t i = 0; i < sector_.size(); i++) {
        auto domain_trees = domain.tree(sector_[i]);
        auto codomain_trees = codomain.tree(sector_[i]);
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
                auto permuted_domain_trees = domain_tree.permute(p_domain);
                auto permuted_codomain_trees = codomain_tree.permute(p_codomain);

#ifdef XPED_MEMORY_EFFICIENT
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = Ttraits::template shuffle_view<decltype(tensor), pds..., pcs...>(tensor);
#elif defined(XPED_TIME_EFFICIENT)
                auto tensor = this->subBlock(domain_tree, codomain_tree);
                auto Tshuffle = Ttraits::template shuffle<Scalar, Rank + CoRank, pds..., pcs...>(tensor);
#endif

                for(const auto& [permuted_domain_tree, coeff_domain] : permuted_domain_trees)
                    for(const auto& [permuted_codomain_tree, coeff_codomain] : permuted_codomain_trees) {
                        if(std::abs(coeff_domain * coeff_codomain) < 1.e-10) { continue; }

                        auto it = out.dict_.find(sector_[i]);
                        if(it == out.dict_.end()) {
                            MatrixType mat(out.domain.inner_dim(sector_[i]), out.codomain.inner_dim(sector_[i]));
                            mat.setZero();
#ifdef XPED_TIME_EFFICIENT
                            mat.block(out.domain.leftOffset(permuted_domain_tree),
                                      out.codomain.leftOffset(permuted_codomain_tree),
                                      permuted_domain_tree.dim,
                                      permuted_codomain_tree.dim) =
                                coeff_domain * coeff_codomain * Eigen::Map<MatrixType>(Tshuffle.data(), domain_tree.dim, codomain_tree.dim);
#endif
                            out.push_back(sector_[i], mat);
#ifdef XPED_MEMORY_EFFICIENT
                            auto t = out.view(permuted_domain_tree, permuted_codomain_tree, i);
                            Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
#endif
                        } else {
#ifdef XPED_MEMORY_EFFICIENT
                            auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                            Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
#elif defined(XPED_TIME_EFFICIENT)
                            out.block_[it->second].block(out.domain.leftOffset(permuted_domain_tree),
                                                         out.codomain.leftOffset(permuted_codomain_tree),
                                                         permuted_domain_tree.dim,
                                                         permuted_codomain_tree.dim) +=
                                coeff_domain * coeff_codomain * Eigen::Map<MatrixType>(Tshuffle.data(), domain_tree.dim, codomain_tree.dim);
#endif
                        }
                    }
            }
    }
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <int shift, std::size_t... ps>
Xped<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::permute_impl(seq::iseq<std::size_t, ps...> per) const
{
    std::array<std::size_t, Rank + CoRank> p_ = {ps...};
    Permutation<Rank + CoRank> p(p_);
    constexpr std::size_t newRank = Rank - shift;
    constexpr std::size_t newCoRank = CoRank + shift;
    Xped<newRank, newCoRank, Symmetry, MatrixType_, TensorLib_> out;
    for(std::size_t i = 0; i < newRank; i++) {
        if(p.pi[i] > Rank - 1) {
            out.uncoupled_domain[i] = uncoupled_codomain[p.pi[i] - Rank].conj();
        } else {
            out.uncoupled_domain[i] = uncoupled_domain[p.pi[i]];
        }
    }

    for(std::size_t i = 0; i < newCoRank; i++) {
        if(p.pi[i + newRank] > Rank - 1) {
            out.uncoupled_codomain[i] = uncoupled_codomain[p.pi[i + newRank] - Rank];
        } else {
            out.uncoupled_codomain[i] = uncoupled_domain[p.pi[i + newRank]].conj();
        }
    }

    out.domain = util::build_FusionTree(out.uncoupled_domain);
    out.codomain = util::build_FusionTree(out.uncoupled_codomain);

    for(size_t i = 0; i < sector_.size(); i++) {
        auto domain_trees = domain.tree(sector_[i]);
        auto codomain_trees = codomain.tree(sector_[i]);
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
#ifdef XPED_MEMORY_EFFICIENT
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = Ttraits::template shuffle_view<decltype(tensor), ps...>(tensor);
#elif defined(XPED_TIME_EFFICIENT)
                auto tensor = this->subBlock(domain_tree, codomain_tree);
                auto Tshuffle = Ttraits::template shuffle<Scalar, Rank + CoRank, ps...>(tensor);
#endif

                for(const auto& [permuted_trees, coeff] : treepair::permute<shift>(domain_tree, codomain_tree, p)) {
                    if(std::abs(coeff) < 1.e-10) { continue; }

                    auto [permuted_domain_tree, permuted_codomain_tree] = permuted_trees;
                    assert(permuted_domain_tree.q_coupled == permuted_codomain_tree.q_coupled);

                    auto it = out.dict_.find(permuted_domain_tree.q_coupled);
                    if(it == out.dict_.end()) {
                        MatrixType mat(out.domain.inner_dim(permuted_domain_tree.q_coupled), out.codomain.inner_dim(permuted_domain_tree.q_coupled));
                        mat.setZero();
#ifdef XPED_TIME_EFFICIENT
                        mat.block(out.domain.leftOffset(permuted_domain_tree),
                                  out.codomain.leftOffset(permuted_codomain_tree),
                                  permuted_domain_tree.dim,
                                  permuted_codomain_tree.dim) =
                            coeff * Eigen::Map<MatrixType>(Tshuffle.data(), permuted_domain_tree.dim, permuted_codomain_tree.dim);
#endif
                        out.push_back(permuted_domain_tree.q_coupled, mat);
#ifdef XPED_MEMORY_EFFICIENT
                        auto t = out.view(permuted_domain_tree, permuted_codomain_tree, out.block_.size() - 1);
                        Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
#endif
                    } else {
#ifdef XPED_MEMORY_EFFICIENT
                        auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                        Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
#elif defined(XPED_TIME_EFFICIENT)
                        out.block_[it->second].block(out.domain.leftOffset(permuted_domain_tree),
                                                     out.codomain.leftOffset(permuted_codomain_tree),
                                                     permuted_domain_tree.dim,
                                                     permuted_codomain_tree.dim) +=
                            coeff * Eigen::Map<MatrixType>(Tshuffle.data(), permuted_domain_tree.dim, permuted_codomain_tree.dim);
#endif
                    }
                }
            }
    }
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <int shift, std::size_t... p>
Xped<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_> Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::permute() const
{
    using s = seq::iseq<std::size_t, p...>;
    using p_domain = seq::take<Rank - shift, s>;
    using p_codomain = seq::after<Rank - shift, s>;

    if constexpr(seq::filter<util::constFct::isGreaterOrEqual<Rank>, p_codomain>::size() == p_codomain::size() and
                 seq::filter<util::constFct::isSmaller<Rank>, p_domain>::size() == p_domain::size() and shift == 0) {
        return permute_impl(seq::take<Rank, s>{}, seq::after<Rank, s>{});
    } else {
        return permute_impl<shift>(s{});
    }
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <typename EpsScalar>
std::tuple<Xped<Rank, 1, Symmetry, MatrixType_, TensorLib_>,
           Xped<1, 1, Symmetry, MatrixType_, TensorLib_>,
           Xped<1, CoRank, Symmetry, MatrixType_, TensorLib_>>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::tSVD(size_t maxKeep,
                                                            EpsScalar eps_svd,
                                                            double& truncWeight,
                                                            double& entropy,
                                                            std::map<qarray<Symmetry::Nq>, Eigen::ArrayXd>& SVspec,
                                                            bool PRESERVE_MULTIPLETS,
                                                            bool RETURN_SPEC) const
{
    entropy = 0.;
    truncWeight = 0;
    Qbasis<Symmetry, 1> middle;
    for(size_t i = 0; i < sector_.size(); i++) { middle.push_back(sector_[i], std::min(block_[i].rows(), block_[i].cols())); }

    Xped<Rank, 1, Symmetry, MatrixType_, TensorLib_> U(uncoupled_domain, {{middle}});
    Xped<1, 1, Symmetry, MatrixType_, TensorLib_> Sigma({{middle}}, {{middle}});
    Xped<1, CoRank, Symmetry, MatrixType_, TensorLib_> Vdag({{middle}}, uncoupled_codomain);

    std::vector<std::pair<typename Symmetry::qType, double>> allSV;
    for(size_t i = 0; i < sector_.size(); ++i) {
#ifdef XPED_DONT_USE_BDCSVD
        Eigen::JacobiSVD<MatrixType> Jack; // standard SVD
#else
        Eigen::BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
#endif

        Jack.compute(block_[i], Eigen::ComputeThinU | Eigen::ComputeThinV);
        for(size_t j = 0; j < Jack.singularValues().size(); j++) { allSV.push_back(std::make_pair(sector_[i], std::real(Jack.singularValues()(j)))); }
        // for (const auto& s:Jack.singularValues()) {allSV.push_back(make_pair(in[q],s));}

        U.push_back(sector_[i], Jack.matrixU());
        Sigma.push_back(sector_[i], Jack.singularValues().asDiagonal());
        Vdag.push_back(sector_[i], Jack.matrixV().adjoint());
    }
    size_t numberOfStates = allSV.size();
    std::sort(allSV.begin(),
              allSV.end(),
              [](const std::pair<typename Symmetry::qType, double>& sv1, const std::pair<typename Symmetry::qType, double>& sv2) {
                  return sv1.second > sv2.second;
              });
    for(size_t i = maxKeep; i < allSV.size(); i++) { truncWeight += Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second), 2.); }
    allSV.resize(std::min(maxKeep, numberOfStates));
    // std::erase_if(allSV, [eps_svd](const pair<typename Symmetry::qType, Scalar> &sv) { return (sv < eps_svd); }); c++-20 version
    allSV.erase(std::remove_if(
                    allSV.begin(), allSV.end(), [eps_svd](const std::pair<typename Symmetry::qType, double>& sv) { return (sv.second < eps_svd); }),
                allSV.end());

    // cout << "saving sv for expansion to file, #sv=" << allSV.size() << endl;
    // ofstream Filer("sv_expand");
    // size_t index=0;
    // for (const auto & [q,sv]: allSV)
    // {
    // 	Filer << index << "\t" << sv << endl;
    // 	index++;
    // }
    // Filer.close();

    if(PRESERVE_MULTIPLETS) {
        // cutLastMultiplet(allSV);
        int endOfMultiplet = -1;
        for(int i = allSV.size() - 1; i > 0; i--) {
            EpsScalar rel_diff = 2 * (allSV[i - 1].second - allSV[i].second) / (allSV[i - 1].second + allSV[i].second);
            if(rel_diff > 0.1) {
                endOfMultiplet = i;
                break;
            }
        }
        if(endOfMultiplet != -1) {
            // std::cout << termcolor::red << "Cutting of the last " << allSV.size()-endOfMultiplet << " singular values to preserve the multiplet" <<
            // termcolor::reset << std::endl;
            allSV.resize(endOfMultiplet);
        }
    }

    // std::cout << "Adding " << allSV.size() << " states from " << numberOfStates << " states" << std::endl;
    std::map<typename Symmetry::qType, std::vector<Scalar>> qn_orderedSV;
    Qbasis<Symmetry, 1> truncBasis;
    for(const auto& [q, s] : allSV) {
        truncBasis.push_back(q, 1ul);
        qn_orderedSV[q].push_back(s);
        entropy += -Symmetry::degeneracy(q) * s * s * std::log(s * s);
    }
    Xped<Rank, 1, Symmetry, MatrixType_, TensorLib_> trunc_U(uncoupled_domain, {{truncBasis}});
    Xped<1, 1, Symmetry, MatrixType_, TensorLib_> trunc_Sigma({{truncBasis}}, {{truncBasis}});
    Xped<1, CoRank, Symmetry, MatrixType_, TensorLib_> trunc_Vdag({{truncBasis}}, uncoupled_codomain);
    for(const auto& [q, vec_sv] : qn_orderedSV) {
        size_t Nret = vec_sv.size();
        // cout << "q=" << q << ", Nret=" << Nret << endl;
        auto itSigma = Sigma.dict_.find({q});
        trunc_Sigma.push_back(q, Sigma.block_[itSigma->second].diagonal().head(Nret).asDiagonal());
        if(RETURN_SPEC) { SVspec.insert(std::make_pair(q, Sigma.block_[itSigma->second].diagonal().head(Nret).real())); }
        auto itU = U.dict_.find({q});
        trunc_U.push_back(q, U.block_[itU->second].leftCols(Nret));
        auto itVdag = Vdag.dict_.find({q});
        trunc_Vdag.push_back(q, Vdag.block_[itVdag->second].topRows(Nret));
    }
    return std::make_tuple(U, Sigma, Vdag);
}

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// MatrixType_& Xped<Rank, CoRank, Symmetry, MatrixType_>::
// operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2)
// {
//         assert(f1.q_coupled == f2.q_coupled);
//         std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
//         std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
//         const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
//         const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
//         const auto it = dict_.find(f1.q_coupled);
//         return block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Eigen::Map<MatrixType_> Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::
// operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
// {
//         if(f1.q_coupled != f2.q_coupled) {return util::zero_init<MatrixType>();}
//         std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
//         std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
//         const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
//         const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
//         const auto it = dict_.find(f1.q_coupled);
//         return block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
auto Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2)
{
    const auto it = dict_.find(f1.q_coupled);
    assert(it != dict_.end());
    return view(f1, f2, it->second);
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
auto Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::view(const FusionTree<Rank, Symmetry>& f1,
                                                                 const FusionTree<CoRank, Symmetry>& f2,
                                                                 std::size_t block_number)
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);
    std::array<Eigen::Index, Rank + CoRank> dims;
    for(size_t i = 0; i < Rank; i++) {
        assert(f1.dims[i] == uncoupledDomain()[i].inner_dim(f1.q_uncoupled[i]));
        dims[i] = f1.dims[i];
    }
    for(size_t i = 0; i < CoRank; i++) {
        assert(f2.dims[i] == uncoupledCodomain()[i].inner_dim(f2.q_uncoupled[i]));
        dims[i + Rank] = f2.dims[i];
    }

    IndexType left_offset_domain = coupledDomain().leftOffset(f1);
    IndexType left_offset_codomain = coupledCodomain().leftOffset(f2);
#ifdef XPED_USE_EIGEN_TENSOR_LIB
    Eigen::TensorMap<Eigen::Tensor<double, 2>> tmat(block_[block_number].data(),
                                                    std::array<IndexType, 2>{block(block_number).rows(), block(block_number).cols()});
    return tmat
        .slice(std::array<Eigen::Index, 2>{left_offset_domain, left_offset_codomain},
               std::array<Eigen::Index, 2>{static_cast<Eigen::Index>(f1.dim), static_cast<Eigen::Index>(f2.dim)})
        .reshape(dims);
#endif

#ifdef XPED_USE_ARRAY_TENSOR_LIB
    nda::dim<-9, -9, 1> first_dim;
    first_dim.set_extent(dims[0]);
    std::array<nda::dim<-9, -9, -9>, Rank + CoRank - 1> shape_data;
    for(size_t i = 1; i < Rank; i++) {
        shape_data[i - 1].set_extent(dims[i]);
        shape_data[i - 1].set_stride(std::accumulate(dims.begin(), dims.begin() + i, 1ul, std::multiplies<Scalar>()));
    }
    size_t start = (Rank > 0) ? 0ul : 1ul;
    double stride_correction = (Rank > 0) ? block(block_number).rows() : 1.;
    for(size_t i = start; i < CoRank; i++) {
        shape_data[i + Rank - 1].set_extent(dims[i + Rank]);
        shape_data[i + Rank - 1].set_stride(stride_correction *
                                            std::accumulate(dims.begin() + Rank, dims.begin() + Rank + i, 1ul, std::multiplies<Scalar>()));
    }
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), Ttraits::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block(block_number).rows() + left_offset_domain;
    TensorMapType out(block_[block_number].data() + total_offset, block_shape);
    return out;
#endif
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
auto Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
{
    const auto it = dict_.find(f1.q_coupled);
    assert(it != dict_.end());
    return view(f1, f2, it->second);
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
auto Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::view(const FusionTree<Rank, Symmetry>& f1,
                                                                 const FusionTree<CoRank, Symmetry>& f2,
                                                                 std::size_t block_number) const
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);
    std::array<Eigen::Index, Rank + CoRank> dims;
    for(size_t i = 0; i < Rank; i++) { dims[i] = f1.dims[i]; } // ncoupledDomain()[i].inner_dim(f1.q_uncoupled[i]); }
    for(size_t i = 0; i < CoRank; i++) { dims[i + Rank] = f2.dims[i]; } // uncoupledCodomain()[i].inner_dim(f2.q_uncoupled[i]); }

    IndexType left_offset_domain = coupledDomain().leftOffset(f1);
    IndexType left_offset_codomain = coupledCodomain().leftOffset(f2);

#ifdef XPED_USE_EIGEN_TENSOR_LIB
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> tmat(block_[block_number].data(),
                                                          std::array<IndexType, 2>{block(block_number).rows(), block(block_number).cols()});
    return tmat
        .slice(std::array<Eigen::Index, 2>{left_offset_domain, left_offset_codomain},
               std::array<Eigen::Index, 2>{static_cast<Eigen::Index>(f1.dim), static_cast<Eigen::Index>(f2.dim)})
        .reshape(dims);
#endif

#ifdef XPED_USE_ARRAY_TENSOR_LIB
    nda::dim<-9, -9, 1> first_dim;
    first_dim.set_extent(dims[0]);
    std::array<nda::dim<-9, -9, -9>, Rank + CoRank - 1> shape_data;
    for(size_t i = 1; i < Rank; i++) {
        shape_data[i - 1].set_extent(dims[i]);
        shape_data[i - 1].set_stride(std::accumulate(dims.begin(), dims.begin() + i, 1ul, std::multiplies<Scalar>()));
    }
    size_t start = (Rank > 0) ? 0ul : 1ul;
    double stride_correction = (Rank > 0) ? block(block_number).rows() : 1.;
    for(size_t i = start; i < CoRank; i++) {
        shape_data[i + Rank - 1].set_extent(dims[i + Rank]);
        shape_data[i + Rank - 1].set_stride(stride_correction *
                                            std::accumulate(dims.begin() + Rank, dims.begin() + Rank + i, 1ul, std::multiplies<Scalar>()));
    }
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), Ttraits::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block(block_number).rows() + left_offset_domain;
    TensorcMapType out(block_[block_number].data() + total_offset, block_shape);
    return out;
#endif
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
typename tensortraits<TensorLib_>::template Ttype<typename MatrixType_::Scalar, Rank + CoRank>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::subBlock(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
{
    const auto it = dict_.find(f1.q_coupled);
    assert(it != dict_.end());
    return subBlock(f1, f2, it->second);
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
typename tensortraits<TensorLib_>::template Ttype<typename MatrixType_::Scalar, Rank + CoRank>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::subBlock(const FusionTree<Rank, Symmetry>& f1,
                                                                const FusionTree<CoRank, Symmetry>& f2,
                                                                std::size_t block_number) const
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);

    const auto left_offset_domain = domain.leftOffset(f1);
    const auto left_offset_codomain = codomain.leftOffset(f2);
    std::array<IndexType, Rank + CoRank> dims;

    for(size_t i = 0; i < Rank; i++) { dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]); }
    for(size_t i = 0; i < CoRank; i++) { dims[i + Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]); }

    MatrixType submatrix = block_[block_number].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
    // std::cout << "from subblock:" << std::endl << submatrix << std::endl;
    TensorcMapType tensorview = Ttraits::cMap(submatrix.data(), dims);
    TensorType T = Ttraits::template construct<Scalar, Rank + CoRank>(tensorview);
    return T;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
MatrixType_ Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::subMatrix(const FusionTree<Rank, Symmetry>& f1,
                                                                             const FusionTree<CoRank, Symmetry>& f2) const
{
    if(f1.q_coupled != f2.q_coupled) { assert(false); }

    const auto left_offset_domain = domain.leftOffset(f1);
    const auto left_offset_codomain = codomain.leftOffset(f2);
    const auto it = dict_.find(f1.q_coupled);

    MatrixType submatrix = block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
    return submatrix;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
typename tensortraits<TensorLib_>::template Ttype<typename MatrixType_::Scalar, Rank + CoRank>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::plainTensor() const
{
    auto sorted_domain = domain;
    sorted_domain.sort();
    auto sorted_codomain = codomain;
    sorted_codomain.sort();
    auto sorted_uncoupled_domain = uncoupled_domain;
    std::for_each(sorted_uncoupled_domain.begin(), sorted_uncoupled_domain.end(), [](Qbasis<Symmetry, 1>& q) { q.sort(); });
    auto sorted_uncoupled_codomain = uncoupled_codomain;
    std::for_each(sorted_uncoupled_codomain.begin(), sorted_uncoupled_codomain.end(), [](Qbasis<Symmetry, 1>& q) { q.sort(); });

    std::vector<std::size_t> index_sort(sector_.size());
    std::iota(index_sort.begin(), index_sort.end(), 0);
    std::sort(index_sort.begin(), index_sort.end(), [this](std::size_t n1, std::size_t n2) {
        qarray<Symmetry::Nq> q1 = sector_[n1];
        qarray<Symmetry::Nq> q2 = sector_[n2];
        return Symmetry::compare(q1, q2);
    });

    auto sorted_sector = sector_;
    auto sorted_block = block_;
    for(std::size_t i = 0; i < sector_.size(); i++) {
        sorted_sector[i] = sector_[index_sort[i]];
        sorted_block[i] = block_[index_sort[i]];
    }
    MatrixType inner_mat(sorted_domain.fullDim(), sorted_codomain.fullDim());
    inner_mat.setZero();
    for(std::size_t i = 0; i < sorted_sector.size(); i++) {
        inner_mat.block(sorted_domain.full_outer_num(sorted_sector[i]),
                        sorted_codomain.full_outer_num(sorted_sector[i]),
                        Symmetry::degeneracy(sorted_sector[i]) * sorted_block[i].rows(),
                        Symmetry::degeneracy(sorted_sector[i]) * sorted_block[i].cols()) =
            Eigen::kroneckerProduct(sorted_block[i],
                                    MatrixType::Identity(Symmetry::degeneracy(sorted_sector[i]), Symmetry::degeneracy(sorted_sector[i])));
    }
    // cout << "inner_mat:" << endl << std::fixed << inner_mat << endl;
    std::array<IndexType, 2> full_dims = {static_cast<IndexType>(sorted_domain.fullDim()), static_cast<IndexType>(sorted_codomain.fullDim())};
    typename Ttraits::template Maptype<Scalar, 2> map = Ttraits::Map(inner_mat.data(), full_dims);
    typename Ttraits::template Ttype<Scalar, 2> inner_tensor = Ttraits::template construct<Scalar, 2>(map);

    std::array<IndexType, Rank + 1> dims_domain;
    for(size_t i = 0; i < Rank; i++) { dims_domain[i] = sorted_uncoupled_domain[i].fullDim(); }
    dims_domain[Rank] = sorted_domain.fullDim();
    // cout << "dims domain: "; for (const auto& d:dims_domain) {cout << d << " ";} cout << endl;
    typename Ttraits::template Ttype<Scalar, Rank + 1> unitary_domain = Ttraits::template construct<Scalar>(dims_domain);
    Ttraits::template setZero<Scalar, Rank + 1>(unitary_domain);

    for(const auto& [q, num, plain] : sorted_domain) {
        for(const auto& tree : sorted_domain.tree(q)) {
            std::size_t uncoupled_dim = 1;
            for(std::size_t i = 0; i < Rank; i++) { uncoupled_dim *= sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]); }
            MatrixType id(uncoupled_dim, uncoupled_dim);
            id.setIdentity();
            typename Ttraits::template cTtype<Scalar, 2> Tid_mat =
                Ttraits::template construct<Scalar, 2>(Ttraits::Map(id.data(), std::array<IndexType, 2>{id.rows(), id.cols()}));

            std::array<IndexType, Rank + 1> dims;
            for(std::size_t i = 0; i < Rank; i++) { dims[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]); }
            dims[Rank] = uncoupled_dim;
            typename Ttraits::template Ttype<Scalar, Rank + 1> Tid = Ttraits::template reshape<Scalar, 2>(Tid_mat, dims);

            auto T = tree.template asTensor<TensorLib>();
            typename Ttraits::template Ttype<Scalar, Rank + 1> Tfull = Ttraits::template tensorProd<Scalar, Rank + 1>(Tid, T);
            std::array<IndexType, Rank + 1> offsets;
            for(std::size_t i = 0; i < Rank; i++) { offsets[i] = sorted_uncoupled_domain[i].full_outer_num(tree.q_uncoupled[i]); }
            offsets[Rank] = sorted_domain.full_outer_num(q) + sorted_domain.leftOffset(tree) * Symmetry::degeneracy(q);

            std::array<IndexType, Rank + 1> extents;
            for(std::size_t i = 0; i < Rank; i++) {
                extents[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
            }
            extents[Rank] = Ttraits::template dimensions<Scalar, Rank + 1>(Tfull)[Rank];
            Ttraits::template setSubTensor<Scalar, Rank + 1>(unitary_domain, offsets, extents, Tfull); // this amounts to =. Do we need +=?
        }
    }
    // std::cout << "domain" << std::endl; unitary_domain.for_each_value([] (double d) {std::cout << d << std::endl;});

    std::array<IndexType, CoRank + 1> dims_codomain;
    for(size_t i = 0; i < CoRank; i++) { dims_codomain[i] = sorted_uncoupled_codomain[i].fullDim(); }
    dims_codomain[CoRank] = sorted_codomain.fullDim();
    typename Ttraits::template Ttype<Scalar, CoRank + 1> unitary_codomain = Ttraits::template construct<Scalar>(dims_codomain);
    Ttraits::template setZero<Scalar, CoRank + 1>(unitary_codomain);

    for(const auto& [q, num, plain] : sorted_codomain) {
        for(const auto& tree : sorted_codomain.tree(q)) {
            IndexType uncoupled_dim = 1;
            for(std::size_t i = 0; i < CoRank; i++) { uncoupled_dim *= sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]); }
            MatrixType id(uncoupled_dim, uncoupled_dim);
            id.setIdentity();
            typename Ttraits::template cTtype<Scalar, 2> Tid_mat =
                Ttraits::template construct<Scalar, 2>(Ttraits::Map(id.data(), std::array<IndexType, 2>{id.rows(), id.cols()}));

            std::array<IndexType, CoRank + 1> dims;
            for(std::size_t i = 0; i < CoRank; i++) { dims[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]); }
            dims[CoRank] = uncoupled_dim;
            typename Ttraits::template Ttype<Scalar, CoRank + 1> Tid = Ttraits::template reshape<Scalar, 2>(Tid_mat, dims);
            auto T = tree.template asTensor<TensorLib>();

            typename Ttraits::template Ttype<Scalar, CoRank + 1> Tfull = Ttraits::template tensorProd<Scalar, CoRank + 1>(Tid, T);
            std::array<IndexType, CoRank + 1> offsets;
            for(std::size_t i = 0; i < CoRank; i++) { offsets[i] = sorted_uncoupled_codomain[i].full_outer_num(tree.q_uncoupled[i]); }
            offsets[CoRank] = sorted_codomain.full_outer_num(q) + sorted_codomain.leftOffset(tree) * Symmetry::degeneracy(q);
            std::array<IndexType, CoRank + 1> extents;
            for(std::size_t i = 0; i < CoRank; i++) {
                extents[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
            }
            extents[CoRank] = Ttraits::template dimensions<Scalar, CoRank + 1>(Tfull)[CoRank];
            Ttraits::template setSubTensor<Scalar, CoRank + 1>(unitary_codomain, offsets, extents, Tfull); // this amounts to =. Do we need +=?
        }
    }
    // std::cout << "codomain" << std::endl; unitary_codomain.for_each_value([] (double d) {std::cout << d << std::endl;});

    std::array<IndexType, Rank + CoRank> dims_result;
    for(size_t i = 0; i < Rank; i++) { dims_result[i] = sorted_uncoupled_domain[i].fullDim(); }
    for(size_t i = 0; i < CoRank; i++) { dims_result[i + Rank] = sorted_uncoupled_codomain[i].fullDim(); }
    TensorType out = Ttraits::template construct<Scalar>(dims_result);
    Ttraits::template setZero<Scalar, Rank + CoRank>(out);

    out = tensortraits<TensorLib_>::template contract<Scalar, Rank + 1, CoRank + 1, Rank, CoRank>(
        tensortraits<TensorLib_>::template contract<Scalar, Rank + 1, 2, Rank, 0>(unitary_domain, inner_tensor), unitary_codomain);
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
std::string Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::print(bool PRINT_MATRICES) const
{
    std::stringstream ss;
    ss << "domain:" << endl << domain << endl; // << "with trees:" << endl << domain.printTrees() << endl;
    ss << "codomain:" << endl << codomain << endl; // << "with trees:" << endl << codomain.printTrees() << endl;
    for(size_t i = 0; i < sector_.size(); i++) {
        ss << "Sector with QN=" << sector_[i] << endl;
        if(PRINT_MATRICES) { ss << std::fixed << block_[i] << endl; }
    }
    return ss.str();
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
std::ostream& operator<<(std::ostream& os, const Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& t)
{
    os << t.print();
    return os;
}

// template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator*(const Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T,
//                                                                   const typename MatrixType_::Scalar& s)
// {
//     Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout(T.uncoupled_domain, T.uncoupled_codomain);
//     for(size_t i = 0; i < T.sector_.size(); i++) { Tout.push_back(T.sector_[i], T.block_[i] * s); }
//     return Tout;
// }

// template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator*(const typename MatrixType_::Scalar& s,
//                                                                   const Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T)
// {
//     Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout(T.uncoupled_domain, T.uncoupled_codomain);
//     for(size_t i = 0; i < T.sector_.size(); i++) { Tout.push_back(T.sector_[i], T.block_[i] * s); }
//     return Tout;
// }

// template <std::size_t Rank, std::size_t CoRank, std::size_t MiddleRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator*(const Xped<Rank, MiddleRank, Symmetry, MatrixType_, TensorLib_>& T1,
//                                                                   const Xped<MiddleRank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
// {
//     assert(T1.codomain == T2.domain);
//     Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout;
//     Tout.domain = T1.domain;
//     Tout.codomain = T2.codomain;
//     Tout.uncoupled_domain = T1.uncoupled_domain;
//     Tout.uncoupled_codomain = T2.uncoupled_codomain;
//     // Tout.sector = T1.sector;
//     // Tout.dict = T1.dict;
//     // Tout.block_.resize(Tout.sector_.size());
//     std::unordered_set<typename Symmetry::qType> uniqueController;
//     for(size_t i = 0; i < T1.sector_.size(); i++) {
//         uniqueController.insert(T1.sector_[i]);
//         auto it = T2.dict_.find(T1.sector_[i]);
//         if(it == T2.dict_.end()) { continue; }
//         Tout.push_back(T1.sector_[i], T1.block_[i] * T2.block_[it->second]);
//         // Tout.block_[i] = T1.block_[i] * T2.block_[it->second];
//     }
//     for(size_t i = 0; i < T2.sector_.size(); i++) {
//         if(auto it = uniqueController.find(T2.sector_[i]); it != uniqueController.end()) { continue; }
//         auto it = T1.dict_.find(T2.sector_[i]);
//         if(it == T1.dict_.end()) { continue; }
//         Tout.push_back(T2.sector_[i], T2.block_[i] * T1.block_[it->second]);
//     }
//     return Tout;
// }

// template <std::size_t Rank, std::size_t CoRank, std::size_t MiddleRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>
// operator*(const AdjointXped<Xped<MiddleRank, Rank, Symmetry, MatrixType_, TensorLib_>>& T1,
//           const Xped<MiddleRank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
// {
//     assert(T1.coupledCodomain() == T2.coupledDomain());
//     Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout;
//     Tout.domain = T1.coupledDomain();
//     Tout.codomain = T2.coupledCodomain();
//     Tout.uncoupled_domain = T1.uncoupledDomain();
//     Tout.uncoupled_codomain = T2.uncoupledCodomain();
//     // Tout.sector = T1.sector;
//     // Tout.dict = T1.dict;
//     // Tout.block_.resize(Tout.sector_.size());
//     std::unordered_set<typename Symmetry::qType> uniqueController;
//     for(size_t i = 0; i < T1.sectors().size(); i++) {
//         uniqueController.insert(T1.sectors()[i]);
//         auto it = T2.qDict().find(T1.sectors()[i]);
//         if(it == T2.qDict().end()) { continue; }
//         Tout.push_back(T1.sectors()[i], T1.blocks()[i].adjoint() * T2.blocks()[it->second]);
//         // Tout.block_[i] = T1.block_[i] * T2.block_[it->second];
//     }
//     for(size_t i = 0; i < T2.sectors().size(); i++) {
//         if(auto it = uniqueController.find(T2.sectors()[i]); it != uniqueController.end()) { continue; }
//         auto it = T1.qDict().find(T2.sectors()[i]);
//         if(it == T1.qDict().end()) { continue; }
//         Tout.push_back(T2.sectors()[i], T2.blocks()[i] * T1.blocks()[it->second]);
//     }
//     return Tout;
// }

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator+(const Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T1,
                                                                const Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
{
    assert(T1.domain == T2.domain);
    assert(T1.codomain == T2.codomain);
    Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout;
    Tout.domain = T1.domain;
    Tout.codomain = T1.codomain;
    Tout.uncoupled_domain = T1.uncoupled_domain;
    Tout.uncoupled_codomain = T1.uncoupled_codomain;
    Tout.sector_ = T1.sector_;
    Tout.dict_ = T1.dict_;
    Tout.block_.resize(Tout.sector_.size());
    for(size_t i = 0; i < T1.sector_.size(); i++) {
        auto it = T2.dict_.find(T1.sector_[i]);
        Tout.block_[i] = T1.block_[i] + T2.block_[it->second];
    }
    return Tout;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator-(const Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T1,
                                                                const Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
{
    assert(T1.domain == T2.domain);
    assert(T1.codomain == T2.codomain);
    Xped<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout;
    Tout.domain = T1.domain;
    Tout.codomain = T1.codomain;
    Tout.uncoupled_domain = T1.uncoupled_domain;
    Tout.uncoupled_codomain = T1.uncoupled_codomain;
    Tout.sector_ = T1.sector_;
    Tout.dict_ = T1.dict_;
    Tout.block_.resize(Tout.sector_.size());
    for(size_t i = 0; i < T1.sector_.size(); i++) {
        auto it = T2.dict_.find(T1.sector_[i]);
        Tout.block_[i] = T1.block_[i] - T2.block_[it->second];
    }
    return Tout;
}
#endif
