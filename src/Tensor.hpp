#ifndef TENSOR_H_
#define TENSOR_H_

#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "seq/seq.h"

#include "Eigen/SVD"
#include "unsupported/Eigen/KroneckerProduct"

#include "NestedLoopIterator.h"

#include "FusionTree.hpp"
#include "Qbasis.hpp"
#include "interfaces/tensor_traits.hpp"
#include "util/Constfct.hpp"
#include "util/Random.hpp"

namespace util {

template <std::size_t Rank1, std::size_t Rank2, typename Symmetry>
std::pair<Qbasis<Symmetry, Rank2 + Rank1>, std::array<Qbasis<Symmetry, 1>, 0>>
build_FusionTree_Helper(const Qbasis<Symmetry, Rank2>& coupled, const std::array<Qbasis<Symmetry, 1>, Rank1>& uncoupled)
{
    if constexpr(Rank1 == 0) {
        return std::make_pair(coupled, uncoupled);
    } else if constexpr(Rank1 == 1) {
        std::array<Qbasis<Symmetry, 1>, 0> trivial;
        return std::make_pair(coupled.combine(uncoupled[0]), trivial);
    } else {
        std::array<Qbasis<Symmetry, 1>, Rank1 - 1> new_uncoupled;
        std::copy(uncoupled.begin() + 1, uncoupled.end(), new_uncoupled.begin());
        return build_FusionTree_Helper(coupled.combine(uncoupled[0]), new_uncoupled);
    }
}

template <std::size_t Rank, typename Symmetry>
Qbasis<Symmetry, Rank> build_FusionTree(const std::array<Qbasis<Symmetry, 1>, Rank>& uncoupled)
{
    if constexpr(Rank == 0) {
        Qbasis<Symmetry, 0> tmp;
        tmp.push_back(Symmetry::qvacuum(), 1);
        return tmp;
    } else {
        std::array<Qbasis<Symmetry, 1>, Rank - 1> basis_domain_shrinked;
        std::copy(uncoupled.begin() + 1, uncoupled.end(), basis_domain_shrinked.begin());
        auto [domain_, discard] = util::build_FusionTree_Helper(uncoupled[0], basis_domain_shrinked);
        return domain_;
    }
}

// template <typename MatrixType_>
// MatrixType_ zero_init()
// {
//     if constexpr(std::is_same<MatrixType_, Eigen::MatrixXd>::value) {
//         return Eigen::MatrixXd::Zero(1, 1);
//     } else if constexpr(std::is_same<MatrixType_, Eigen::SparseMatrix<double>>::value) {
//         Eigen::SparseMatrix<double> M(1, 1);
//         return M;
//     } else if constexpr(std::is_same<MatrixType_, Eigen::DiagonalMatrix<double, -1>>::value) {
//         Eigen::DiagonalMatrix<double, -1> M(1);
//         M.diagonal() << 0.;
//         return M;
//     }
// }

// template <typename MatrixType_>
// string print_matrix(const MatrixType_& mat)
// {
//     std::stringstream ss;
//     if constexpr(std::is_same<MatrixType_, Eigen::MatrixXd>::value) {
//         ss << mat;
//         return ss.str();
//     } else if constexpr(std::is_same<MatrixType_, Eigen::SparseMatrix<double>>::value) {
//         ss << mat;
//         return ss.str();
//     } else if constexpr(std::is_same<MatrixType_, Eigen::DiagonalMatrix<double, -1>>::value) {
//         ss << mat.toDenseMatrix();
//         return ss.str();
//     }
// }
} // namespace util

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_ = Eigen::MatrixXd, typename TensorLib_ = M_TENSORLIB>
class Tensor
{
    template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    operator*(const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T, const typename MatrixType__::Scalar& s);

    template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    operator*(const typename MatrixType__::Scalar& s, const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T);

    template <std::size_t Rank_, std::size_t CoRank_, std::size_t MiddleRank, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    operator*(const Tensor<Rank_, MiddleRank, Symmetry_, MatrixType__, TensorLib__>& T1,
              const Tensor<MiddleRank, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T2);

    template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    operator+(const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T1,
              const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T2);

    template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>
    operator-(const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T1,
              const Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorLib__>& T2);

    template <std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorLib__>
    friend class Tensor;
    typedef Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> self;

public:
    typedef MatrixType_ MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef TensorLib_ TensorLib;
    typedef tensortraits<TensorLib> Ttraits;
    typedef typename Ttraits::template Ttype<Scalar, Rank + CoRank> TensorType;
    typedef typename Ttraits::template Maptype<Scalar, Rank + CoRank> TensorMapType;
    typedef typename Ttraits::template cMaptype<Scalar, Rank + CoRank> TensorcMapType;
    typedef typename Ttraits::Indextype IndexType;
    typedef typename Symmetry::qType qType;

    /**Does nothing.*/
    Tensor(){};

    Tensor(const std::array<Qbasis<Symmetry, 1>, Rank> basis_domain, const std::array<Qbasis<Symmetry, 1>, CoRank> basis_codomain);

    constexpr std::size_t rank() const { return Rank; }
    constexpr std::size_t corank() const { return CoRank; }

    const std::vector<qType> sectors() const { return sector; }

    const MatrixType operator()(const qType& q_coupled) const
    {
        auto it = dict.find(q_coupled);
        assert(it != dict.end());
        return block[it->second];
    }
    // Eigen::TensorMap<TensorType> operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);
    // Eigen::TensorMap<TensorType> operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;

    // Eigen::TensorMap<TensorType> view(const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);
    auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;

    auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2, std::size_t block_number);

    TensorType subBlock(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;
    // MatrixType& operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);

    std::string print(bool PRINT_MATRICES = true) const;

    void setRandom();
    void setZero();
    void setIdentity();
    void setConstant(const Scalar& val);

    void clear()
    {
        block.clear();
        dict.clear();
        sector.clear();
    }

    // Apply the basis transformation of domain and codomain to the block matrices to get a plain array/tensor
    TensorType plainTensor() const;
    // MatrixType plainMatrix() const;

    Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> adjoint() const;
    self conjugate() const;
    Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> transpose() const;

    Scalar trace() const;

    Scalar squaredNorm() const { return (*this * this->adjoint()).trace(); }

    Scalar norm() const { return std::sqrt(squaredNorm()); }

    template <int shift, std::size_t...>
    Tensor<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_> permute() const;

    template <typename EpsScalar>
    std::tuple<Tensor<Rank, 1, Symmetry, MatrixType_, TensorLib_>,
               Tensor<1, 1, Symmetry, MatrixType_, TensorLib_>,
               Tensor<1, CoRank, Symmetry, MatrixType_, TensorLib_>>
    tSVD(size_t maxKeep,
         EpsScalar eps_svd,
         double& truncWeight,
         double& entropy,
         std::map<qarray<Symmetry::Nq>, Eigen::ArrayXd>& SVspec,
         bool PRESERVE_MULTIPLETS = true,
         bool RETURN_SPEC = true) const;

    template <typename EpsScalar>
    std::tuple<Tensor<Rank, 1, Symmetry, MatrixType_, TensorLib_>,
               Tensor<1, 1, Symmetry, MatrixType_, TensorLib_>,
               Tensor<1, CoRank, Symmetry, MatrixType_, TensorLib_>>
    tSVD(size_t maxKeep, EpsScalar eps_svd, double& truncWeight, bool PRESERVE_MULTIPLETS = true) const
    {
        double S_dumb;
        std::map<qarray<Symmetry::Nq>, Eigen::ArrayXd> SVspec_dumb;
        return tSVD(maxKeep, eps_svd, truncWeight, S_dumb, SVspec_dumb, PRESERVE_MULTIPLETS, false); // false: Dont return singular value spectrum
    }

    std::vector<FusionTree<Rank, Symmetry>> domain_trees(const qType& q) const { return domain.tree(q); }
    std::vector<FusionTree<CoRank, Symmetry>> codomain_trees(const qType& q) const { return codomain.tree(q); }

private:
    std::vector<MatrixType> block;

    std::unordered_map<qType, std::size_t> dict; // sector --> number
    std::vector<qType> sector;

    std::array<Qbasis<Symmetry, 1>, Rank> uncoupled_domain;
    std::array<Qbasis<Symmetry, 1>, CoRank> uncoupled_codomain;
    Qbasis<Symmetry, Rank> domain;
    Qbasis<Symmetry, CoRank> codomain;

    void push_back(const qType& q, const MatrixType& M)
    {
        block.push_back(M);
        sector.push_back(q);
        dict.insert(std::make_pair(q, sector.size() - 1));
    }

    template <std::size_t... p_domain, std::size_t... p_codomain>
    self permute_impl(seq::iseq<std::size_t, p_domain...> pd, seq::iseq<std::size_t, p_codomain...> pc) const;

    template <int shift, std::size_t... ps>
    Tensor<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_> permute_impl(seq::iseq<std::size_t, ps...> per) const;
};

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::Tensor(const std::array<Qbasis<Symmetry, 1>, Rank> basis_domain,
                                                                const std::array<Qbasis<Symmetry, 1>, CoRank> basis_codomain)
    : uncoupled_domain(basis_domain)
    , uncoupled_codomain(basis_codomain)
{
    domain = util::build_FusionTree(basis_domain);
    codomain = util::build_FusionTree(basis_codomain);
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setRandom()
{
    if(domain.dim() < codomain.dim()) {
        for(const auto& [q, dim, plain] : domain) {
            if(codomain.IS_PRESENT(q)) {
                sector.push_back(q);
                dict.insert(std::make_pair(q, sector.size() - 1));
            }
        }
    } else {
        for(const auto& [q, dim, plain] : codomain) {
            if(domain.IS_PRESENT(q)) {
                sector.push_back(q);
                dict.insert(std::make_pair(q, sector.size() - 1));
            }
        }
    }
    block.resize(sector.size());
    for(size_t i = 0; i < sector.size(); i++) {
        block[i].resize(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i]));
        block[i].setRandom();
        // for (IndexType row=0; row<block[i].rows(); row++)
        //         for (IndexType col=0; col<block[i].cols(); col++) {
        // 		block[i](row,col) = util::random::threadSafeRandUniform<Scalar>(-1.,1.,true);
        //         }
    }
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setZero()
{
    std::unordered_set<qType> uniqueController;
    for(const auto& [q, dim, plain] : domain) {
        if(auto it = uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
            sector.push_back(q);
            uniqueController.insert(q);
            dict.insert(std::make_pair(q, sector.size() - 1));
        }
    }
    block.resize(sector.size());
    for(size_t i = 0; i < sector.size(); i++) {
        MatrixType mat(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i]));
        mat.setZero();
        block[i] = mat;
    }
}
template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setIdentity()
{
    std::unordered_set<qType> uniqueController;
    for(const auto& [q, dim, plain] : domain) {
        if(auto it = uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
            sector.push_back(q);
            uniqueController.insert(q);
            dict.insert(std::make_pair(q, sector.size() - 1));
        }
    }
    block.resize(sector.size());
    for(size_t i = 0; i < sector.size(); i++) {
        MatrixType mat(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i]));
        mat.setIdentity();
        block[i] = mat;
    }
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::setConstant(const Scalar& val)
{
    std::unordered_set<qType> uniqueController;
    for(const auto& [q, dim, plain] : domain) {
        if(auto it = uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
            sector.push_back(q);
            uniqueController.insert(q);
            dict.insert(std::make_pair(q, sector.size() - 1));
        }
    }
    block.resize(sector.size());
    for(size_t i = 0; i < sector.size(); i++) {
        MatrixType mat(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i]));
        mat.setConstant(val);
        block[i] = mat;
    }
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::adjoint() const
{
    Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> T;
    T.domain = codomain;
    T.codomain = domain;
    T.uncoupled_domain = uncoupled_codomain;
    T.uncoupled_codomain = uncoupled_domain;
    T.sector = sector;
    T.dict = dict;
    T.block.resize(T.sector.size());
    for(size_t i = 0; i < sector.size(); i++) { T.block[i] = block[i].adjoint(); }
    return T;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::transpose() const
{
    Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorLib_> T;
    T.domain = codomain;
    T.codomain = domain;
    T.uncoupled_domain = uncoupled_codomain;
    T.uncoupled_codomain = uncoupled_domain;
    T.sector = sector;
    T.dict = dict;
    T.block.resize(T.sector.size());
    for(size_t i = 0; i < sector.size(); i++) { T.block[i] = block[i].transpose(); }
    return T;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::conjugate() const
{
    self T;
    T.domain = domain;
    T.codomain = codomain;
    T.uncoupled_domain = uncoupled_domain;
    T.uncoupled_codomain = uncoupled_codomain;
    T.sector = sector;
    T.dict = dict;
    T.block.resize(T.sector.size());
    for(size_t i = 0; i < sector.size(); i++) { T.block[i] = block[i].conjugate(); }
    return T;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
typename MatrixType_::Scalar Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::trace() const
{
    assert(domain == codomain);
    Scalar out = 0.;
    for(size_t i = 0; i < sector.size(); i++) { out += block[i].trace() * Symmetry::degeneracy(sector[i]); }
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <std::size_t... pds, std::size_t... pcs>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::permute_impl(seq::iseq<std::size_t, pds...> pd, seq::iseq<std::size_t, pcs...> pc) const
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
    self out;
    out.uncoupled_codomain = uncoupled_codomain;
    p_codomain.apply(out.uncoupled_codomain);

    out.uncoupled_domain = uncoupled_domain;
    p_domain.apply(out.uncoupled_domain);

    out.domain = util::build_FusionTree(out.uncoupled_domain);
    out.codomain = util::build_FusionTree(out.uncoupled_codomain);

    for(size_t i = 0; i < sector.size(); i++) {
        auto domain_trees = domain.tree(sector[i]);
        auto codomain_trees = codomain.tree(sector[i]);
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
                auto permuted_domain_trees = domain_tree.permute(p_domain);
                auto permuted_codomain_trees = codomain_tree.permute(p_codomain);
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = Ttraits::template shuffle_view<decltype(tensor), pds..., pcs...>(tensor);
                for(const auto& [permuted_domain_tree, coeff_domain] : permuted_domain_trees)
                    for(const auto& [permuted_codomain_tree, coeff_codomain] : permuted_codomain_trees) {
                        if(std::abs(coeff_domain * coeff_codomain) < 1.e-10) { continue; }

                        auto it = out.dict.find(sector[i]);
                        if(it == out.dict.end()) {
                            MatrixType mat(out.domain.inner_dim(sector[i]), out.codomain.inner_dim(sector[i]));
                            mat.setZero();
                            out.push_back(sector[i], mat);
                            auto t = out.view(permuted_domain_tree, permuted_codomain_tree, i);
                            Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
                        } else {
                            auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                            Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
                        }
                    }
            }
    }
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <int shift, std::size_t... ps>
Tensor<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::permute_impl(seq::iseq<std::size_t, ps...> per) const
{
    std::array<std::size_t, Rank + CoRank> p_ = {ps...};
    Permutation<Rank + CoRank> p(p_);
    constexpr std::size_t newRank = Rank - shift;
    constexpr std::size_t newCoRank = CoRank + shift;
    Tensor<newRank, newCoRank, Symmetry, MatrixType_, TensorLib_> out;
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

    for(size_t i = 0; i < sector.size(); i++) {
        auto domain_trees = domain.tree(sector[i]);
        auto codomain_trees = codomain.tree(sector[i]);
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = Ttraits::template shuffle_view<decltype(tensor), ps...>(tensor);
                for(const auto& [permuted_trees, coeff] : treepair::permute<shift>(domain_tree, codomain_tree, p)) {
                    if(std::abs(coeff) < 1.e-10) { continue; }

                    auto [permuted_domain_tree, permuted_codomain_tree] = permuted_trees;
                    assert(permuted_domain_tree.q_coupled == permuted_codomain_tree.q_coupled);

                    auto it = out.dict.find(permuted_domain_tree.q_coupled);
                    if(it == out.dict.end()) {
                        MatrixType mat(out.domain.inner_dim(permuted_domain_tree.q_coupled), out.codomain.inner_dim(permuted_domain_tree.q_coupled));
                        mat.setZero();
                        out.push_back(permuted_domain_tree.q_coupled, mat);
                        auto t = out.view(permuted_domain_tree, permuted_codomain_tree, out.block.size() - 1);
                        Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
                    } else {
                        auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                        Ttraits::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
                    }
                }
            }
    }
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
template <int shift, std::size_t... p>
Tensor<Rank - shift, CoRank + shift, Symmetry, MatrixType_, TensorLib_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::permute() const
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
std::tuple<Tensor<Rank, 1, Symmetry, MatrixType_, TensorLib_>,
           Tensor<1, 1, Symmetry, MatrixType_, TensorLib_>,
           Tensor<1, CoRank, Symmetry, MatrixType_, TensorLib_>>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::tSVD(size_t maxKeep,
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
    for(size_t i = 0; i < sector.size(); i++) { middle.push_back(sector[i], std::min(block[i].rows(), block[i].cols())); }

    Tensor<Rank, 1, Symmetry, MatrixType_, TensorLib_> U(uncoupled_domain, {{middle}});
    Tensor<1, 1, Symmetry, MatrixType_, TensorLib_> Sigma({{middle}}, {{middle}});
    Tensor<1, CoRank, Symmetry, MatrixType_, TensorLib_> Vdag({{middle}}, uncoupled_codomain);

    std::vector<std::pair<typename Symmetry::qType, double>> allSV;
    for(size_t i = 0; i < sector.size(); ++i) {
#ifdef DONT_USE_BDCSVD
        Eigen::JacobiSVD<MatrixType> Jack; // standard SVD
#else
        Eigen::BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
#endif

        Jack.compute(block[i], Eigen::ComputeThinU | Eigen::ComputeThinV);
        for(size_t j = 0; j < Jack.singularValues().size(); j++) { allSV.push_back(std::make_pair(sector[i], std::real(Jack.singularValues()(j)))); }
        // for (const auto& s:Jack.singularValues()) {allSV.push_back(make_pair(in[q],s));}

        U.push_back(sector[i], Jack.matrixU());
        Sigma.push_back(sector[i], Jack.singularValues().asDiagonal());
        Vdag.push_back(sector[i], Jack.matrixV().adjoint());
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
    Tensor<Rank, 1, Symmetry, MatrixType_, TensorLib_> trunc_U(uncoupled_domain, {{truncBasis}});
    Tensor<1, 1, Symmetry, MatrixType_, TensorLib_> trunc_Sigma({{truncBasis}}, {{truncBasis}});
    Tensor<1, CoRank, Symmetry, MatrixType_, TensorLib_> trunc_Vdag({{truncBasis}}, uncoupled_codomain);
    for(const auto& [q, vec_sv] : qn_orderedSV) {
        size_t Nret = vec_sv.size();
        // cout << "q=" << q << ", Nret=" << Nret << endl;
        auto itSigma = Sigma.dict.find({q});
        trunc_Sigma.push_back(q, Sigma.block[itSigma->second].diagonal().head(Nret).asDiagonal());
        if(RETURN_SPEC) { SVspec.insert(std::make_pair(q, Sigma.block[itSigma->second].diagonal().head(Nret).real())); }
        auto itU = U.dict.find({q});
        trunc_U.push_back(q, U.block[itU->second].leftCols(Nret));
        auto itVdag = Vdag.dict.find({q});
        trunc_Vdag.push_back(q, Vdag.block[itVdag->second].topRows(Nret));
    }
    return std::make_tuple(U, Sigma, Vdag);
}

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// MatrixType_& Tensor<Rank, CoRank, Symmetry, MatrixType_>::
// operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2)
// {
//         assert(f1.q_coupled == f2.q_coupled);
//         std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
//         std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
//         const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
//         const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
//         const auto it = dict.find(f1.q_coupled);
//         return block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Eigen::Map<MatrixType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::
// operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
// {
//         if(f1.q_coupled != f2.q_coupled) {return util::zero_init<MatrixType>();}
//         std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
//         std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
//         const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
//         const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
//         const auto it = dict.find(f1.q_coupled);
//         return block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
auto Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
{
    if(f1.q_coupled != f2.q_coupled) { assert(false); }

    const auto it = dict.find(f1.q_coupled);

    std::array<std::size_t, Rank + CoRank> dims;
    for(size_t i = 0; i < Rank; i++) { dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]); }
    for(size_t i = 0; i < CoRank; i++) { dims[i + Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]); }

    IndexType left_offset_domain = domain.leftOffset(f1);
    IndexType left_offset_codomain = codomain.leftOffset(f2);

#ifdef XPED_USE_EIGEN_TENSOR_LIB
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> tmat(block[it->second].data(), {block[it->second].rows(), block[it->second].cols()});
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
    double stride_correction = (Rank > 0) ? block[it->second].rows() : 1.;
    for(size_t i = start; i < CoRank; i++) {
        shape_data[i + Rank - 1].set_extent(dims[i + Rank]);
        shape_data[i + Rank - 1].set_stride(stride_correction *
                                            std::accumulate(dims.begin() + Rank, dims.begin() + Rank + i, 1ul, std::multiplies<Scalar>()));
    }
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), Ttraits::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block[it->second].rows() + left_offset_domain;
    TensorcMapType out(block[it->second].data() + total_offset, block_shape);
    return out;
#endif
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
auto Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::view(const FusionTree<Rank, Symmetry>& f1,
                                                                   const FusionTree<CoRank, Symmetry>& f2,
                                                                   std::size_t block_number)
{
    assert(block_number < sector.size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector[block_number] == f1.q_coupled);
    std::array<Eigen::Index, Rank + CoRank> dims;
    for(size_t i = 0; i < Rank; i++) { dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]); }
    for(size_t i = 0; i < CoRank; i++) { dims[i + Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]); }

    IndexType left_offset_domain = domain.leftOffset(f1);
    IndexType left_offset_codomain = codomain.leftOffset(f2);

#ifdef XPED_USE_EIGEN_TENSOR_LIB
    Eigen::TensorMap<Eigen::Tensor<double, 2>> tmat(block[block_number].data(), {block[block_number].rows(), block[block_number].cols()});
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
    double stride_correction = (Rank > 0) ? block[block_number].rows() : 1.;
    for(size_t i = start; i < CoRank; i++) {
        shape_data[i + Rank - 1].set_extent(dims[i + Rank]);
        shape_data[i + Rank - 1].set_stride(stride_correction *
                                            std::accumulate(dims.begin() + Rank, dims.begin() + Rank + i, 1ul, std::multiplies<Scalar>()));
    }
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), Ttraits::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block[block_number].rows() + left_offset_domain;
    TensorMapType out(block[block_number].data() + total_offset, block_shape);
    return out;
#endif
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
typename tensortraits<TensorLib_>::template Ttype<typename MatrixType_::Scalar, Rank + CoRank>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::subBlock(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
{
    if(f1.q_coupled != f2.q_coupled) { assert(false); }

    const auto left_offset_domain = domain.leftOffset(f1);
    const auto left_offset_codomain = codomain.leftOffset(f2);
    const auto it = dict.find(f1.q_coupled);
    std::array<IndexType, Rank + CoRank> dims;

    for(size_t i = 0; i < Rank; i++) { dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]); }
    for(size_t i = 0; i < CoRank; i++) { dims[i + Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]); }

    MatrixType submatrix = block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
    // std::cout << "from subblock:" << std::endl << submatrix << std::endl;
    TensorcMapType tensorview = Ttraits::cMap(submatrix.data(), dims);
    TensorType T = Ttraits::template construct<Scalar, Rank + CoRank>(tensorview);
    return T;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
typename tensortraits<TensorLib_>::template Ttype<typename MatrixType_::Scalar, Rank + CoRank>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::plainTensor() const
{
    auto sorted_domain = domain;
    sorted_domain.sort();
    auto sorted_codomain = codomain;
    sorted_codomain.sort();
    auto sorted_uncoupled_domain = uncoupled_domain;
    std::for_each(sorted_uncoupled_domain.begin(), sorted_uncoupled_domain.end(), [](Qbasis<Symmetry, 1>& q) { q.sort(); });
    auto sorted_uncoupled_codomain = uncoupled_codomain;
    std::for_each(sorted_uncoupled_codomain.begin(), sorted_uncoupled_codomain.end(), [](Qbasis<Symmetry, 1>& q) { q.sort(); });

    std::vector<std::size_t> index_sort(sector.size());
    std::iota(index_sort.begin(), index_sort.end(), 0);
    std::sort(index_sort.begin(), index_sort.end(), [this](std::size_t n1, std::size_t n2) {
        qarray<Symmetry::Nq> q1 = sector[n1];
        qarray<Symmetry::Nq> q2 = sector[n2];
        return Symmetry::compare(q1, q2);
    });

    auto sorted_sector = sector;
    auto sorted_block = block;
    for(std::size_t i = 0; i < sector.size(); i++) {
        sorted_sector[i] = sector[index_sort[i]];
        sorted_block[i] = block[index_sort[i]];
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
std::string Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>::print(bool PRINT_MATRICES) const
{
    std::stringstream ss;
    ss << "domain:" << endl << domain << endl; // << "with trees:" << endl << domain.printTrees() << endl;
    ss << "codomain:" << endl << codomain << endl; // << "with trees:" << endl << codomain.printTrees() << endl;
    for(size_t i = 0; i < sector.size(); i++) {
        ss << "Sector with QN=" << sector[i] << endl;
        if(PRINT_MATRICES) { ss << std::fixed << block[i] << endl; }
    }
    return ss.str();
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
std::ostream& operator<<(std::ostream& os, const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& t)
{
    os << t.print();
    return os;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator*(const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T,
                                                                  const typename MatrixType_::Scalar& s)
{
    Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout(T.uncoupled_domain, T.uncoupled_codomain);
    for(size_t i = 0; i < T.sector.size(); i++) { Tout.push_back(T.sector[i], T.block[i] * s); }
    return Tout;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator*(const typename MatrixType_::Scalar& s,
                                                                  const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T)
{
    Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout(T.uncoupled_domain, T.uncoupled_codomain);
    for(size_t i = 0; i < T.sector.size(); i++) { Tout.push_back(T.sector[i], T.block[i] * s); }
    return Tout;
}

template <std::size_t Rank, std::size_t CoRank, std::size_t MiddleRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator*(const Tensor<Rank, MiddleRank, Symmetry, MatrixType_, TensorLib_>& T1,
                                                                  const Tensor<MiddleRank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
{
    assert(T1.codomain == T2.domain);
    Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout;
    Tout.domain = T1.domain;
    Tout.codomain = T2.codomain;
    Tout.uncoupled_domain = T1.uncoupled_domain;
    Tout.uncoupled_codomain = T2.uncoupled_codomain;
    // Tout.sector = T1.sector;
    // Tout.dict = T1.dict;
    // Tout.block.resize(Tout.sector.size());
    std::unordered_set<typename Symmetry::qType> uniqueController;
    for(size_t i = 0; i < T1.sector.size(); i++) {
        uniqueController.insert(T1.sector[i]);
        auto it = T2.dict.find(T1.sector[i]);
        if(it == T2.dict.end()) { continue; }
        Tout.push_back(T1.sector[i], T1.block[i] * T2.block[it->second]);
        // Tout.block[i] = T1.block[i] * T2.block[it->second];
    }
    for(size_t i = 0; i < T2.sector.size(); i++) {
        if(auto it = uniqueController.find(T2.sector[i]); it != uniqueController.end()) { continue; }
        auto it = T1.dict.find(T2.sector[i]);
        if(it == T1.dict.end()) { continue; }
        Tout.push_back(T2.sector[i], T2.block[i] * T1.block[it->second]);
    }
    return Tout;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator+(const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T1,
                                                                  const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
{
    assert(T1.domain == T2.domain);
    assert(T1.codomain == T2.codomain);
    Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout;
    Tout.domain = T1.domain;
    Tout.codomain = T1.codomain;
    Tout.uncoupled_domain = T1.uncoupled_domain;
    Tout.uncoupled_codomain = T1.uncoupled_codomain;
    Tout.sector = T1.sector;
    Tout.dict = T1.dict;
    Tout.block.resize(Tout.sector.size());
    for(size_t i = 0; i < T1.sector.size(); i++) {
        auto it = T2.dict.find(T1.sector[i]);
        Tout.block[i] = T1.block[i] + T2.block[it->second];
    }
    return Tout;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> operator-(const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T1,
                                                                  const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
{
    assert(T1.domain == T2.domain);
    assert(T1.codomain == T2.codomain);
    Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorLib_> Tout;
    Tout.domain = T1.domain;
    Tout.codomain = T1.codomain;
    Tout.uncoupled_domain = T1.uncoupled_domain;
    Tout.uncoupled_codomain = T1.uncoupled_codomain;
    Tout.sector = T1.sector;
    Tout.dict = T1.dict;
    Tout.block.resize(Tout.sector.size());
    for(size_t i = 0; i < T1.sector.size(); i++) {
        auto it = T2.dict.find(T1.sector[i]);
        Tout.block[i] = T1.block[i] - T2.block[it->second];
    }
    return Tout;
}
#endif
