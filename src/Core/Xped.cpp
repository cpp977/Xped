#include <iostream>
#include <sstream>
#include <unordered_set>

#include "spdlog/spdlog.h"

#include "spdlog/fmt/bundled/ranges.h"
#include "spdlog/fmt/ostr.h"

#include "Xped/Util/Constfct.hpp"
#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Random.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

using std::cout;
using std::endl;
using std::size_t;

#include "Xped/Core/Xped.hpp"

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::Xped(const std::array<Qbasis<Symmetry, 1>, Rank> basis_domain,
                                                       const std::array<Qbasis<Symmetry, 1>, CoRank> basis_codomain,
                                                       util::mpi::XpedWorld& world)
    : uncoupled_domain(basis_domain)
    , uncoupled_codomain(basis_codomain)
    , world_(&world, util::mpi::TrivialDeleter<util::mpi::XpedWorld>{})
{
    domain = XpedHelper::build_FusionTree(basis_domain);
    codomain = XpedHelper::build_FusionTree(basis_codomain);
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
void Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::setRandom()
{
    SPDLOG_TRACE("Entering set Random().");
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
    SPDLOG_TRACE("Start randomization loop with #={} iterations.", sector_.size());
    for(size_t i = 0; i < sector_.size(); i++) {
        // auto mat = Plain::template construct<Scalar>(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]), world_);
        // Plain::template setRandom<Scalar>(mat);
        block_[i] = Plain::template construct<Scalar>(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]), *world_);
        Plain::template setRandom<Scalar>(block_[i]);
        SPDLOG_TRACE("Set block #={} to random.", i);
        // block_[i].print_matrix();
        // for (IndexType row=0; row<block_[i].rows(); row++)
        //         for (IndexType col=0; col<block_[i].cols(); col++) {
        // 		block_[i](row,col) = util::random::threadSafeRandUniform<Scalar>(-1.,1.,true);
        //         }
    }
    SPDLOG_TRACE("Leaving set Random().");
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
void Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::setZero()
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
        // MatrixType mat(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]));
        auto mat = Plain::template construct<Scalar>(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]), *world_);
        Plain::template setZero<Scalar>(mat);
        // mat.setZero();
        block_[i] = mat;
    }
}
template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
void Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::setIdentity()
{
    SPDLOG_TRACE("Entering Xped::setIdentity().");
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
        // MatrixType mat(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]));
        auto mat = Plain::template construct<Scalar>(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]), *world_);
        Plain::template setIdentity<Scalar>(mat);
        block_[i] = mat;
    }
    SPDLOG_TRACE("Leaving Xped::setIdentity().");
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
void Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::setConstant(const Scalar& val)
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
        // MatrixType mat(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]));
        auto mat = Plain::template construct<Scalar>(domain.inner_dim(sector_[i]), codomain.inner_dim(sector_[i]), *world_);
        Plain::template setConstant<Scalar>(mat, val);
        block_[i] = mat;
    }
}

// template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
// Xped<CoRank, Rank, Symmetry, MatrixLib_, TensorLib_> Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::adjoint() const
// {
//     Xped<CoRank, Rank, Symmetry, MatrixLib_, TensorLib_> T;
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

// template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
// Xped<CoRank, Rank, Symmetry, MatrixLib_, TensorLib_> Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::transpose() const
// {
//     Xped<CoRank, Rank, Symmetry, MatrixLib_, TensorLib_> T;
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

// template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
// Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_,
// VectorLib_>::conjugate() const
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

// template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
// typename MatrixLib_::Scalar Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::trace() const
// {
//     assert(domain == codomain);
//     Scalar out = 0.;
//     for(size_t i = 0; i < sector_.size(); i++) { out += block_[i].trace() * Symmetry::degeneracy(sector_[i]); }
//     return out;
// }

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
template <std::size_t... pds, std::size_t... pcs>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::permute_impl(seq::iseq<std::size_t, pds...> pd, seq::iseq<std::size_t, pcs...> pc) const
{
    std::array<std::size_t, Rank> pdomain_ = {pds...};
    std::array<std::size_t, CoRank> pcodomain_ = {(pcs - Rank)...};
    Permutation p_domain(pdomain_);
    Permutation p_codomain(pcodomain_);

    std::array<IndexType, Rank + CoRank> total_p;
    auto it_total = std::copy(p_domain.pi.begin(), p_domain.pi.end(), total_p.begin());
    auto pi_codomain_shifted = p_codomain.pi;
    std::for_each(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), [](std::size_t& elem) { return elem += Rank; });
    std::copy(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), it_total);
    Self out;
    out.world_ = this->world_;
    out.uncoupled_codomain = uncoupled_codomain;
    p_codomain.apply(out.uncoupled_codomain);

    out.uncoupled_domain = uncoupled_domain;
    p_domain.apply(out.uncoupled_domain);

    out.domain = XpedHelper::build_FusionTree(out.uncoupled_domain);
    out.codomain = XpedHelper::build_FusionTree(out.uncoupled_codomain);

    for(size_t i = 0; i < sector_.size(); i++) {
        auto domain_trees = domain.tree(sector_[i]);
        auto codomain_trees = codomain.tree(sector_[i]);
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
                auto permuted_domain_trees = domain_tree.permute(p_domain);
                auto permuted_codomain_trees = codomain_tree.permute(p_codomain);

#ifdef XPED_MEMORY_EFFICIENT
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = Plain::template shuffle_view<decltype(tensor), pds..., pcs...>(tensor);
#elif defined(XPED_TIME_EFFICIENT)
                auto tensor = this->subBlock(domain_tree, codomain_tree);
                auto Tshuffle = Plain::template shuffle<Scalar, Rank + CoRank, pds..., pcs...>(tensor);
#endif

                for(const auto& [permuted_domain_tree, coeff_domain] : permuted_domain_trees)
                    for(const auto& [permuted_codomain_tree, coeff_codomain] : permuted_codomain_trees) {
                        if(std::abs(coeff_domain * coeff_codomain) < 1.e-10) { continue; }

                        auto it = out.dict_.find(sector_[i]);
                        if(it == out.dict_.end()) {
                            // MatrixType mat(out.domain.inner_dim(sector_[i]), out.codomain.inner_dim(sector_[i]));
                            // mat.setZero();
                            auto mat = Plain::template construct_with_zero<Scalar>(
                                out.domain.inner_dim(sector_[i]), out.codomain.inner_dim(sector_[i]), *world_);
#ifdef XPED_TIME_EFFICIENT
                            IndexType row = out.domain.leftOffset(permuted_domain_tree);
                            IndexType col = out.codomain.leftOffset(permuted_codomain_tree);
                            IndexType rows = permuted_domain_tree.dim;
                            IndexType cols = permuted_codomain_tree.dim;
                            Plain::template set_block_from_tensor<Scalar, Rank + CoRank>(
                                mat, row, col, rows, cols, coeff_domain * coeff_codomain, Tshuffle);
                            // assert(permuted_domain_tree.dim == domain_tree.dim);
                            // assert(permuted_codomain_tree.dim == codomain_tree.dim);
                            // mat.block(out.domain.leftOffset(permuted_domain_tree),
                            //             out.codomain.leftOffset(permuted_codomain_tree),
                            //             permuted_domain_tree.dim,
                            //             permuted_codomain_tree.dim) =
                            //     coeff_domain * coeff_codomain *
                            //     Eigen::Map<MatrixType>(
                            //         Plain::template get_raw_data<Scalar, Rank + CoRank>(Tshuffle), domain_tree.dim, codomain_tree.dim);
#endif
                            out.push_back(sector_[i], mat);
#ifdef XPED_MEMORY_EFFICIENT
                            auto t = out.view(permuted_domain_tree, permuted_codomain_tree, i);
                            Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
#endif
                        } else {
#ifdef XPED_MEMORY_EFFICIENT
                            auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                            Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
#elif defined(XPED_TIME_EFFICIENT)
                            IndexType row = out.domain.leftOffset(permuted_domain_tree);
                            IndexType col = out.codomain.leftOffset(permuted_codomain_tree);
                            IndexType rows = permuted_domain_tree.dim;
                            IndexType cols = permuted_codomain_tree.dim;
                            Plain::template add_to_block_from_tensor<Scalar, Rank + CoRank>(
                                out.block_[it->second], row, col, rows, cols, coeff_domain * coeff_codomain, Tshuffle);
                            // assert(permuted_domain_tree.dim == domain_tree.dim);
                            // assert(permuted_codomain_tree.dim == codomain_tree.dim);
                            // out.block_[it->second].block(out.domain.leftOffset(permuted_domain_tree),
                            //                              out.codomain.leftOffset(permuted_codomain_tree),
                            //                              permuted_domain_tree.dim,
                            //                              permuted_codomain_tree.dim) +=
                            //     coeff_domain * coeff_codomain *
                            //     Eigen::Map<MatrixType>(
                            //         Plain::template get_raw_data<Scalar, Rank + CoRank>(Tshuffle), domain_tree.dim, codomain_tree.dim);
#endif
                        }
                    }
            }
    }
    return out;
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
template <int shift, std::size_t... ps>
Xped<Scalar_, Rank - shift, CoRank + shift, Symmetry, PlainLib_>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::permute_impl(seq::iseq<std::size_t, ps...> per) const
{
    std::array<std::size_t, Rank + CoRank> p_ = {ps...};
    Permutation p(p_);
    constexpr std::size_t newRank = Rank - shift;
    constexpr std::size_t newCoRank = CoRank + shift;
    Xped<Scalar, newRank, newCoRank, Symmetry, PlainLib_> out;
    out.world_ = this->world_;
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

    out.domain = XpedHelper::build_FusionTree(out.uncoupled_domain);
    out.codomain = XpedHelper::build_FusionTree(out.uncoupled_codomain);

    for(size_t i = 0; i < sector_.size(); i++) {
        auto domain_trees = domain.tree(sector_[i]);
        auto codomain_trees = codomain.tree(sector_[i]);
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
#ifdef XPED_MEMORY_EFFICIENT
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = Plain::template shuffle_view<decltype(tensor), ps...>(tensor);
#elif defined(XPED_TIME_EFFICIENT)
                auto tensor = this->subBlock(domain_tree, codomain_tree);
                auto Tshuffle = Plain::template shuffle<Scalar, Rank + CoRank, ps...>(tensor);
#endif

                for(const auto& [permuted_trees, coeff] : treepair::permute<shift>(domain_tree, codomain_tree, p)) {
                    if(std::abs(coeff) < 1.e-10) { continue; }

                    auto [permuted_domain_tree, permuted_codomain_tree] = permuted_trees;
                    assert(permuted_domain_tree.q_coupled == permuted_codomain_tree.q_coupled);

                    auto it = out.dict_.find(permuted_domain_tree.q_coupled);
                    if(it == out.dict_.end()) {
                        auto mat = Plain::template construct_with_zero<Scalar>(
                            out.domain.inner_dim(permuted_domain_tree.q_coupled), out.codomain.inner_dim(permuted_domain_tree.q_coupled), *world_);
                        // MatrixType mat(out.domain.inner_dim(permuted_domain_tree.q_coupled),
                        // out.codomain.inner_dim(permuted_domain_tree.q_coupled)); mat.setZero();
#ifdef XPED_TIME_EFFICIENT
                        IndexType row = out.domain.leftOffset(permuted_domain_tree);
                        IndexType col = out.codomain.leftOffset(permuted_codomain_tree);
                        IndexType rows = permuted_domain_tree.dim;
                        IndexType cols = permuted_codomain_tree.dim;
                        Plain::template set_block_from_tensor<Scalar, Rank + CoRank>(mat, row, col, rows, cols, coeff, Tshuffle);
                        // mat.block(out.domain.leftOffset(permuted_domain_tree),
                        //           out.codomain.leftOffset(permuted_codomain_tree),
                        //           permuted_domain_tree.dim,
                        //           permuted_codomain_tree.dim) =
                        //     coeff * Eigen::Map<MatrixType>(Plain::template get_raw_data<Scalar, Rank + CoRank>(Tshuffle),
                        //                                    permuted_domain_tree.dim,
                        //                                    permuted_codomain_tree.dim);
#endif
                        out.push_back(permuted_domain_tree.q_coupled, mat);
#ifdef XPED_MEMORY_EFFICIENT
                        auto t = out.view(permuted_domain_tree, permuted_codomain_tree, out.block_.size() - 1);
                        Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
#endif
                    } else {
#ifdef XPED_MEMORY_EFFICIENT
                        auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                        Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
#elif defined(XPED_TIME_EFFICIENT)
                        IndexType row = out.domain.leftOffset(permuted_domain_tree);
                        IndexType col = out.codomain.leftOffset(permuted_codomain_tree);
                        IndexType rows = permuted_domain_tree.dim;
                        IndexType cols = permuted_codomain_tree.dim;
                        Plain::template add_to_block_from_tensor<Scalar, Rank + CoRank>(
                            out.block_[it->second], row, col, rows, cols, coeff, Tshuffle);
                        // out.block_[it->second].block(out.domain.leftOffset(permuted_domain_tree),
                        //                              out.codomain.leftOffset(permuted_codomain_tree),
                        //                              permuted_domain_tree.dim,
                        //                              permuted_codomain_tree.dim) +=
                        //     coeff * Eigen::Map<MatrixType>(Plain::template get_raw_data<Scalar, Rank + CoRank>(Tshuffle),
                        //                                    permuted_domain_tree.dim,
                        //                                    permuted_codomain_tree.dim);
#endif
                    }
                }
            }
    }
    return out;
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
template <int shift, std::size_t... p>
Xped<Scalar_, Rank - shift, CoRank + shift, Symmetry, PlainLib_> Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::permute() const
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

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
std::tuple<Xped<Scalar_, Rank, 1, Symmetry, PlainLib_>,
           Xped<typename ScalarTraits<Scalar_>::Real, 1, 1, Symmetry, PlainLib_>,
           Xped<Scalar_, 1, CoRank, Symmetry, PlainLib_>>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::tSVD(size_t maxKeep,
                                                       RealScalar eps_svd,
                                                       RealScalar& truncWeight,
                                                       RealScalar& entropy,
                                                       std::map<qarray<Symmetry::Nq>, VectorType>& SVspec,
                                                       bool PRESERVE_MULTIPLETS,
                                                       bool RETURN_SPEC) XPED_CONST
{
    SPDLOG_INFO("Entering Xped::tSVD()");
    SPDLOG_INFO("Input param eps_svd={}", eps_svd);
    entropy = 0.;
    truncWeight = 0;
    Qbasis<Symmetry, 1> middle;
    for(size_t i = 0; i < sector_.size(); i++) { middle.push_back(sector_[i], std::min(Plain::rows(block_[i]), Plain::cols(block_[i]))); }

    Xped<Scalar, Rank, 1, Symmetry, PlainLib_> U(uncoupled_domain, {{middle}});
    Xped<RealScalar, 1, 1, Symmetry, PlainLib_> Sigma({{middle}}, {{middle}});
    Xped<Scalar, 1, CoRank, Symmetry, PlainLib_> Vdag({{middle}}, uncoupled_codomain);

    std::vector<std::pair<typename Symmetry::qType, RealScalar>> allSV;
    SPDLOG_INFO("Performing the svd loop (size={})", sector_.size());
    for(size_t i = 0; i < sector_.size(); ++i) {
        SPDLOG_INFO("Step i={} for mat with dim=({},{})", i, Plain::template rows<Scalar>(block_[i]), Plain::template rows<Scalar>(block_[i]));
        auto [Umat, Sigmavec, Vmatdag] = Plain::template svd<Scalar>(block_[i]);
        SPDLOG_INFO("Performed svd for step i={}", i);
        // #ifdef XPED_DONT_USE_BDCSVD
        //         Eigen::JacobiSVD<MatrixType> Jack; // standard SVD
        // #else
        //         Eigen::BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
        // #endif

        //         Jack.compute(block_[i], Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::vector<Scalar> svs;
        Plain::template vec_to_stdvec<Scalar>(Sigmavec, svs);

        for(const auto& sv : svs) {
            SPDLOG_INFO("Move the element {} from sigma to allSV", sv);
            allSV.push_back(std::make_pair(sector_[i], sv));
        }
        SPDLOG_INFO("Extracted singular values for step i={}", i);
        auto Sigmamat = Plain::template vec_to_diagmat<Scalar>(Sigmavec);
        U.push_back(sector_[i], Umat);
        Sigma.push_back(sector_[i], Sigmamat);
        Vdag.push_back(sector_[i], Vmatdag);
    }
    size_t numberOfStates = allSV.size();
    SPDLOG_INFO("numberOfStates={}", numberOfStates);
    SPDLOG_INFO("allSV={}\n", allSV);
    std::sort(allSV.begin(),
              allSV.end(),
              [](const std::pair<typename Symmetry::qType, double>& sv1, const std::pair<typename Symmetry::qType, double>& sv2) {
                  return sv1.second > sv2.second;
              });
    SPDLOG_INFO("numberOfStates after sort {}", allSV.size());
    for(size_t i = maxKeep; i < allSV.size(); i++) { truncWeight += Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second), 2.); }
    allSV.resize(std::min(maxKeep, numberOfStates));
    SPDLOG_INFO("numberOfStates after resize {}", allSV.size());
    // std::erase_if(allSV, [eps_svd](const pair<typename Symmetry::qType, Scalar> &sv) { return (sv < eps_svd); }); c++-20 version
    allSV.erase(std::remove_if(
                    allSV.begin(), allSV.end(), [eps_svd](const std::pair<typename Symmetry::qType, double>& sv) { return (sv.second < eps_svd); }),
                allSV.end());
    SPDLOG_INFO("numberOfStates after erase {}", allSV.size());
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
            RealScalar rel_diff = 2 * (allSV[i - 1].second - allSV[i].second) / (allSV[i - 1].second + allSV[i].second);
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
    SPDLOG_INFO("Adding {} states from {} states", allSV.size(), numberOfStates);
    // std::cout << "Adding " << allSV.size() << " states from " << numberOfStates << " states" << std::endl;
    std::map<typename Symmetry::qType, std::vector<Scalar>> qn_orderedSV;
    Qbasis<Symmetry, 1> truncBasis;
    for(const auto& [q, s] : allSV) {
        truncBasis.push_back(q, 1ul);
        qn_orderedSV[q].push_back(s);
        entropy += -Symmetry::degeneracy(q) * s * s * std::log(s * s);
    }
    SPDLOG_INFO("Set up the truncated basis.");
    std::stringstream ss;
    ss << truncBasis.print();
    SPDLOG_INFO(ss.str());

    Xped<Scalar, Rank, 1, Symmetry, PlainLib_> trunc_U(uncoupled_domain, {{truncBasis}});
    Xped<RealScalar, 1, 1, Symmetry, PlainLib_> trunc_Sigma({{truncBasis}}, {{truncBasis}});
    Xped<Scalar, 1, CoRank, Symmetry, PlainLib_> trunc_Vdag({{truncBasis}}, uncoupled_codomain);
    SPDLOG_INFO("Starting the loop for truncating U,S,V (size={})", qn_orderedSV.size());
    for(const auto& [q, vec_sv] : qn_orderedSV) {
        SPDLOG_INFO("Step with q={}", q.data[0]);
        size_t Nret = vec_sv.size();
        // cout << "q=" << q << ", Nret=" << Nret << endl;
        auto itSigma = Sigma.dict_.find({q});
        SPDLOG_INFO("Searched the dict of Sigma.");
        auto sigma_mat = Plain::template block(Sigma.block_[itSigma->second], 0, 0, Nret, Nret);
        SPDLOG_INFO("Got subblock of Sigma.");
        trunc_Sigma.push_back(q, sigma_mat);
        // if(RETURN_SPEC) { SVspec.insert(std::make_pair(q, Sigma.block_[itSigma->second].diagonal().head(Nret).real())); }
        SPDLOG_INFO("Before return spec.");
        if(RETURN_SPEC) {
            VectorType spec;
            Plain::template diagonal_head_matrix_to_vector<RealScalar>(spec, Sigma.block_[itSigma->second], Nret);
            SVspec.insert(std::make_pair(q, spec));
        }
        SPDLOG_INFO("After return spec.");
        auto itU = U.dict_.find({q});
        trunc_U.push_back(q, Plain::template block(U.block_[itU->second], 0, 0, Plain::rows(U.block_[itU->second]), Nret));
        auto itVdag = Vdag.dict_.find({q});
        trunc_Vdag.push_back(q, Plain::template block(Vdag.block_[itVdag->second], 0, 0, Nret, Plain::cols(U.block_[itU->second])));
    }
    SPDLOG_INFO("Leaving Xped::tSVD()");
    return std::make_tuple(U, Sigma, Vdag);
}

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixLib_, typename TensorLib_>
// MatrixLib_& Xped<Rank, CoRank, Symmetry, MatrixLib_>::
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

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixLib_, typename TensorLib_>
// Eigen::Map<MatrixLib_> Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::
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

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
auto Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2)
{
    const auto it = dict_.find(f1.q_coupled);
    assert(it != dict_.end());
    return view(f1, f2, it->second);
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
auto Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::view(const FusionTree<Rank, Symmetry>& f1,
                                                            const FusionTree<CoRank, Symmetry>& f2,
                                                            std::size_t block_number)
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);
    std::array<IndexType, Rank + CoRank> dims;
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
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), Plain::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block(block_number).rows() + left_offset_domain;
    TensorMapType out(block_[block_number].data() + total_offset, block_shape);
    return out;
#endif
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
auto Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
{
    const auto it = dict_.find(f1.q_coupled);
    assert(it != dict_.end());
    return view(f1, f2, it->second);
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
auto Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::view(const FusionTree<Rank, Symmetry>& f1,
                                                            const FusionTree<CoRank, Symmetry>& f2,
                                                            std::size_t block_number) const
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);
    std::array<IndexType, Rank + CoRank> dims;
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
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), Plain::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block(block_number).rows() + left_offset_domain;
    TensorcMapType out(block_[block_number].data() + total_offset, block_shape);
    return out;
#endif
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
typename PlainLib_::template TType<Scalar_, Rank + CoRank>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::subBlock(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
{
    const auto it = dict_.find(f1.q_coupled);
    assert(it != dict_.end());
    return subBlock(f1, f2, it->second);
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
typename PlainLib_::template TType<Scalar_, Rank + CoRank>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::subBlock(const FusionTree<Rank, Symmetry>& f1,
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

    return Plain::template tensor_from_matrix_block<Scalar, Rank + CoRank>(
        block_[block_number], left_offset_domain, left_offset_codomain, f1.dim, f2.dim, dims);
    // MatrixType submatrix = block_[block_number].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
    // std::cout << "from subblock:" << std::endl << submatrix << std::endl;
    // TensorcMapType tensorview = Plain::cMap(submatrix.data(), dims);
    // TensorType T = Plain::template construct<Scalar, Rank + CoRank>(tensorview);
    // return T;
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
typename PlainLib_::template MType<Scalar_> Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::subMatrix(const FusionTree<Rank, Symmetry>& f1,
                                                                                                        const FusionTree<CoRank, Symmetry>& f2) const
{
    if(f1.q_coupled != f2.q_coupled) { assert(false); }

    const auto left_offset_domain = domain.leftOffset(f1);
    const auto left_offset_codomain = codomain.leftOffset(f2);
    const auto it = dict_.find(f1.q_coupled);

    auto submatrix = Plain::template block<Scalar>(block_[it->second], left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
    // auto submatrix = block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
    return submatrix;
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
typename PlainLib_::template TType<Scalar_, Rank + CoRank> Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::plainTensor() const
{
    SPDLOG_INFO("Entering plainTensor()");
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
    SPDLOG_INFO("sorted everything");

    auto inner_mat = Plain::template construct_with_zero<Scalar>(sorted_domain.fullDim(), sorted_codomain.fullDim(), *world_);
    SPDLOG_INFO("Constructed inner_mat (size={},{}) and perform loop with {} steps.",
                sorted_domain.fullDim(),
                sorted_codomain.fullDim(),
                sorted_sector.size());
    for(std::size_t i = 0; i < sorted_sector.size(); i++) {
        SPDLOG_INFO("step #={}", i);
        auto id_cgc = Plain::template Identity<Scalar>(Symmetry::degeneracy(sorted_sector[i]), Symmetry::degeneracy(sorted_sector[i]), *world_);
        SPDLOG_INFO("Static identity done");
        // SPDLOG_INFO("block[{}]", i);
        // sorted_block[i].print();
        auto mat = Plain::template kronecker_prod<Scalar>(sorted_block[i], id_cgc);
        SPDLOG_INFO("Kronecker Product done.");
        // mat.print();
        Plain::template add_to_block<Scalar>(inner_mat,
                                             sorted_domain.full_outer_num(sorted_sector[i]),
                                             sorted_codomain.full_outer_num(sorted_sector[i]),
                                             Symmetry::degeneracy(sorted_sector[i]) * Plain::template rows<Scalar>(sorted_block[i]),
                                             Symmetry::degeneracy(sorted_sector[i]) * Plain::template cols<Scalar>(sorted_block[i]),
                                             mat);
        SPDLOG_INFO("Block added.");
        // inner_mat.block(sorted_domain.full_outer_num(sorted_sector[i]),
        //                 sorted_codomain.full_outer_num(sorted_sector[i]),
        //                 Symmetry::degeneracy(sorted_sector[i]) * sorted_block[i].rows(),
        //                 Symmetry::degeneracy(sorted_sector[i]) * sorted_block[i].cols()) =
        //     Eigen::kroneckerProduct(sorted_block[i],
        //                             MatrixType::Identity(Symmetry::degeneracy(sorted_sector[i]), Symmetry::degeneracy(sorted_sector[i])));
    }
    // cout << "inner_mat:" << endl << std::fixed << inner_mat << endl;
    std::array<IndexType, 2> full_dims = {static_cast<IndexType>(sorted_domain.fullDim()), static_cast<IndexType>(sorted_codomain.fullDim())};
    // typename Plain::template MapTType<Scalar, 2> map = Plain::Map(inner_mat.data(), full_dims);
    // typename Plain::template TType<Scalar, 2> inner_tensor = Plain::template construct<Scalar, 2>(map);

    typename Plain::template TType<Scalar, 2> inner_tensor = Plain::template tensor_from_matrix_block<Scalar, 2>(
        inner_mat, 0, 0, Plain::template rows<Scalar>(inner_mat), Plain::template cols<Scalar>(inner_mat), full_dims);
    SPDLOG_INFO("constructed inner_tensor");
    //    inner_tensor.print();
    std::array<IndexType, Rank + 1> dims_domain;
    for(size_t i = 0; i < Rank; i++) { dims_domain[i] = sorted_uncoupled_domain[i].fullDim(); }
    dims_domain[Rank] = sorted_domain.fullDim();
    SPDLOG_INFO("dims domain: {}", dims_domain);
    typename Plain::template TType<Scalar, Rank + 1> unitary_domain = Plain::template construct<Scalar>(dims_domain, *world_);
    Plain::template setZero<Scalar, Rank + 1>(unitary_domain);

    for(const auto& [q, num, plain] : sorted_domain) {
        for(const auto& tree : sorted_domain.tree(q)) {
            std::size_t uncoupled_dim = 1;
            for(std::size_t i = 0; i < Rank; i++) { uncoupled_dim *= sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]); }
            MatrixType id = Plain::template construct<Scalar>(uncoupled_dim, uncoupled_dim, *world_);
            Plain::template setIdentity<Scalar>(id);
            // id.setIdentity();
            typename Plain::template cTType<Scalar, 2> Tid_mat = Plain::template tensor_from_matrix_block<Scalar, 2>(
                id,
                0,
                0,
                Plain::template rows<Scalar>(id),
                Plain::template cols<Scalar>(id),
                std::array<IndexType, 2>{Plain::template rows<Scalar>(id), Plain::template cols<Scalar>(id)});

            std::array<IndexType, Rank + 1> dims;
            for(std::size_t i = 0; i < Rank; i++) { dims[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]); }
            dims[Rank] = uncoupled_dim;
            typename Plain::template TType<Scalar, Rank + 1> Tid = Plain::template reshape<Scalar, 2>(Tid_mat, dims);

            auto T = tree.template asTensor<PlainLib>(*world_);
            typename Plain::template TType<Scalar, Rank + 1> Tfull = Plain::template tensorProd<Scalar, Rank + 1>(Tid, T);
            std::array<IndexType, Rank + 1> offsets;
            for(std::size_t i = 0; i < Rank; i++) { offsets[i] = sorted_uncoupled_domain[i].full_outer_num(tree.q_uncoupled[i]); }
            offsets[Rank] = sorted_domain.full_outer_num(q) + sorted_domain.leftOffset(tree) * Symmetry::degeneracy(q);

            std::array<IndexType, Rank + 1> extents;
            for(std::size_t i = 0; i < Rank; i++) {
                extents[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
            }
            extents[Rank] = Plain::template dimensions<Scalar, Rank + 1>(Tfull)[Rank];
            Plain::template setSubTensor<Scalar, Rank + 1>(unitary_domain, offsets, extents, Tfull); // this amounts to =. Do we need +=?
        }
    }
    SPDLOG_INFO("constructed domain unitary");
    // std::cout << "domain" << std::endl;
    //    unitary_domain.print();
    // unitary_domain.for_each_value([](double d) { std::cout << d << std::endl; });

    std::array<IndexType, CoRank + 1> dims_codomain;
    for(size_t i = 0; i < CoRank; i++) { dims_codomain[i] = sorted_uncoupled_codomain[i].fullDim(); }
    dims_codomain[CoRank] = sorted_codomain.fullDim();
    SPDLOG_INFO("dims codomain: {}", dims_codomain);
    typename Plain::template TType<Scalar, CoRank + 1> unitary_codomain = Plain::template construct<Scalar>(dims_codomain, *world_);
    Plain::template setZero<Scalar, CoRank + 1>(unitary_codomain);
    // std::cout << "codomain" << std::endl;
    // unitary_codomain.print();
    for(const auto& [q, num, plain] : sorted_codomain) {
        for(const auto& tree : sorted_codomain.tree(q)) {
            IndexType uncoupled_dim = 1;
            for(std::size_t i = 0; i < CoRank; i++) { uncoupled_dim *= sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]); }
            MatrixType id = Plain::template construct<Scalar>(uncoupled_dim, uncoupled_dim, *world_);
            Plain::template setIdentity<Scalar>(id);
            // id.setIdentity();
            typename Plain::template cTType<Scalar, 2> Tid_mat = Plain::template tensor_from_matrix_block<Scalar, 2>(
                id,
                0,
                0,
                Plain::template rows<Scalar>(id),
                Plain::template cols<Scalar>(id),
                std::array<IndexType, 2>{Plain::template rows<Scalar>(id), Plain::template cols<Scalar>(id)});

            // MatrixType id(uncoupled_dim, uncoupled_dim);
            // id.setIdentity();
            // typename Plain::template cTType<Scalar, 2> Tid_mat =
            //     Plain::template construct<Scalar, 2>(Plain::Map(id.data(), std::array<IndexType, 2>{id.rows(), id.cols()}));

            std::array<IndexType, CoRank + 1> dims;
            for(std::size_t i = 0; i < CoRank; i++) { dims[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]); }
            dims[CoRank] = uncoupled_dim;
            typename Plain::template TType<Scalar, CoRank + 1> Tid = Plain::template reshape<Scalar, 2>(Tid_mat, dims);
            auto T = tree.template asTensor<PlainLib>(*world_);
            typename Plain::template TType<Scalar, CoRank + 1> Tfull = Plain::template tensorProd<Scalar, CoRank + 1>(Tid, T);
            std::array<IndexType, CoRank + 1> offsets;
            for(std::size_t i = 0; i < CoRank; i++) { offsets[i] = sorted_uncoupled_codomain[i].full_outer_num(tree.q_uncoupled[i]); }
            offsets[CoRank] = sorted_codomain.full_outer_num(q) + sorted_codomain.leftOffset(tree) * Symmetry::degeneracy(q);
            std::array<IndexType, CoRank + 1> extents;
            for(std::size_t i = 0; i < CoRank; i++) {
                extents[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
            }
            extents[CoRank] = Plain::template dimensions<Scalar, CoRank + 1>(Tfull)[CoRank];
            Plain::template setSubTensor<Scalar, CoRank + 1>(unitary_codomain, offsets, extents, Tfull); // this amounts to =. Do we need +=?
        }
    }
    SPDLOG_INFO("constructed codomain unitary");
    // std::cout << "codomain" << std::endl;
    //    unitary_codomain.print();
    // unitary_codomain.for_each_value([](double d) { std::cout << d << std::endl; });
    // XPED_MPI_BARRIER(world_->comm);
    std::array<IndexType, Rank + CoRank> dims_result;
    for(size_t i = 0; i < Rank; i++) { dims_result[i] = sorted_uncoupled_domain[i].fullDim(); }
    for(size_t i = 0; i < CoRank; i++) { dims_result[i + Rank] = sorted_uncoupled_codomain[i].fullDim(); }
    TensorType out = Plain::template construct<Scalar>(dims_result, *world_);
    Plain::template setZero<Scalar, Rank + CoRank>(out);

    auto intermediate = Plain::template contract<Scalar, Rank + 1, 2, Rank, 0>(unitary_domain, inner_tensor);
    out = Plain::template contract<Scalar, Rank + 1, CoRank + 1, Rank, CoRank>(intermediate, unitary_codomain);
    return out;
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
void Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>::print(std::ostream& o, bool PRINT_MATRICES) XPED_CONST
{
    // std::stringstream ss;
    o << "domain:" << endl << domain << endl; // << "with trees:" << endl << domain.printTrees() << endl;
    o << "codomain:" << endl << codomain << endl; // << "with trees:" << endl << codomain.printTrees() << endl;
    for(size_t i = 0; i < sector_.size(); i++) {
        o << "Sector with QN=" << Sym::format<Symmetry>(sector_[i]) << endl;
        // if(PRINT_MATRICES) {
        // o << std::fixed << block_[i] << endl;
        // block_[i].print_matrix();
        //     // Plain::template print<Scalar>(block_[i]);
        // }
    }
    // return ss;
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
std::ostream& operator<<(std::ostream& os, XPED_CONST Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>& t)
{
    t.print(os);
    return os;
}

// template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
// Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> operator*(const Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_>& T,
//                                                                   const typename MatrixType_::Scalar& s)
// {
//     Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> Tout(T.uncoupled_domain, T.uncoupled_codomain);
//     for(size_t i = 0; i < T.sector_.size(); i++) { Tout.push_back(T.sector_[i], T.block_[i] * s); }
//     return Tout;
// }

// template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
// Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> operator*(const typename MatrixType_::Scalar& s,
//                                                                   const Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_>& T)
// {
//     Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> Tout(T.uncoupled_domain, T.uncoupled_codomain);
//     for(size_t i = 0; i < T.sector_.size(); i++) { Tout.push_back(T.sector_[i], T.block_[i] * s); }
//     return Tout;
// }

// template <std::size_t Rank, std::size_t CoRank, std::size_t MiddleRank, typename Symmetry, typename MatrixType_, typename TensorLib_>
// Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> operator*(const Xped<Rank, MiddleRank, Symmetry, MatrixType_, TensorLib_>& T1,
//                                                                   const Xped<MiddleRank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
// {
//     assert(T1.codomain == T2.domain);
//     Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> Tout;
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
// Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_>
// operator*(const AdjointXped<Xped<MiddleRank, Rank, Symmetry, MatrixType_, TensorLib_>>& T1,
//           const Xped<MiddleRank, CoRank, Symmetry, MatrixType_, TensorLib_>& T2)
// {
//     assert(T1.coupledCodomain() == T2.coupledDomain());
//     Xped<Scalar_, Rank, CoRank, Symmetry, MatrixLib_, TensorLib_> Tout;
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

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_> operator+(XPED_CONST Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>& T1,
                                                           XPED_CONST Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>& T2)
{
    assert(T1.domain == T2.domain);
    assert(T1.codomain == T2.codomain);
    Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_> Tout;
    Tout.domain = T1.domain;
    Tout.codomain = T1.codomain;
    Tout.uncoupled_domain = T1.uncoupled_domain;
    Tout.uncoupled_codomain = T1.uncoupled_codomain;
    Tout.sector_ = T1.sector_;
    Tout.dict_ = T1.dict_;
    Tout.block_.resize(Tout.sector_.size());
    for(size_t i = 0; i < T1.sector_.size(); i++) {
        auto it = T2.dict_.find(T1.sector_[i]);
        Tout.block_[i] = PlainLib_::template add<Scalar_>(T1.block_[i], T2.block_[it->second]);
    }
    return Tout;
}

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib_>
Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_> operator-(XPED_CONST Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>& T1,
                                                           XPED_CONST Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_>& T2)
{
    assert(T1.domain == T2.domain);
    assert(T1.codomain == T2.codomain);
    Xped<Scalar_, Rank, CoRank, Symmetry, PlainLib_> Tout;
    Tout.domain = T1.domain;
    Tout.codomain = T1.codomain;
    Tout.uncoupled_domain = T1.uncoupled_domain;
    Tout.uncoupled_codomain = T1.uncoupled_codomain;
    Tout.sector_ = T1.sector_;
    Tout.dict_ = T1.dict_;
    Tout.block_.resize(Tout.sector_.size());
    for(size_t i = 0; i < T1.sector_.size(); i++) {
        auto it = T2.dict_.find(T1.sector_[i]);
        Tout.block_[i] = PlainLib_::template difference<Scalar_>(T1.block_[i], T2.block_[it->second]);
    }
    return Tout;
}
