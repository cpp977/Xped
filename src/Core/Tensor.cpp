#include <iostream>
#include <sstream>
#include <unordered_set>

#include "spdlog/spdlog.h"

#include "fmt/os.h"
#include "fmt/ranges.h"

#include "Xped/Util/Constfct.hpp"
#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Random.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

using std::cout;
using std::endl;
using std::size_t;

#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
void Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::setRandom()
{
    storage_.resize();
    SPDLOG_TRACE("Entering setRandom().");
    SPDLOG_TRACE("Start randomization loop with #={} iterations.", sector().size());
    for(size_t i = 0; i < sector().size(); ++i) {
        PlainInterface::setRandom(block(i));
        SPDLOG_TRACE("Set block #={} to random.", i);
    }
    SPDLOG_TRACE("Leaving setRandom().");
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
void Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::setZero()
{
    storage_.resize();
    SPDLOG_TRACE("Entering setZero().");
    SPDLOG_TRACE("Start loop with #={} iterations.", sector_.size());
    for(size_t i = 0; i < sector().size(); ++i) {
        PlainInterface::setZero(block(i));
        SPDLOG_TRACE("Set block #={} to zero.", i);
    }
    SPDLOG_TRACE("Leaving setZero().");
}
template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
void Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::setIdentity()
{
    storage_.resize();
    SPDLOG_TRACE("Entering setIdentity().");
    SPDLOG_TRACE("Start loop with #={} iterations.", sector_.size());
    for(size_t i = 0; i < sector().size(); ++i) {
        PlainInterface::setIdentity(block(i));
        SPDLOG_TRACE("Set block #={} to zero.", i);
    }
    SPDLOG_TRACE("Leaving setIdentity().");
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
void Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::setConstant(Scalar val)
{
    storage_.resize();
    SPDLOG_TRACE("Entering setConstant().");
    SPDLOG_TRACE("Start loop with #={} iterations.", sector_.size());
    for(size_t i = 0; i < sector().size(); ++i) {
        PlainInterface::setConstant(block(i), val);
        SPDLOG_TRACE("Set block #={} to zero.", i);
    }
    SPDLOG_TRACE("Leaving setConstant().");
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <std::size_t... pds, std::size_t... pcs>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::permute_impl(seq::iseq<std::size_t, pds...> pd,
                                                                              seq::iseq<std::size_t, pcs...> pc) const
{
    std::array<std::size_t, Rank> arr_domain = {pds...};
    std::array<std::size_t, CoRank> arr_codomain = {(pcs - Rank)...};
    util::Permutation p_domain(arr_domain);
    util::Permutation p_codomain(arr_codomain);

    std::array<IndexType, Rank + CoRank> arr_total;
    auto it_total = std::copy(p_domain.pi.begin(), p_domain.pi.end(), arr_total.begin());
    auto pi_codomain_shifted = p_codomain.pi;
    std::for_each(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), [](std::size_t& elem) { return elem += Rank; });
    std::copy(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), it_total);

    auto new_domain = uncoupledDomain();
    p_domain.apply(new_domain);

    auto new_codomain = uncoupledCodomain();
    p_codomain.apply(new_codomain);

    Self out(new_domain, new_codomain, this->world_);
    out.setZero();

    for(size_t i = 0; i < sector().size(); ++i) {
        auto domain_trees = coupledDomain().tree(sector(i));
        auto codomain_trees = coupledCodomain().tree(sector(i));
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
                auto permuted_domain_trees = domain_tree.permute(p_domain);
                auto permuted_codomain_trees = codomain_tree.permute(p_codomain);

#ifdef XPED_MEMORY_EFFICIENT
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = PlainInterface::shuffle_view<decltype(tensor), pds..., pcs...>(tensor);
#elif defined(XPED_TIME_EFFICIENT)
                auto tensor = this->subBlock(domain_tree, codomain_tree);
                auto Tshuffle = PlainInterface::shuffle<Scalar, Rank + CoRank, pds..., pcs...>(tensor);
#endif

                for(const auto& [permuted_domain_tree, coeff_domain] : permuted_domain_trees)
                    for(const auto& [permuted_codomain_tree, coeff_codomain] : permuted_codomain_trees) {
                        if(std::abs(coeff_domain * coeff_codomain) < 1.e-10) { continue; }

                        auto it = out.dict().find(sector(i));
                        assert(it != out.dict().end());
#ifdef XPED_MEMORY_EFFICIENT
                        auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                        PlainInterface::addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
#elif defined(XPED_TIME_EFFICIENT)
                        IndexType row = out.coupledDomain().leftOffset(permuted_domain_tree);
                        IndexType col = out.coupledCodomain().leftOffset(permuted_codomain_tree);
                        IndexType rows = permuted_domain_tree.dim;
                        IndexType cols = permuted_codomain_tree.dim;
                        PlainInterface::add_to_block_from_tensor<Rank + CoRank>(
                            out.block(it->second), row, col, rows, cols, coeff_domain * coeff_codomain, Tshuffle);
#endif
                    }
            }
    }
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <int shift, std::size_t... ps>
Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::permute_impl(seq::iseq<std::size_t, ps...> per) const
{
    std::array<std::size_t, Rank + CoRank> p_ = {ps...};
    util::Permutation p(p_);
    constexpr std::size_t newRank = Rank - shift;
    constexpr std::size_t newCoRank = CoRank + shift;
    std::array<Qbasis<Symmetry, 1, AllocationPolicy>, newRank> new_domain;
    std::array<Qbasis<Symmetry, 1, AllocationPolicy>, newCoRank> new_codomain;

    for(std::size_t i = 0; i < newRank; ++i) {
        if(p.pi[i] > Rank - 1 or Rank == 0) {
            new_domain[i] = uncoupledCodomain()[p.pi[i] - Rank].conj();
        } else {
            new_domain[i] = uncoupledDomain()[p.pi[i]];
        }
    }

    for(std::size_t i = 0; i < newCoRank; ++i) {
        if(p.pi[i + newRank] > Rank - 1 or Rank == 0) {
            new_codomain[i] = uncoupledCodomain()[p.pi[i + newRank] - Rank];
        } else {
            new_codomain[i] = uncoupledDomain()[p.pi[i + newRank]].conj();
        }
    }

    Tensor<Scalar, newRank, newCoRank, Symmetry, false, AllocationPolicy> out(new_domain, new_codomain, this->world_);
    out.setZero();

    for(size_t i = 0; i < sector().size(); ++i) {
        auto domain_trees = coupledDomain().tree(sector(i));
        auto codomain_trees = coupledCodomain().tree(sector(i));
        for(const auto& domain_tree : domain_trees)
            for(const auto& codomain_tree : codomain_trees) {
#ifdef XPED_MEMORY_EFFICIENT
                auto tensor = this->view(domain_tree, codomain_tree);
                auto Tshuffle = PlainInterface::shuffle_view<decltype(tensor), ps...>(tensor);
#elif defined(XPED_TIME_EFFICIENT)
                auto tensor = this->subBlock(domain_tree, codomain_tree);
                auto Tshuffle = PlainInterface::shuffle<Scalar, Rank + CoRank, ps...>(tensor);
#endif

                for(const auto& [permuted_trees, coeff] : treepair::permute<shift>(domain_tree, codomain_tree, p)) {
                    if(std::abs(coeff) < 1.e-10) { continue; }

                    auto [permuted_domain_tree, permuted_codomain_tree] = permuted_trees;
                    assert(permuted_domain_tree.q_coupled == permuted_codomain_tree.q_coupled);

                    auto it = out.dict().find(permuted_domain_tree.q_coupled);
                    assert(it != out.dict().end());
#ifdef XPED_MEMORY_EFFICIENT
                    auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
                    PlainInterface::addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
#elif defined(XPED_TIME_EFFICIENT)
                    IndexType row = out.coupledDomain().leftOffset(permuted_domain_tree);
                    IndexType col = out.coupledCodomain().leftOffset(permuted_codomain_tree);
                    IndexType rows = permuted_domain_tree.dim;
                    IndexType cols = permuted_codomain_tree.dim;
                    PlainInterface::add_to_block_from_tensor<Rank + CoRank>(out.block(it->second), row, col, rows, cols, coeff, Tshuffle);
#endif
                }
            }
    }
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <int shift, std::size_t... p>
Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, false, AllocationPolicy>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::permute() const
{
    // if constexpr(Rank + CoRank == 0) { return *this; }
    static_assert(std::cmp_greater_equal(Rank, shift), "Invalid call to permute()"); // c++ - 20
    static_assert(std::cmp_greater_equal(CoRank, -shift), "Invalid call to permute()"); // c++ - 20
    using s = seq::iseq<std::size_t, p...>;
    if constexpr(shift == 0 and std::is_same_v<s, seq::make<std::size_t, Rank + CoRank>>) { return *this; }

    using p_domain = seq::take<Rank - shift, s>;
    using p_codomain = seq::after<Rank - shift, s>;

    if constexpr(seq::filter<util::constFct::isGreaterOrEqual<Rank>, p_codomain>::size() == p_codomain::size() and
                 seq::filter<util::constFct::isSmaller<Rank>, p_domain>::size() == p_domain::size() and shift == 0) {
        return permute_impl(seq::take<Rank, s>{}, seq::after<Rank, s>{});
    } else {
        return permute_impl<shift>(s{});
    }
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <bool>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::twist(std::size_t leg) const
{
    if constexpr(not Symmetry::IS_FERMIONIC) { return *this; }
    SPDLOG_INFO("Performing twist of leg={}", leg);
    Self out = *this;
    for(std::size_t i = 0; i < out.sector().size(); ++i) {
        auto domain_trees = out.coupledDomain().tree(out.sector(i));
        auto codomain_trees = out.coupledCodomain().tree(out.sector(i));
        for(const auto& domain_tree : domain_trees) {
            for(const auto& codomain_tree : codomain_trees) {
                auto coeff =
                    (leg < Rank) ? Symmetry::coeff_twist(domain_tree.q_uncoupled[leg]) : Symmetry::coeff_twist(codomain_tree.q_uncoupled[leg - Rank]);
                out.subMatrix(domain_tree, codomain_tree) *= coeff;
            }
        }
    }
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <std::size_t leg>
Tensor<Scalar, util::constFct::trimDim<Rank>(leg), Rank + CoRank - 1 - util::constFct::trimDim<Rank>(leg), Symmetry, false, AllocationPolicy>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::trim() const
{
    constexpr std::size_t new_rank = util::constFct::trimDim<Rank>(leg);
    constexpr std::size_t new_corank = Rank + CoRank - 1 - util::constFct::trimDim<Rank>(leg);
    std::array<Qbasis<Symmetry, 1>, new_rank> uncoupled_domain;
    std::array<Qbasis<Symmetry, 1>, new_corank> uncoupled_codomain;
    std::size_t count = 0;
    for(std::size_t r = 0; r < Rank; ++r) {
        if(r == leg) { continue; }
        uncoupled_domain[count++] = uncoupledDomain()[r];
    }
    count = 0;
    for(std::size_t c = Rank; c < Rank + CoRank; ++c) {
        if(c == leg) { continue; }
        uncoupled_codomain[count++] = uncoupledCodomain()[c - Rank];
    }

    Tensor<Scalar, util::constFct::trimDim<Rank>(leg), Rank + CoRank - 1 - util::constFct::trimDim<Rank>(leg), Symmetry, false, AllocationPolicy> out(
        uncoupled_domain, uncoupled_codomain, this->data(), this->plainSize(), world());
    // out.setZero();
    // out.storage().m_data = this->storage().m_data;
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
template <std::size_t... legs>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::shiftQN(qType charge) const
{
    std::array<std::size_t, sizeof...(legs)> arr_legs = {legs...};

    std::array<Qbasis<Symmetry, 1>, Rank> uncoupled_domain;
    std::array<Qbasis<Symmetry, 1>, CoRank> uncoupled_codomain;
    for(std::size_t r = 0; r < Rank; ++r) {
        if(auto it = std::find(arr_legs.begin(), arr_legs.end(), r); it == arr_legs.end()) {
            uncoupled_domain[r] = uncoupledDomain()[r];
        } else {
            uncoupled_domain[r] = uncoupledDomain()[r].shift(charge);
        }
    }
    for(std::size_t c = Rank; c < Rank + CoRank; ++c) {
        if(auto it = std::find(arr_legs.begin(), arr_legs.end(), c); it == arr_legs.end()) {
            uncoupled_codomain[c - Rank] = uncoupledCodomain()[c - Rank];
        } else {
            uncoupled_codomain[c - Rank] = uncoupledCodomain()[c - Rank].shift(charge);
        }
    }

    Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy> out(
        uncoupled_domain, uncoupled_codomain, this->data(), this->plainSize(), world());
    // out.setZero();
    // out.storage().m_data = this->storage().m_data;
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
std::tuple<Tensor<Scalar, Rank, 1, Symmetry, false, AllocationPolicy>,
           Tensor<typename ScalarTraits<Scalar>::Real, 1, 1, Symmetry, false, AllocationPolicy>,
           Tensor<Scalar, 1, CoRank, Symmetry, false, AllocationPolicy>>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::tSVD(size_t maxKeep,
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
    for(size_t i = 0; i < sector().size(); ++i) {
        middle.push_back(sector(i), std::min(PlainInterface::rows(block(i)), PlainInterface::cols(block(i))));
    }

    Tensor<Scalar, Rank, 1, Symmetry, false, AllocationPolicy> U(uncoupledDomain(), {{middle}});
    Tensor<RealScalar, 1, 1, Symmetry, false, AllocationPolicy> Sigma({{middle}}, {{middle}});
    Tensor<Scalar, 1, CoRank, Symmetry, false, AllocationPolicy> Vdag({{middle}}, uncoupledCodomain());

    std::vector<std::pair<typename Symmetry::qType, RealScalar>> allSV;
    SPDLOG_INFO("Performing the svd loop (size={})", sector().size());
    for(size_t i = 0; i < sector().size(); ++i) {
        SPDLOG_INFO("Step i={} for mat with dim=({},{})", i, PlainInterface::rows<Scalar>(block(i)), PlainInterface::rows<Scalar>(block(i)));
        auto [Umat, Sigmavec, Vmatdag] = PlainInterface::svd(block(i));
        SPDLOG_INFO("Performed svd for step i={}", i);
        std::vector<Scalar> svs;
        PlainInterface::vec_to_stdvec<Scalar>(Sigmavec, svs);

        for(const auto& sv : svs) {
            SPDLOG_INFO("Move the element {} from sigma to allSV", sv);
            allSV.push_back(std::make_pair(sector(i), sv));
        }
        SPDLOG_INFO("Extracted singular values for step i={}", i);
        auto Sigmamat = PlainInterface::vec_to_diagmat<Scalar>(Sigmavec);
        U.push_back(sector(i), Umat);
        Sigma.push_back(sector(i), Sigmamat);
        Vdag.push_back(sector(i), Vmatdag);
    }
    size_t numberOfStates = allSV.size();
    assert(numberOfStates > 0);
    auto first_entry = allSV[0];
    SPDLOG_INFO("numberOfStates={}", numberOfStates);
    SPDLOG_INFO("allSV={}\n", allSV);
    std::sort(allSV.begin(),
              allSV.end(),
              [](const std::pair<typename Symmetry::qType, double>& sv1, const std::pair<typename Symmetry::qType, double>& sv2) {
                  return sv1.second > sv2.second;
              });
    SPDLOG_INFO("numberOfStates after sort {}", allSV.size());
    for(size_t i = maxKeep; i < allSV.size(); ++i) { truncWeight += Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second), 2.); }
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
    if(allSV.size() == 0) { allSV.push_back(first_entry); }

    std::map<typename Symmetry::qType, std::vector<Scalar>> qn_orderedSV;
    Qbasis<Symmetry, 1> truncBasis;
    for(const auto& [q, s] : allSV) {
        // truncBasis.push_back(q, 1ul);
        qn_orderedSV[q].push_back(s);
        entropy += -Symmetry::degeneracy(q) * s * s * std::log(s * s);
    }
    for(const auto& [q, vec_sv] : qn_orderedSV) { truncBasis.push_back(q, vec_sv.size()); }
    SPDLOG_INFO("Set up the truncated basis.");
    std::stringstream ss;
    ss << truncBasis.print();
    SPDLOG_INFO(ss.str());

    Tensor<Scalar, Rank, 1, Symmetry, false, AllocationPolicy> trunc_U(uncoupledDomain(), {{truncBasis}});
    Tensor<RealScalar, 1, 1, Symmetry, false, AllocationPolicy> trunc_Sigma({{truncBasis}}, {{truncBasis}});
    Tensor<Scalar, 1, CoRank, Symmetry, false, AllocationPolicy> trunc_Vdag({{truncBasis}}, uncoupledCodomain());
    SPDLOG_INFO("Starting the loop for truncating U,S,V (size={})", qn_orderedSV.size());
    for(const auto& [q, vec_sv] : qn_orderedSV) {
        SPDLOG_INFO("Step with q={}", q.data[0]);
        size_t Nret = vec_sv.size();
        // cout << "q=" << q << ", Nret=" << Nret << endl;
        auto itSigma = Sigma.dict().find({q});
        SPDLOG_INFO("Searched the dict of Sigma.");
        auto sigma_mat = PlainInterface::block(Sigma.block(itSigma->second), 0, 0, Nret, Nret);
        SPDLOG_INFO("Got subblock of Sigma.");
        trunc_Sigma.push_back(q, sigma_mat);
        // if(RETURN_SPEC) { SVspec.insert(std::make_pair(q, Sigma.block_[itSigma->second].diagonal().head(Nret).real())); }
        SPDLOG_INFO("Before return spec.");
        if(RETURN_SPEC) {
            VectorType spec;
            PlainInterface::diagonal_head_matrix_to_vector(spec, Sigma.block(itSigma->second), Nret);
            SVspec.insert(std::make_pair(q, spec));
        }
        SPDLOG_INFO("After return spec.");
        auto itU = U.dict().find({q});
        trunc_U.push_back(q, PlainInterface::block(U.block(itU->second), 0, 0, PlainInterface::rows(U.block(itU->second)), Nret));
        auto itVdag = Vdag.dict().find({q});
        trunc_Vdag.push_back(q, PlainInterface::block(Vdag.block(itVdag->second), 0, 0, Nret, PlainInterface::cols(Vdag.block(itVdag->second))));
    }
    SPDLOG_INFO("Leaving Xped::tSVD()");
    return std::make_tuple(trunc_U, trunc_Sigma, trunc_Vdag);
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
std::pair<Tensor<typename ScalarTraits<Scalar>::Real, 1, 1, Symmetry, false, AllocationPolicy>,
          Tensor<typename ScalarTraits<Scalar>::Real, Rank, 1, Symmetry, false, AllocationPolicy>>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::eigh() XPED_CONST
{
    static_assert(Rank == CoRank, "Eigenvalue decomposition only possible for tensors with Rank==CoRank");
    assert(coupledDomain() == coupledCodomain() and "Eigenvalue decomposition only possible for square matrices.");
    // assert(*this == this->adjoint().eval() and "Input for eigh() needs to be Hermitian.");

    Tensor<RealScalar, 1, 1, Symmetry, false, AllocationPolicy> D(
        {{coupledDomain().forgetHistory()}}, {{coupledCodomain().forgetHistory()}}, world());
    Tensor<RealScalar, Rank, 1, Symmetry, false, AllocationPolicy> V(uncoupledDomain(), {{coupledCodomain().forgetHistory()}}, world());

    for(size_t i = 0; i < sector().size(); ++i) {
        auto [Eigvals, Eigvecs] = PlainInterface::eigh(block(i));

        D.push_back(sector(i), Eigvals);
        V.push_back(sector(i), Eigvecs);
    }

    return std::make_pair(D, V);
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
//         const auto it = dict().find(f1.q_coupled);
//         return block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixLib_, typename TensorLib_>
// Eigen::Map<MatrixLib_> Xped<Scalar_, Rank, CoRank, Symmetry_>::
// operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
// {
//         if(f1.q_coupled != f2.q_coupled) {return util::zero_init<MatrixType>();}
//         std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
//         std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
//         const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
//         const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
//         const auto it = dict().find(f1.q_coupled);
//         return block_[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
auto Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::view(const FusionTree<Rank, Symmetry>& f1,
                                                                           const FusionTree<CoRank, Symmetry>& f2)
{
    const auto it = dict().find(f1.q_coupled);
    assert(it != dict().end());
    return view(f1, f2, it->second);
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
auto Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::view(const FusionTree<Rank, Symmetry>& f1,
                                                                           const FusionTree<CoRank, Symmetry>& f2,
                                                                           std::size_t block_number)
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);
    std::array<IndexType, Rank + CoRank> dims;
    for(size_t i = 0; i < Rank; ++i) {
        assert(f1.dims[i] == uncoupledDomain()[i].inner_dim(f1.q_uncoupled[i]));
        dims[i] = f1.dims[i];
    }
    for(size_t i = 0; i < CoRank; ++i) {
        assert(f2.dims[i] == uncoupledCodomain()[i].inner_dim(f2.q_uncoupled[i]));
        dims[i + Rank] = f2.dims[i];
    }

    IndexType left_offset_domain = coupledDomain().leftOffset(f1);
    IndexType left_offset_codomain = coupledCodomain().leftOffset(f2);
#ifdef XPED_USE_EIGEN_TENSOR_LIB
    Eigen::TensorMap<Eigen::Tensor<double, 2>> tmat(block(block_number).data(),
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
    for(size_t i = 1; i < Rank; ++i) {
        shape_data[i - 1].set_extent(dims[i]);
        shape_data[i - 1].set_stride(std::accumulate(dims.begin(), dims.begin() + i, 1ul, std::multiplies<Scalar>()));
    }
    size_t start = (Rank > 0) ? 0ul : 1ul;
    double stride_correction = (Rank > 0) ? block(block_number).rows() : 1.;
    for(size_t i = start; i < CoRank; ++i) {
        shape_data[i + Rank - 1].set_extent(dims[i + Rank]);
        shape_data[i + Rank - 1].set_stride(stride_correction *
                                            std::accumulate(dims.begin() + Rank, dims.begin() + Rank + i, 1ul, std::multiplies<Scalar>()));
    }
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), PlainInterface::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block(block_number).rows() + left_offset_domain;
    TensorMapType out(block(block_number).data() + total_offset, block_shape);
    return out;
#endif
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
auto Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::view(const FusionTree<Rank, Symmetry>& f1,
                                                                           const FusionTree<CoRank, Symmetry>& f2) const
{
    const auto it = dict().find(f1.q_coupled);
    assert(it != dict().end());
    return view(f1, f2, it->second);
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
auto Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::view(const FusionTree<Rank, Symmetry>& f1,
                                                                           const FusionTree<CoRank, Symmetry>& f2,
                                                                           std::size_t block_number) const
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);
    std::array<IndexType, Rank + CoRank> dims;
    for(size_t i = 0; i < Rank; ++i) { dims[i] = f1.dims[i]; } // ncoupledDomain()[i].inner_dim(f1.q_uncoupled[i]); }
    for(size_t i = 0; i < CoRank; ++i) { dims[i + Rank] = f2.dims[i]; } // uncoupledCodomain()[i].inner_dim(f2.q_uncoupled[i]); }

    IndexType left_offset_domain = coupledDomain().leftOffset(f1);
    IndexType left_offset_codomain = coupledCodomain().leftOffset(f2);

#ifdef XPED_USE_EIGEN_TENSOR_LIB
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> tmat(block(block_number).data(),
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
    for(size_t i = 1; i < Rank; ++i) {
        shape_data[i - 1].set_extent(dims[i]);
        shape_data[i - 1].set_stride(std::accumulate(dims.begin(), dims.begin() + i, 1ul, std::multiplies<Scalar>()));
    }
    size_t start = (Rank > 0) ? 0ul : 1ul;
    double stride_correction = (Rank > 0) ? block(block_number).rows() : 1.;
    for(size_t i = start; i < CoRank; ++i) {
        shape_data[i + Rank - 1].set_extent(dims[i + Rank]);
        shape_data[i + Rank - 1].set_stride(stride_correction *
                                            std::accumulate(dims.begin() + Rank, dims.begin() + Rank + i, 1ul, std::multiplies<Scalar>()));
    }
    auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), PlainInterface::as_tuple(shape_data));

    nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

    const auto total_offset = left_offset_codomain * block(block_number).rows() + left_offset_domain;
    TensorcMapType out(block(block_number).data() + total_offset, block_shape);
    return out;
#endif
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
typename PlainInterface::TType<Scalar, Rank + CoRank>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::subBlock(const FusionTree<Rank, Symmetry>& f1,
                                                                          const FusionTree<CoRank, Symmetry>& f2) const
{
    const auto it = dict().find(f1.q_coupled);
    assert(it != dict().end());
    return subBlock(f1, f2, it->second);
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
typename PlainInterface::TType<Scalar, Rank + CoRank>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::subBlock(const FusionTree<Rank, Symmetry>& f1,
                                                                          const FusionTree<CoRank, Symmetry>& f2,
                                                                          std::size_t block_number) const
{
    assert(block_number < sector().size());
    assert(f1.q_coupled == f2.q_coupled);
    assert(sector(block_number) == f1.q_coupled);

    const auto left_offset_domain = coupledDomain().leftOffset(f1);
    const auto left_offset_codomain = coupledCodomain().leftOffset(f2);
    std::array<IndexType, Rank + CoRank> dims;

    for(size_t i = 0; i < Rank; ++i) { dims[i] = uncoupledDomain()[i].inner_dim(f1.q_uncoupled[i]); }
    for(size_t i = 0; i < CoRank; ++i) { dims[i + Rank] = uncoupledCodomain()[i].inner_dim(f2.q_uncoupled[i]); }

    return PlainInterface::template tensor_from_matrix_block<Rank + CoRank>(
        block(block_number), left_offset_domain, left_offset_codomain, f1.dim, f2.dim, dims);
    // MatrixType submatrix = block_[block_number].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
    // std::cout << "from subblock:" << std::endl << submatrix << std::endl;
    // TensorcMapType tensorview = PlainInterface::cMap(submatrix.data(), dims);
    // TensorType T = PlainInterface::construct<Scalar, Rank + CoRank>(tensorview);
    // return T;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
typename PlainInterface::TType<Scalar, Rank + CoRank> Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::plainTensor() const
{
    SPDLOG_INFO("Entering plainTensor()");
    auto sorted_domain = coupledDomain();
    sorted_domain.sort();
    auto sorted_codomain = coupledCodomain();
    sorted_codomain.sort();
    auto sorted_uncoupled_domain = uncoupledDomain();
    std::for_each(sorted_uncoupled_domain.begin(), sorted_uncoupled_domain.end(), [](Qbasis<Symmetry, 1>& q) { q.sort(); });
    auto sorted_uncoupled_codomain = uncoupledCodomain();
    std::for_each(sorted_uncoupled_codomain.begin(), sorted_uncoupled_codomain.end(), [](Qbasis<Symmetry, 1>& q) { q.sort(); });

    std::vector<std::size_t> index_sort(sector().size());
    std::iota(index_sort.begin(), index_sort.end(), 0);
    std::sort(index_sort.begin(), index_sort.end(), [this](std::size_t n1, std::size_t n2) {
        qarray<Symmetry::Nq> q1 = sector(n1);
        qarray<Symmetry::Nq> q2 = sector(n2);
        return Symmetry::compare(q1, q2);
    });

    auto sorted_sector = sector();
    std::vector<MatrixType> sorted_block(sorted_sector.size());
    for(std::size_t i = 0; i < sector().size(); ++i) {
        sorted_sector[i] = sector(index_sort[i]);
        sorted_block[i] = block(index_sort[i]);
    }
    SPDLOG_INFO("sorted everything");

    auto inner_mat = PlainInterface::construct_with_zero<Scalar>(sorted_domain.fullDim(), sorted_codomain.fullDim(), *world_);
    SPDLOG_INFO("Constructed inner_mat (size={},{}) and perform loop with {} steps.",
                sorted_domain.fullDim(),
                sorted_codomain.fullDim(),
                sorted_sector.size());
    for(std::size_t i = 0; i < sorted_sector.size(); ++i) {
        SPDLOG_INFO("step #={}", i);
        auto id_cgc = PlainInterface::Identity<Scalar>(Symmetry::degeneracy(sorted_sector[i]), Symmetry::degeneracy(sorted_sector[i]), *world_);
        SPDLOG_INFO("Static identity done");
        // SPDLOG_INFO("block[{}]", i);
        // sorted_block[i].print();
        auto mat = PlainInterface::kronecker_prod(sorted_block[i], id_cgc);
        SPDLOG_INFO("Kronecker Product done.");
        // mat.print();
        PlainInterface::add_to_block(inner_mat,
                                     sorted_domain.full_outer_num(sorted_sector[i]),
                                     sorted_codomain.full_outer_num(sorted_sector[i]),
                                     Symmetry::degeneracy(sorted_sector[i]) * PlainInterface::rows(sorted_block[i]),
                                     Symmetry::degeneracy(sorted_sector[i]) * PlainInterface::cols(sorted_block[i]),
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
    // typename PlainInterface::MapTType<Scalar, 2> map = PlainInterface::Map(inner_mat.data(), full_dims);
    // typename PlainInterface::TType<Scalar, 2> inner_tensor = PlainInterface::construct<Scalar, 2>(map);

    typename PlainInterface::TType<Scalar, 2> inner_tensor =
        PlainInterface::tensor_from_matrix_block<2>(inner_mat, 0, 0, PlainInterface::rows(inner_mat), PlainInterface::cols(inner_mat), full_dims);
    SPDLOG_INFO("constructed inner_tensor");
    //    inner_tensor.print();
    std::array<IndexType, Rank + 1> dims_domain;
    for(size_t i = 0; i < Rank; ++i) { dims_domain[i] = sorted_uncoupled_domain[i].fullDim(); }
    dims_domain[Rank] = sorted_domain.fullDim();
    SPDLOG_INFO("dims domain: {}", dims_domain);
    typename PlainInterface::TType<Scalar, Rank + 1> unitary_domain = PlainInterface::construct<Scalar>(dims_domain, *world_);
    PlainInterface::setZero<Scalar, Rank + 1>(unitary_domain);

    for(const auto& [q, num, plain] : sorted_domain) {
        for(const auto& tree : sorted_domain.tree(q)) {
            std::size_t uncoupled_dim = 1;
            for(std::size_t i = 0; i < Rank; ++i) { uncoupled_dim *= sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]); }
            MatrixType id = PlainInterface::construct<Scalar>(uncoupled_dim, uncoupled_dim, *world_);
            PlainInterface::setIdentity(id);
            // id.setIdentity();
            typename PlainInterface::cTType<Scalar, 2> Tid_mat =
                PlainInterface::tensor_from_matrix_block<2>(id,
                                                            0,
                                                            0,
                                                            PlainInterface::rows(id),
                                                            PlainInterface::cols(id),
                                                            std::array<IndexType, 2>{PlainInterface::rows(id), PlainInterface::cols(id)});

            std::array<IndexType, Rank + 1> dims;
            for(std::size_t i = 0; i < Rank; ++i) { dims[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]); }
            dims[Rank] = uncoupled_dim;
            typename PlainInterface::TType<Scalar, Rank + 1> Tid = PlainInterface::reshape<Scalar, 2>(Tid_mat, dims);

            auto T = tree.template asTensor<PlainInterface>(*world_);
            typename PlainInterface::TType<Scalar, Rank + 1> Tfull = PlainInterface::tensorProd<Scalar, Rank + 1>(Tid, T);
            std::array<IndexType, Rank + 1> offsets;
            for(std::size_t i = 0; i < Rank; ++i) { offsets[i] = sorted_uncoupled_domain[i].full_outer_num(tree.q_uncoupled[i]); }
            offsets[Rank] = sorted_domain.full_outer_num(q) + sorted_domain.leftOffset(tree) * Symmetry::degeneracy(q);

            std::array<IndexType, Rank + 1> extents;
            for(std::size_t i = 0; i < Rank; ++i) {
                extents[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
            }
            extents[Rank] = PlainInterface::dimensions<Scalar, Rank + 1>(Tfull)[Rank];
            PlainInterface::setSubTensor<Scalar, Rank + 1>(unitary_domain, offsets, extents, Tfull); // this amounts to =. Do we need +=?
        }
    }
    SPDLOG_INFO("constructed domain unitary");
    // std::cout << "domain" << std::endl;
    //    unitary_domain.print();
    // unitary_domain.for_each_value([](double d) { std::cout << d << std::endl; });

    std::array<IndexType, CoRank + 1> dims_codomain;
    for(size_t i = 0; i < CoRank; ++i) { dims_codomain[i] = sorted_uncoupled_codomain[i].fullDim(); }
    dims_codomain[CoRank] = sorted_codomain.fullDim();
    SPDLOG_INFO("dims codomain: {}", dims_codomain);
    typename PlainInterface::TType<Scalar, CoRank + 1> unitary_codomain = PlainInterface::construct<Scalar>(dims_codomain, *world_);
    PlainInterface::setZero<Scalar, CoRank + 1>(unitary_codomain);
    // std::cout << "codomain" << std::endl;
    // unitary_codomain.print();
    for(const auto& [q, num, plain] : sorted_codomain) {
        for(const auto& tree : sorted_codomain.tree(q)) {
            IndexType uncoupled_dim = 1;
            for(std::size_t i = 0; i < CoRank; ++i) { uncoupled_dim *= sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]); }
            MatrixType id = PlainInterface::construct<Scalar>(uncoupled_dim, uncoupled_dim, *world_);
            PlainInterface::setIdentity(id);
            // id.setIdentity();
            typename PlainInterface::cTType<Scalar, 2> Tid_mat =
                PlainInterface::tensor_from_matrix_block<2>(id,
                                                            0,
                                                            0,
                                                            PlainInterface::rows(id),
                                                            PlainInterface::cols(id),
                                                            std::array<IndexType, 2>{PlainInterface::rows(id), PlainInterface::cols(id)});

            // MatrixType id(uncoupled_dim, uncoupled_dim);
            // id.setIdentity();
            // typename PlainInterface::cTType<Scalar, 2> Tid_mat =
            //     PlainInterface::construct<Scalar, 2>(PlainInterface::Map(id.data(), std::array<IndexType, 2>{id.rows(), id.cols()}));

            std::array<IndexType, CoRank + 1> dims;
            for(std::size_t i = 0; i < CoRank; ++i) { dims[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]); }
            dims[CoRank] = uncoupled_dim;
            typename PlainInterface::TType<Scalar, CoRank + 1> Tid = PlainInterface::reshape<Scalar, 2>(Tid_mat, dims);
            auto T = tree.template asTensor<PlainInterface>(*world_);
            typename PlainInterface::TType<Scalar, CoRank + 1> Tfull = PlainInterface::tensorProd<Scalar, CoRank + 1>(Tid, T);
            std::array<IndexType, CoRank + 1> offsets;
            for(std::size_t i = 0; i < CoRank; ++i) { offsets[i] = sorted_uncoupled_codomain[i].full_outer_num(tree.q_uncoupled[i]); }
            offsets[CoRank] = sorted_codomain.full_outer_num(q) + sorted_codomain.leftOffset(tree) * Symmetry::degeneracy(q);
            std::array<IndexType, CoRank + 1> extents;
            for(std::size_t i = 0; i < CoRank; ++i) {
                extents[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
            }
            extents[CoRank] = PlainInterface::dimensions<Scalar, CoRank + 1>(Tfull)[CoRank];
            PlainInterface::setSubTensor<Scalar, CoRank + 1>(unitary_codomain, offsets, extents, Tfull); // this amounts to =. Do we need +=?
        }
    }
    SPDLOG_INFO("constructed codomain unitary");
    // std::cout << "codomain" << std::endl;
    //    unitary_codomain.print();
    // unitary_codomain.for_each_value([](double d) { std::cout << d << std::endl; });
    // XPED_MPI_BARRIER(world_->comm);
    std::array<IndexType, Rank + CoRank> dims_result;
    for(size_t i = 0; i < Rank; ++i) { dims_result[i] = sorted_uncoupled_domain[i].fullDim(); }
    for(size_t i = 0; i < CoRank; ++i) { dims_result[i + Rank] = sorted_uncoupled_codomain[i].fullDim(); }
    TensorType out = PlainInterface::construct<Scalar>(dims_result, *world_);
    PlainInterface::setZero<Scalar, Rank + CoRank>(out);

    auto intermediate = PlainInterface::contract<Scalar, Rank + 1, 2, Rank, 0>(unitary_domain, inner_tensor);
    out = PlainInterface::contract<Scalar, Rank + 1, CoRank + 1, Rank, CoRank>(intermediate, unitary_codomain);
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
void Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>::print(std::ostream& o, bool PRINT_MATRICES) const
{
    // std::stringstream ss;
    fmt::print(o, "Tensor<{},{}>: domain=[", Rank, CoRank);
    if(Rank == 0) o << "], codomain[";
    for(auto i = 0ul; i < Rank; ++i) {
        o << uncoupledDomain()[i].info();
        i < Rank - 1 ? o << ", " : o << "], codomain[";
    }
    if(CoRank == 0) { o << "]"; }
    for(auto i = 0ul; i < CoRank; ++i) {
        o << uncoupledCodomain()[i].info();
        i < CoRank - 1 ? o << ", " : o << "]";
    }

    // o << "domain:" << endl << coupledDomain() << endl; // << "with trees:" << endl << domain.printTrees() << endl;
    // o << "codomain:" << endl << coupledCodomain() << endl; // << "with trees:" << endl << codomain.printTrees() << endl;
    if(PRINT_MATRICES) {
        o << std::endl;
        for(std::size_t i = 0; i < sector().size(); ++i) {
            fmt::print(o, "Sector i={} with QN={}\n", i, Sym::format<Symmetry>(sector(i)));
            o << std::fixed << std::setprecision(12) << block(i) << std::endl;
            // storage_.block(i).print_matrix();
            // PlainInterface::print<Scalar>(storage_.block(i));
        }
    }
    // return ss;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
std::ostream& operator<<(std::ostream& os, XPED_CONST Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>& t)
{
    t.print(os);
    return os;
}

} // namespace Xped
