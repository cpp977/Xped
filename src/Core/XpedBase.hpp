#ifndef XPED_BASE_H_
#define XPED_BASE_H_

#include "Core/XpedHelper.hpp"
#include "Interfaces/PlainInterface.hpp"
#include "Util/Constfct.hpp"

template <typename Derived>
struct XpedTraits
{};

// forward declarations
template <typename XprType>
class AdjointOp;

template <typename XprType>
class ScaledOp;

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib>
class Xped;

template <typename Derived>
class XpedBase
{
public:
    typedef typename XpedTraits<Derived>::Scalar Scalar;
    typedef typename XpedTraits<Derived>::Symmetry Symmetry;
    typedef typename XpedTraits<Derived>::PlainLib PlainLib;
    typedef typename XpedTraits<Derived>::MatrixType MatrixType;
    typedef typename XpedTraits<Derived>::TensorType TensorType;
    typedef typename XpedTraits<Derived>::VectorType VectorType;

    static constexpr std::size_t Rank = XpedTraits<Derived>::Rank;
    static constexpr std::size_t CoRank = XpedTraits<Derived>::CoRank;
    typedef typename PlainLib::template MapTType<Scalar, Rank + CoRank> TensorMapType;
    typedef typename PlainLib::template cMapTType<Scalar, Rank + CoRank> TensorcMapType;
    typedef typename PlainLib::Indextype IndexType;

    const ScaledOp<Derived> operator*(const Scalar scale) const { return ScaledOp(derived, scale); }

    XPED_CONST AdjointOp<Derived> adjoint() XPED_CONST { return AdjointOp<Derived>(derived()); }

    template <typename OtherDerived>
    auto operator*(OtherDerived&& other) XPED_CONST;

    template <int shift, std::size_t...>
    auto permute() const;

    // auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;
    // auto view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2, std::size_t block_number) const;

    TensorType subBlock(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const;

    Scalar trace() XPED_CONST;

    Scalar squaredNorm() XPED_CONST { return (*this * this->adjoint()).trace(); }

    Scalar norm() XPED_CONST { return std::sqrt(squaredNorm()); }

    Xped<Scalar, Rank, CoRank, Symmetry, PlainLib> eval() const { return Xped<Scalar, Rank, CoRank, Symmetry, PlainLib>(derived()); };

protected:
    template <typename Scalar, std::size_t Rank__, std::size_t CoRank__, typename Symmetry__, typename PlainLib__>
    friend class Xped;
    template <typename OtherDerived>
    friend class XpedBase;

    const Derived& derived() const { return *static_cast<const Derived*>(this); }
    Derived& derived() { return *static_cast<Derived*>(this); }

    template <std::size_t... p_domain, std::size_t... p_codomain>
    auto permute_impl(seq::iseq<std::size_t, p_domain...> pd, seq::iseq<std::size_t, p_codomain...> pc) const;

    template <int shift, std::size_t... ps>
    auto permute_impl(seq::iseq<std::size_t, ps...> per) const;
};

template <typename Derived>
typename XpedTraits<Derived>::Scalar XpedBase<Derived>::trace() XPED_CONST
{
    assert(derived().coupledDomain() == derived().coupledCodomain());
    Scalar out = 0.;
    for(size_t i = 0; i < derived().sector().size(); i++) {
        out += PlainLib::template trace<Scalar>(derived().block(i)) * Symmetry::degeneracy(derived().sector(i));
        // out += derived().block(i).trace() * Symmetry::degeneracy(derived().sector(i));
    }
    return out;
}

// template <typename Derived>
// template <std::size_t... pds, std::size_t... pcs>
// auto XpedBase<Derived>::permute_impl(seq::iseq<std::size_t, pds...> pd, seq::iseq<std::size_t, pcs...> pc) const
// {
//     auto derived_ref = derived();
//     std::array<std::size_t, Rank> pdomain_ = {pds...};
//     std::array<std::size_t, CoRank> pcodomain_ = {(pcs - Rank)...};
//     Permutation<Rank> p_domain(pdomain_);
//     Permutation<CoRank> p_codomain(pcodomain_);

//     std::array<IndexType, Rank + CoRank> total_p;
//     auto it_total = std::copy(p_domain.pi.begin(), p_domain.pi.end(), total_p.begin());
//     auto pi_codomain_shifted = p_codomain.pi;
//     std::for_each(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), [](std::size_t& elem) { return elem += Rank; });
//     std::copy(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), it_total);
//     Xped<Rank, CoRank, Symmetry, MatrixType, TensorLib> out;
//     out.uncoupled_codomain = derived_ref.uncoupledCodomain();
//     p_codomain.apply(out.uncoupled_codomain);

//     out.uncoupled_domain = derived_ref.uncoupledDomain();
//     p_domain.apply(out.uncoupled_domain);

//     out.domain = util::build_FusionTree(out.uncoupled_domain);
//     out.codomain = util::build_FusionTree(out.uncoupled_codomain);

//     for(size_t i = 0; i < derived_ref.sector().size(); i++) {
//         auto domain_trees = derived_ref.domainTrees(derived_ref.sector(i));
//         auto codomain_trees = derived_ref.codomainTrees(derived_ref.sector(i));
//         for(const auto& domain_tree : domain_trees)
//             for(const auto& codomain_tree : codomain_trees) {
//                 auto permuted_domain_trees = domain_tree.permute(p_domain);
//                 auto permuted_codomain_trees = codomain_tree.permute(p_codomain);
//                 auto tensor = this->view(domain_tree, codomain_tree);
//                 auto Tshuffle = Plain::template shuffle_view<decltype(tensor), pds..., pcs...>(tensor);
//                 for(const auto& [permuted_domain_tree, coeff_domain] : permuted_domain_trees)
//                     for(const auto& [permuted_codomain_tree, coeff_codomain] : permuted_codomain_trees) {
//                         if(std::abs(coeff_domain * coeff_codomain) < 1.e-10) { continue; }

//                         auto it = out.dict_.find(derived_ref.sector(i));
//                         if(it == out.dict_.end()) {
//                             MatrixType mat(out.domain.inner_dim(derived_ref.sector(i)), out.codomain.inner_dim(derived_ref.sector(i)));
//                             mat.setZero();
//                             out.push_back(derived_ref.sector(i), mat);
//                             auto t = out.view(permuted_domain_tree, permuted_codomain_tree, i);
//                             Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
//                         } else {
//                             auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
//                             Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff_domain * coeff_codomain);
//                         }
//                     }
//             }
//     }
//     return out;
// }

// template <typename Derived>
// template <int shift, std::size_t... ps>
// auto XpedBase<Derived>::permute_impl(seq::iseq<std::size_t, ps...> per) const
// {
//     auto derived_ref = derived();
//     std::array<std::size_t, Rank + CoRank> p_ = {ps...};
//     Permutation<Rank + CoRank> p(p_);
//     constexpr std::size_t newRank = Rank - shift;
//     constexpr std::size_t newCoRank = CoRank + shift;
//     Xped<newRank, newCoRank, Symmetry, MatrixType, TensorLib> out;
//     for(std::size_t i = 0; i < newRank; i++) {
//         if(p.pi[i] > Rank - 1) {
//             out.uncoupled_domain[i] = derived_ref.uncoupledCodomain()[p.pi[i] - Rank].conj();
//         } else {
//             out.uncoupled_domain[i] = derived_ref.uncoupledDomain()[p.pi[i]];
//         }
//     }

//     for(std::size_t i = 0; i < newCoRank; i++) {
//         if(p.pi[i + newRank] > Rank - 1) {
//             out.uncoupled_codomain[i] = derived_ref.uncoupledCodomain()[p.pi[i + newRank] - Rank];
//         } else {
//             out.uncoupled_codomain[i] = derived_ref.uncoupledDomain()[p.pi[i + newRank]].conj();
//         }
//     }

//     out.domain = util::build_FusionTree(out.uncoupled_domain);
//     out.codomain = util::build_FusionTree(out.uncoupled_codomain);

//     for(size_t i = 0; i < derived_ref.sector().size(); i++) {
//         auto domain_trees = derived_ref.domainTrees(derived_ref.sector(i));
//         auto codomain_trees = derived_ref.codomainTrees(derived_ref.sector(i));
//         for(const auto& domain_tree : domain_trees)
//             for(const auto& codomain_tree : codomain_trees) {
//                 auto tensor = this->view(domain_tree, codomain_tree);
//                 auto Tshuffle = Plain::template shuffle_view<decltype(tensor), ps...>(tensor);
//                 for(const auto& [permuted_trees, coeff] : treepair::permute<shift>(domain_tree, codomain_tree, p)) {
//                     if(std::abs(coeff) < 1.e-10) { continue; }

//                     auto [permuted_domain_tree, permuted_codomain_tree] = permuted_trees;
//                     assert(permuted_domain_tree.q_coupled == permuted_codomain_tree.q_coupled);

//                     auto it = out.dict_.find(permuted_domain_tree.q_coupled);
//                     if(it == out.dict_.end()) {
//                         MatrixType mat(out.domain.inner_dim(permuted_domain_tree.q_coupled),
//                         out.codomain.inner_dim(permuted_domain_tree.q_coupled)); mat.setZero(); out.push_back(permuted_domain_tree.q_coupled, mat);
//                         auto t = out.view(permuted_domain_tree, permuted_codomain_tree, out.block_.size() - 1);
//                         Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
//                     } else {
//                         auto t = out.view(permuted_domain_tree, permuted_codomain_tree, it->second);
//                         Plain::template addScale<Scalar, Rank + CoRank>(Tshuffle, t, coeff);
//                     }
//                 }
//             }
//     }
//     return out;
// }

// template <typename Derived>
// template <int shift, std::size_t... p>
// auto XpedBase<Derived>::permute() const
// {
//     using s = seq::iseq<std::size_t, p...>;
//     using p_domain = seq::take<Rank - shift, s>;
//     using p_codomain = seq::after<Rank - shift, s>;

//     if constexpr(seq::filter<util::constFct::isGreaterOrEqual<Rank>, p_codomain>::size() == p_codomain::size() and
//                  seq::filter<util::constFct::isSmaller<Rank>, p_domain>::size() == p_domain::size() and shift == 0) {
//         return permute_impl(seq::take<Rank, s>{}, seq::after<Rank, s>{});
//     } else {
//         return permute_impl<shift>(s{});
//     }
// }

template <typename Derived>
template <typename OtherDerived>
auto XpedBase<Derived>::operator*(OtherDerived&& other) XPED_CONST
{
    typedef typename std::remove_const<std::remove_reference_t<OtherDerived>>::type OtherDerived_;
    static_assert(CoRank == XpedTraits<OtherDerived_>::Rank);
    auto derived_ref = derived();
    auto other_derived_ref = other.derived();
    assert(derived_ref.coupledCodomain() == other_derived_ref.coupledDomain());
    Xped<Scalar, Rank, XpedTraits<OtherDerived_>::CoRank, Symmetry, PlainLib> Tout;
    Tout.domain = derived_ref.coupledDomain();
    Tout.codomain = other_derived_ref.coupledCodomain();
    Tout.uncoupled_domain = derived_ref.uncoupledDomain();
    Tout.uncoupled_codomain = other_derived_ref.uncoupledCodomain();
    // Tout.sector = T1.sector;
    // Tout.dict = T1.dict;
    // Tout.block_.resize(Tout.sector_.size());
    std::unordered_set<typename Symmetry::qType> uniqueController;
    auto other_dict = other_derived_ref.dict();
    auto this_dict = derived_ref.dict();
    for(size_t i = 0; i < derived_ref.sector().size(); i++) {
        uniqueController.insert(derived_ref.sector(i));
        auto it = other_dict.find(derived_ref.sector(i));
        if(it == other_dict.end()) { continue; }
        // Tout.push_back(derived_ref.sector(i), Plain::template prod<Scalar>(derived_ref.block(i), other_derived_ref.block(it->second)));
        Tout.push_back(derived_ref.sector(i), PlainLib::template prod<Scalar>(derived_ref.block(i), other_derived_ref.block(it->second)));
        // Tout.block_[i] = T1.block_[i] * T2.block_[it->second];
    }
    for(size_t i = 0; i < other_derived_ref.sector().size(); i++) {
        if(auto it = uniqueController.find(other_derived_ref.sector(i)); it != uniqueController.end()) { continue; }
        auto it = this_dict.find(other_derived_ref.sector(i));
        if(it == this_dict.end()) { continue; }
        // Tout.push_back(other_derived_ref.sector(i), Plain::template prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
        Tout.push_back(other_derived_ref.sector(i), PlainLib::template prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
    }
    return Tout;
}

// template <typename Derived>
// auto XpedBase<Derived>::view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2) const
// {
//     auto this_dict = derived().dict();
//     const auto it = this_dict.find(f1.q_coupled);
//     assert(it != this_dict.end());
//     return view(f1, f2, it->second);
// }

// template <typename Derived>
// auto XpedBase<Derived>::view(const FusionTree<Rank, Symmetry>& f1, const FusionTree<CoRank, Symmetry>& f2, std::size_t block_number) const
// {
//     auto derived_ref = derived();
//     assert(block_number < derived_ref.sector().size());
//     assert(f1.q_coupled == f2.q_coupled);
//     assert(derived_ref.sector(block_number) == f1.q_coupled);
//     std::array<Eigen::Index, Rank + CoRank> dims;
//     for(size_t i = 0; i < Rank; i++) { dims[i] = derived_ref.uncoupledDomain()[i].inner_dim(f1.q_uncoupled[i]); }
//     for(size_t i = 0; i < CoRank; i++) { dims[i + Rank] = derived_ref.uncoupledCodomain()[i].inner_dim(f2.q_uncoupled[i]); }

//     IndexType left_offset_domain = derived_ref.coupledDomain().leftOffset(f1);
//     IndexType left_offset_codomain = derived_ref.coupledCodomain().leftOffset(f2);

// #ifdef XPED_USE_EIGEN_TENSOR_LIB
//     auto dataptr = derived_ref.block(block_number).data();
//     Eigen::TensorMap<const Eigen::Tensor<double, 2>> tmat(dataptr, {derived_ref.block(block_number).rows(),
//     derived_ref.block(block_number).cols()}); return tmat
//         .slice(std::array<Eigen::Index, 2>{left_offset_domain, left_offset_codomain},
//                std::array<Eigen::Index, 2>{static_cast<Eigen::Index>(f1.dim), static_cast<Eigen::Index>(f2.dim)})
//         .reshape(dims);
// #endif

// #ifdef XPED_USE_ARRAY_TENSOR_LIB
//     nda::dim<-9, -9, 1> first_dim;
//     first_dim.set_extent(dims[0]);
//     std::array<nda::dim<-9, -9, -9>, Rank + CoRank - 1> shape_data;
//     for(size_t i = 1; i < Rank; i++) {
//         shape_data[i - 1].set_extent(dims[i]);
//         shape_data[i - 1].set_stride(std::accumulate(dims.begin(), dims.begin() + i, 1ul, std::multiplies<Scalar>()));
//     }
//     size_t start = (Rank > 0) ? 0ul : 1ul;
//     double stride_correction = (Rank > 0) ? derived_ref.block(block_number).rows() : 1.;
//     for(size_t i = start; i < CoRank; i++) {
//         shape_data[i + Rank - 1].set_extent(dims[i + Rank]);
//         shape_data[i + Rank - 1].set_stride(stride_correction *
//                                             std::accumulate(dims.begin() + Rank, dims.begin() + Rank + i, 1ul, std::multiplies<Scalar>()));
//     }
//     auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), Plain::as_tuple(shape_data));

//     nda::dense_shape<Rank + CoRank> block_shape(dims_tuple);

//     const auto total_offset = left_offset_codomain * derived_ref.block(block_number).rows() + left_offset_domain;
//     auto dataptr = derived_ref.block(block_number).data();
//     TensorcMapType out(dataptr + total_offset, block_shape);
//     return out;
// #endif
// }

// template <typename Derived>
// typename XpedTraits<Derived>::TensorType XpedBase<Derived>::subBlock(const FusionTree<Rank, Symmetry>& f1,
//                                                                      const FusionTree<CoRank, Symmetry>& f2) const
// {
//     auto derived_ref = derived();
//     if(f1.q_coupled != f2.q_coupled) { assert(false); }

//     const auto left_offset_domain = derived_ref.coupledDomain().leftOffset(f1);
//     const auto left_offset_codomain = derived_ref.coupledCodomain().leftOffset(f2);
//     const auto it = derived_ref.dict().find(f1.q_coupled);
//     std::array<IndexType, Rank + CoRank> dims;

//     for(size_t i = 0; i < Rank; i++) { dims[i] = derived_ref.uncoupledDomain()[i].inner_dim(f1.q_uncoupled[i]); }
//     for(size_t i = 0; i < CoRank; i++) { dims[i + Rank] = derived_ref.uncoupledCodomain()[i].inner_dim(f2.q_uncoupled[i]); }

//     MatrixType submatrix = derived_ref.block(it->second).block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
//     // std::cout << "from subblock:" << std::endl << submatrix << std::endl;
//     TensorcMapType tensorview = Plain::cMap(submatrix.data(), dims);
//     TensorType T = Plain::template construct<Scalar, Rank + CoRank>(tensorview);
//     return T;
// }

#endif
