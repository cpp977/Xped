#include <iostream>
#include <string>
#include <array>
#include <unordered_map>

using std::cout;
using std::endl;
using std::size_t;

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
#endif

#include "TOOLS/numeric_limits.h"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

#include "Xped/Core/FusionTree.hpp"
#include "Xped/Core/treepair.hpp"

namespace treepair {
//                           ☐
//                           _
//   c                 c     b
//   |                  \   /
//   |                   \ /
//  / \   ---> coeff*     |
// /   \                  |
// a     b                 a
template <std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank + 1, Symmetry>, FusionTree<CoRank - 1, Symmetry>>, typename Symmetry::Scalar>
turn_right(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2)
{
    static_assert(CoRank > 0);
    assert(t1.q_coupled == t2.q_coupled);

    FusionTree<Rank + 1, Symmetry> t1p;
    std::copy(t1.q_uncoupled.begin(), t1.q_uncoupled.end(), t1p.q_uncoupled.begin());
    t1p.q_uncoupled.back() = Symmetry::conj(t2.q_uncoupled.back());
    std::copy(t1.q_intermediates.begin(), t1.q_intermediates.end(), t1p.q_intermediates.begin());
    if(Rank + 1 > 2) { t1p.q_intermediates.back() = t1.q_coupled; }
    std::copy(t1.IS_DUAL.begin(), t1.IS_DUAL.end(), t1p.IS_DUAL.begin());
    t1p.IS_DUAL.back() = !t2.IS_DUAL.back();
    if(CoRank == 1) {
        t1p.q_coupled = Symmetry::qvacuum();
    } else if(CoRank == 2) {
        t1p.q_coupled = t2.q_uncoupled[0];
    } else {
        t1p.q_coupled = t2.q_intermediates.back();
    }
    std::copy(t1.dims.begin(), t1.dims.end(), t1p.dims.begin());
    t1p.dims.back() = t2.dims.back();
    t1p.computeDim();
    FusionTree<CoRank - 1, Symmetry> t2p;
    std::copy(t2.q_uncoupled.begin(), t2.q_uncoupled.end() - 1, t2p.q_uncoupled.begin());
    std::copy(t2.IS_DUAL.begin(), t2.IS_DUAL.end() - 1, t2p.IS_DUAL.begin());
    if constexpr(CoRank > 2) { std::copy(t2.q_intermediates.begin(), t2.q_intermediates.end() - 1, t2p.q_intermediates.begin()); }
    if(CoRank == 1) {
        t2p.q_coupled = Symmetry::qvacuum();
    } else if(CoRank == 2) {
        t2p.q_coupled = t2.q_uncoupled[0];
    } else {
        t2p.q_coupled = t2.q_intermediates.back();
    }
    std::copy(t2.dims.begin(), t2.dims.end() - 1, t2p.dims.begin());
    t2p.computeDim();
    typename Symmetry::qType a;
    if(CoRank == 1) {
        a = Symmetry::qvacuum();
    } else if(CoRank == 2) {
        a = t2.q_uncoupled[0];
    } else {
        a = t2.q_intermediates.back();
    }
    typename Symmetry::qType b = t2.q_uncoupled.back();
    typename Symmetry::qType c = t2.q_coupled;
    auto coeff = Symmetry::coeff_turn(a, b, c);
    if(t2.IS_DUAL.back()) { coeff *= Symmetry::coeff_FS(Symmetry::conj(b)); }
    std::unordered_map<std::pair<FusionTree<Rank + 1, Symmetry>, FusionTree<CoRank - 1, Symmetry>>, typename Symmetry::Scalar> out;
    if(std::abs(coeff) < mynumeric_limits<typename Symmetry::Scalar>::epsilon()) { return out; }
    out.insert(std::make_pair(std::make_pair(t1p, t2p), coeff));
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - 1, Symmetry>, FusionTree<CoRank + 1, Symmetry>>, typename Symmetry::Scalar>
turn_left(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2)
{
    auto tmp = turn_right(t2, t1);
    std::unordered_map<std::pair<FusionTree<Rank - 1, Symmetry>, FusionTree<CoRank + 1, Symmetry>>, typename Symmetry::Scalar> out;
    for(const auto& [trees, coeff] : tmp) {
        auto [t1p, t2p] = trees;
        out.insert(std::make_pair(std::make_pair(t2p, t1p), coeff));
    }
    return out;
}

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar>
turn(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2)
{
    assert(t1.q_coupled == t2.q_coupled);
    if constexpr(shift > 0) {
        static_assert(shift <= static_cast<int>(Rank));
    } else if constexpr(shift < 0) {
        static_assert(-shift <= static_cast<int>(CoRank));
    }
    if constexpr(shift == 0) {
        std::unordered_map<std::pair<FusionTree<Rank, Symmetry>, FusionTree<CoRank, Symmetry>>, typename Symmetry::Scalar> out;
        out.insert(std::make_pair(std::make_pair(t1, t2), 1.));
        return out;
    } else {
        constexpr std::size_t newRank = Rank - shift;
        constexpr std::size_t newCoRank = CoRank + shift;
        std::unordered_map<std::pair<FusionTree<newRank, Symmetry>, FusionTree<newCoRank, Symmetry>>, typename Symmetry::Scalar> out;
        if constexpr(shift > 0) {
            for(const auto& [trees1, coeff1] : turn_left(t1, t2)) {
                auto [t1p, t2p] = trees1;
                for(const auto& [trees2, coeff2] : turn<shift - 1>(t1p, t2p)) {
                    auto [t1pp, t2pp] = trees2;
                    out.insert(std::make_pair(std::make_pair(t1pp, t2pp), coeff1 * coeff2));
                }
            }
            return out;
        } else {
            for(const auto& [trees1, coeff1] : turn_right(t1, t2)) {
                auto [t1p, t2p] = trees1;
                for(const auto& [trees2, coeff2] : turn<shift + 1>(t1p, t2p)) {
                    auto [t1pp, t2pp] = trees2;
                    out.insert(std::make_pair(std::make_pair(t1pp, t2pp), coeff1 * coeff2));
                }
            }
            return out;
        }
    }
}

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar>
permute(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2, const Permutation& p)
{
#ifdef XPED_CACHE_PERMUTE_OUTPUT
    if constexpr(Symmetry::NON_ABELIAN) {
        if(tree_cache<shift, Rank, CoRank, Symmetry>.cache.contains(std::make_tuple(t1, t2, p))) {
            return tree_cache<shift, Rank, CoRank, Symmetry>.cache.lookup(std::make_tuple(t1, t2, p));
        }
    }
#endif
    assert(t1.q_coupled == t2.q_coupled);

    // transform the permutation. Needed because turn<>() reverses the order of the FusionTree.
    std::array<std::size_t, Rank + CoRank> pi_id;
    std::iota(pi_id.begin(), pi_id.end(), 0ul);
    for(std::size_t i = 0; i < CoRank; i++) { pi_id[i + Rank] = (CoRank - 1) - i + Rank; }
    constexpr std::size_t newRank = Rank - shift;
    constexpr std::size_t newCoRank = CoRank + shift;
    std::array<std::size_t, Rank + CoRank> pi_tmp;
    for(std::size_t i = 0; i < Rank + CoRank; i++) { pi_tmp[i] = pi_id[p.pi[i]]; }
    std::array<std::size_t, Rank + CoRank> pi_corrected;
    std::copy(pi_tmp.begin(), pi_tmp.begin() + newRank, pi_corrected.begin());
    for(std::size_t i = 0; i < newCoRank; i++) { pi_corrected[i + newRank] = pi_tmp[(newCoRank - 1) - i + newRank]; }
    Permutation p_corrected(pi_corrected);

    constexpr int reshift = CoRank + shift;
    std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar> out;
    for(const auto& [trees, coeff1] : turn<-static_cast<int>(CoRank)>(t1, t2)) {
        auto [t1p, trivial] = trees;
        for(const auto& [t1pp, coeff2] : t1p.permute(p_corrected)) {
            for(const auto& [trees_final, coeff3] : turn<reshift>(t1pp, trivial)) {
                if(auto it = out.find(trees_final); it == out.end()) {
                    out.insert(std::make_pair(trees_final, coeff1 * coeff2 * coeff3));
                } else {
                    out[trees_final] += coeff1 * coeff2 * coeff3;
                }
            }
        }
    }
    std::size_t zero_count = 0ul;
    for(const auto& [trees, coeff] : out) {
        if(std::abs(coeff) < 1.e-9) { zero_count++; }
    }
    if(zero_count > 0) { cout << "permute pair operation created #=" << zero_count << " 0s." << endl; }
#ifdef XPED_CACHE_PERMUTE_OUTPUT
    if constexpr(Symmetry::NON_ABELIAN) { tree_cache<shift, Rank, CoRank, Symmetry>.cache.emplace(std::make_tuple(t1, t2, p), out); }
#endif
    return out;
}

} // namespace treepair
