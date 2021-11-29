#include "Xped/Core/XpedHelper.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace XpedHelper {

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
        auto [domain_, discard] = build_FusionTree_Helper(uncoupled[0], basis_domain_shrinked);
        return domain_;
    }
}

} // namespace XpedHelper
