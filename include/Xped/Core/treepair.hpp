#ifndef TREEPAIR_H_
#define TREEPAIR_H_

#include <unordered_map>

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
#endif

#include "Xped/Hash/hash.hpp"
#include "Xped/Util/Permutations.hpp"

namespace Xped {

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
turn_right(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2);

template <std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - 1, Symmetry>, FusionTree<CoRank + 1, Symmetry>>, typename Symmetry::Scalar>
turn_left(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2);

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar>
turn(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2);

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar>
permute(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2, const util::Permutation& p);

} // end namespace treepair

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/treepair.cpp"
#endif

#endif
