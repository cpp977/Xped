#ifndef HASH_H_
#define HASH_H_

#include <boost/functional/hash.hpp>

#include "Xped/Symmetry/qarray.hpp"

// forward declaration
namespace Xped {
template <std::size_t Rank, typename Symmetry>
struct FusionTree;

namespace util {
struct Permutation;
}
} // namespace Xped

namespace std {
template <size_t Nq>
struct hash<Xped::qarray<Nq>>
{
    inline size_t operator()(const Xped::qarray<Nq>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

/**Hashes an array of quantum numbers using boost's \p hash_combine.*/
template <size_t Nq, size_t Nlegs>
struct hash<array<Xped::qarray<Nq>, Nlegs>>
{
    inline size_t operator()(const array<Xped::qarray<Nq>, Nlegs>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

template <size_t Rank, typename Symmetry>
struct hash<Xped::FusionTree<Rank, Symmetry>>
{
    inline size_t operator()(const Xped::FusionTree<Rank, Symmetry>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

template <size_t Rank, size_t CoRank, typename Symmetry>
struct hash<std::pair<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>>>
{
    inline size_t operator()(const std::pair<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix.first);
        boost::hash_combine(seed, ix.second);
        return seed;
    }
};

template <size_t Rank, size_t CoRank, typename Symmetry>
struct hash<std::tuple<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>, Xped::util::Permutation>>
{
    inline size_t
    operator()(const std::tuple<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>, Xped::util::Permutation>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(ix));
        boost::hash_combine(seed, std::get<1>(ix));
        boost::hash_combine(seed, std::get<2>(ix));
        return seed;
    }
};
} // namespace std
#endif
