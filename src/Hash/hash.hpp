#ifndef HASH_H_
#define HASH_H_

#include <boost/functional/hash.hpp>

#include "Symmetry/qarray.hpp"

// forward declaration
template <std::size_t Rank, typename Symmetry>
struct FusionTree;
template <std::size_t N>
struct Permutation;

namespace std {
template <size_t Nq>
struct hash<qarray<Nq>>
{
    inline size_t operator()(const qarray<Nq>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

/**Hashes an array of quantum numbers using boost's \p hash_combine.*/
template <size_t Nq, size_t Nlegs>
struct hash<array<qarray<Nq>, Nlegs>>
{
    inline size_t operator()(const array<qarray<Nq>, Nlegs>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

template <size_t Rank, typename Symmetry>
struct hash<FusionTree<Rank, Symmetry>>
{
    inline size_t operator()(const FusionTree<Rank, Symmetry>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

template <size_t Rank, size_t CoRank, typename Symmetry>
struct hash<std::pair<FusionTree<Rank, Symmetry>, FusionTree<CoRank, Symmetry>>>
{
    inline size_t operator()(const std::pair<FusionTree<Rank, Symmetry>, FusionTree<CoRank, Symmetry>>& ix) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, ix.first);
        boost::hash_combine(seed, ix.second);
        return seed;
    }
};

template <size_t Rank, size_t CoRank, typename Symmetry>
struct hash<std::tuple<FusionTree<Rank, Symmetry>, FusionTree<CoRank, Symmetry>, Permutation<Rank + CoRank>>>
{
    inline size_t operator()(const std::tuple<FusionTree<Rank, Symmetry>, FusionTree<CoRank, Symmetry>, Permutation<Rank + CoRank>>& ix) const
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
