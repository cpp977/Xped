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

template <std::size_t Nq>
struct std::hash<Xped::qarray<Nq>>
{
    inline std::size_t operator()(const Xped::qarray<Nq>& ix) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

/**Hashes an array of quantum numbers using boost's \p hash_combine.*/
template <std::size_t Nq, std::size_t Nlegs>
struct std::hash<std::array<Xped::qarray<Nq>, Nlegs>>
{
    inline std::size_t operator()(const std::array<Xped::qarray<Nq>, Nlegs>& ix) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

template <std::size_t Rank, typename Symmetry>
struct std::hash<Xped::FusionTree<Rank, Symmetry>>
{
    inline std::size_t operator()(const Xped::FusionTree<Rank, Symmetry>& ix) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, ix);
        return seed;
    }
};

template <std::size_t Rank, std::size_t CoRank, typename Symmetry>
struct std::hash<std::pair<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>>>
{
    inline std::size_t operator()(const std::pair<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>>& ix) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, ix.first);
        boost::hash_combine(seed, ix.second);
        return seed;
    }
};

template <std::size_t Rank, std::size_t CoRank, typename Symmetry>
struct std::hash<std::tuple<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>, Xped::util::Permutation>>
{
    inline std::size_t
    operator()(const std::tuple<Xped::FusionTree<Rank, Symmetry>, Xped::FusionTree<CoRank, Symmetry>, Xped::util::Permutation>& ix) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(ix));
        boost::hash_combine(seed, std::get<1>(ix));
        boost::hash_combine(seed, std::get<2>(ix));
        return seed;
    }
};

#endif
