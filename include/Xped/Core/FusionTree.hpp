#ifndef FUSIONTREE_H_
#define FUSIONTREE_H_

#include <unordered_map>

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/Hash/hash.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/Permutations.hpp"

namespace Xped {

namespace util {
constexpr std::size_t inter_dim(std::size_t Rank) { return (Rank == 1 or Rank == 0) ? 0 : Rank - 2; }
constexpr std::size_t mult_dim(std::size_t Rank) { return (Rank == 0) ? 0 : Rank - 1; }
constexpr std::size_t numberOfVertices(std::size_t Rank) { return (Rank == 0) ? 0 : Rank - 1; }
constexpr std::size_t numberOfInnerlines(std::size_t Rank) { return (Rank == 1 or Rank == 0) ? 0 : Rank - 2; }
} // namespace util

template <std::size_t Rank, typename Symmetry>
struct FusionTree
{
    typedef typename Symmetry::qType qType;
    typedef typename Symmetry::Scalar Scalar;

    std::array<qType, Rank> q_uncoupled;
    qType q_coupled;
    std::size_t dim;
    std::array<size_t, Rank> dims;
    std::array<qType, util::inter_dim(Rank)> q_intermediates;
    std::array<size_t, util::mult_dim(Rank)> multiplicities =
        std::array<size_t, util::mult_dim(Rank)>(); // only for non-Abelian symmetries with outermultiplicity.
    std::array<bool, Rank> IS_DUAL{};

    template <typename Ar>
    inline void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("FusionTree",
                           ("q_uncoupled", q_uncoupled),
                           ("q_coupled", q_coupled),
                           ("dim", dim),
                           ("dims", dims),
                           ("q_intermediates", q_intermediates),
                           ("multiplicities", multiplicities),
                           ("IS_DUAL", IS_DUAL));
    }

    inline void computeDim() { dim = std::accumulate(dims.begin(), dims.end(), 1ul, std::multiplies<std::size_t>()); }

    void computeIntermediates();

    bool operator<(const FusionTree<Rank, Symmetry>& other) const;
    bool operator>(const FusionTree<Rank, Symmetry>& other) const;

    bool operator==(const FusionTree<Rank, Symmetry>& other) const;

    inline bool operator!=(const FusionTree<Rank, Symmetry>& other) const { return !this->operator==(other); }

    friend std::size_t hash_value(const FusionTree<Rank, Symmetry>& tree)
    {
        size_t seed = 0;
        boost::hash_combine(seed, tree.q_uncoupled);
        boost::hash_combine(seed, tree.q_coupled);
        boost::hash_combine(seed, tree.q_intermediates);
        boost::hash_combine(seed, tree.IS_DUAL);
        boost::hash_combine(seed, tree.multiplicities);
        boost::hash_combine(seed, tree.dims);
        return seed;
    }

    std::string printTree(const std::array<std::string, Rank>& s_uncoupled,
                          const std::array<std::string, util::inter_dim(Rank)>& s_intermediates,
                          const std::string& s_coupled,
                          const std::array<std::string, util::mult_dim(Rank)>& s_multiplicities) const;

    std::string draw() const;

    std::string print() const;

    template <typename PlainLib>
    typename PlainLib::template TType<Scalar, Rank + 1> asTensor(const mpi::XpedWorld& world = mpi::getUniverse()) const;

    FusionTree<Rank + 1, Symmetry> enlarge(const FusionTree<1, Symmetry>& other) const;

    std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> permute(const util::Permutation& p) const;

    std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> swap(const std::size_t& pos) const; // swaps sites pos and pos+1
};

template <std::size_t depth, typename Symmetry>
std::ostream& operator<<(std::ostream& os, const FusionTree<depth, Symmetry>& tree)
{
    os << tree.print();
    return os;
}

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/FusionTree.cpp"
#endif

#endif
