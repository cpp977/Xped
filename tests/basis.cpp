#include <iostream>
#include <cstddef>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

using std::size_t;
using std::cout;
using std::endl;
using std::string;

#include "TextTable.h"

#include "macros.h"

template<std::size_t Rank, typename Symmetry> struct FusionTree;
template<std::size_t N> struct Permutation;

#ifdef CACHE_PERMUTE_OUTPUT
#include "lru/lru.hpp"

template<int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
struct CacheManager
{
        typedef FusionTree<CoRank, Symmetry> CoTree;
        typedef FusionTree<CoRank+shift, Symmetry> NewCoTree;
        typedef FusionTree<Rank, Symmetry> Tree;
        typedef FusionTree<Rank-shift, Symmetry> NewTree;
        typedef typename Symmetry::Scalar Scalar;
        
        typedef LRU::Cache<std::tuple<Tree, CoTree, Permutation<Rank+CoRank> >, std::unordered_map<std::pair<NewTree, NewCoTree >, Scalar> >  CacheType;
        CacheManager(std::size_t cache_size) {
                cache = CacheType(cache_size);
                // cache.monitor();
        }
        CacheType cache;
};

template<int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
CacheManager<shift, Rank, CoRank, Symmetry> tree_cache(100);
#endif

#include "../src/Qbasis.hpp"
#include "../src/symmetry/kind_dummies.hpp"
#include "../src/symmetry/SU2.hpp"
#include "../src/symmetry/U1.hpp"
#include "../src/symmetry/U0.hpp"

#include "doctest/doctest.h"

TEST_SUITE_BEGIN("Qbasis");

TEST_CASE("Testing combine() in Qbasis.") {

        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(50); C.setRandom(50);
                auto BC = B.combine(C);
                CHECK(BC.fullDim() == B.fullDim()*C.fullDim());
                auto BCB = BC.combine(B);
                CHECK(BCB.fullDim() == B.fullDim()*C.fullDim()*B.fullDim());
                auto BCBC = BCB.combine(C);
                CHECK(BCBC.fullDim() == B.fullDim()*C.fullDim()*B.fullDim()*C.fullDim());
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(50); C.setRandom(50);                
                auto BC = B.combine(C);
                CHECK(BC.fullDim() == B.fullDim()*C.fullDim());
                auto BCB = BC.combine(B);
                CHECK(BCB.fullDim() == B.fullDim()*C.fullDim()*B.fullDim());
                auto BCBC = BCB.combine(C);
                CHECK(BCBC.fullDim() == B.fullDim()*C.fullDim()*B.fullDim()*C.fullDim());
        }
        
        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(100); C.setRandom(100);
                auto BC = B.combine(C);
                CHECK(BC.fullDim() == B.fullDim()*C.fullDim());
                auto BCB = BC.combine(B);
                CHECK(BCB.fullDim() == B.fullDim()*C.fullDim()*B.fullDim());
                auto BCBC = BCB.combine(C);
                CHECK(BCBC.fullDim() == B.fullDim()*C.fullDim()*B.fullDim()*C.fullDim());
        }
}

TEST_SUITE_END();
