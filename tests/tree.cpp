#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <TextTable.h>

#include "macros.h"

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "ArgParser.h"

template <std::size_t Rank, typename Symmetry>
struct FusionTree;
template <std::size_t N>
struct Permutation;

#ifdef CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
struct CacheManager
{
    typedef FusionTree<CoRank, Symmetry> CoTree;
    typedef FusionTree<CoRank + shift, Symmetry> NewCoTree;
    typedef FusionTree<Rank, Symmetry> Tree;
    typedef FusionTree<Rank - shift, Symmetry> NewTree;
    typedef typename Symmetry::Scalar Scalar;

    typedef LRU::Cache<std::tuple<Tree, CoTree, Permutation<Rank + CoRank>>, std::unordered_map<std::pair<NewTree, NewCoTree>, Scalar>> CacheType;
    CacheManager(std::size_t cache_size)
    {
        cache = CacheType(cache_size);
        // cache.monitor();
    }
    CacheType cache;
};

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
CacheManager<shift, Rank, CoRank, Symmetry> tree_cache(100);
#endif

#include "../src/FusionTree.hpp"
#include "../src/Qbasis.hpp"
#include "../src/symmetry/SU2.hpp"
#include "../src/symmetry/U0.hpp"
#include "../src/symmetry/U1.hpp"
#include "../src/symmetry/kind_dummies.hpp"

#include "doctest/doctest.h"

#include "tree_tests.hpp"

TEST_SUITE_BEGIN("FusionTrees");

TEST_CASE("Testing the elementary swap.")
{
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(20);
        C.setRandom(20);
        D.setRandom(20);
        E.setRandom(20);
        test_tree_swap(B, C, D, E);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(10);
        C.setRandom(10);
        D.setRandom(10);
        E.setRandom(10);
        test_tree_swap(B, C, D, E);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0 Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(5);
        C.setRandom(5);
        D.setRandom(5);
        E.setRandom(5);
        test_tree_swap(B, C, D, E);
    }
}

TEST_CASE("Testing the permutation.")
{
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(20);
        C.setRandom(20);
        D.setRandom(20);
        E.setRandom(20);
        test_tree_permute(B, C, D, E);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(10);
        C.setRandom(10);
        D.setRandom(10);
        E.setRandom(10);
        test_tree_permute(B, C, D, E);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0 Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(5);
        C.setRandom(5);
        D.setRandom(5);
        E.setRandom(5);
        test_tree_permute(B, C, D, E);
    }
}

TEST_CASE("Testing the turn operation for FusionTree pairs.")
{
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(20);
        C.setRandom(20);
        test_tree_pair_turn<-2>(B, C);
        test_tree_pair_turn<-1>(B, C);
        test_tree_pair_turn<+0>(B, C);
        test_tree_pair_turn<+1>(B, C);
        test_tree_pair_turn<+2>(B, C);
    }
    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(10);
        C.setRandom(10);
        test_tree_pair_turn<-2>(B, C);
        test_tree_pair_turn<-1>(B, C);
        test_tree_pair_turn<+0>(B, C);
        test_tree_pair_turn<+1>(B, C);
        test_tree_pair_turn<+2>(B, C);
    }
    SUBCASE("U0")
    {
        typedef Sym::U0 Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(5);
        C.setRandom(5);
        test_tree_pair_turn<-2>(B, C);
        test_tree_pair_turn<-1>(B, C);
        test_tree_pair_turn<+0>(B, C);
        test_tree_pair_turn<+1>(B, C);
        test_tree_pair_turn<+2>(B, C);
    }
}

TEST_CASE("Testing the permutation for FusionTree pairs.")
{
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(20);
        C.setRandom(20);
        test_tree_pair_permute<-2>(B, C);
        test_tree_pair_permute<-1>(B, C);
        test_tree_pair_permute<+0>(B, C);
        test_tree_pair_permute<+1>(B, C);
        test_tree_pair_permute<+2>(B, C);
    }
    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(10);
        C.setRandom(10);
        test_tree_pair_permute<-2>(B, C);
        test_tree_pair_permute<-1>(B, C);
        test_tree_pair_permute<+0>(B, C);
        test_tree_pair_permute<+1>(B, C);
        test_tree_pair_permute<+2>(B, C);
    }
    SUBCASE("U0")
    {
        typedef Sym::U0 Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(5);
        C.setRandom(5);
        test_tree_pair_permute<-2>(B, C);
        test_tree_pair_permute<-1>(B, C);
        test_tree_pair_permute<+0>(B, C);
        test_tree_pair_permute<+1>(B, C);
        test_tree_pair_permute<+2>(B, C);
    }
}

TEST_SUITE_END();
