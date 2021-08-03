#ifdef _OPENMP
#    pragma message("Xped is using OpenMP parallelization")
#    include "omp.h"
#endif

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef XPED_USE_CYCLOPS_TENSOR_LIB
#    define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN

#include "Util/Macros.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    pragma message("Xped is using LRU cache for the output of FusionTree manipulations.")
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Core/FusionTree.hpp"
#include "Core/Qbasis.hpp"
#include "Symmetry/SU2.hpp"
#include "Symmetry/U0.hpp"
#include "Symmetry/U1.hpp"
#include "Symmetry/kind_dummies.hpp"

#include "doctest/doctest.h"
#ifdef XPED_USE_MPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

#include "tree_tests.hpp"

TEST_SUITE_BEGIN("FusionTrees");

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the elementary swap.", 2)
#else
TEST_CASE("Testing the elementary swap.")
#endif
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
        typedef Sym::U0<> Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(5);
        C.setRandom(5);
        D.setRandom(5);
        E.setRandom(5);
        test_tree_swap(B, C, D, E);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the permutation.", 2)
#else
TEST_CASE("Testing the permutation.")
#endif
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
        typedef Sym::U0<> Symmetry;
        Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(5);
        C.setRandom(5);
        D.setRandom(5);
        E.setRandom(5);
        test_tree_permute(B, C, D, E);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the turn operation for FusionTree pairs.", 2)
#else
TEST_CASE("Testing the turn operation for FusionTree pairs.")
#endif
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
        typedef Sym::U0<> Symmetry;
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

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the permutation for FusionTree pairs.", 2)
#else
TEST_CASE("Testing the permutation for FusionTree pairs.")
#endif
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
        typedef Sym::U0<> Symmetry;
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
