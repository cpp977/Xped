#ifdef _OPENMP
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

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/TomlHelpers.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Xped/Core/FusionTree.hpp"
#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/treepair.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

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
        typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(20);
        C.setRandom(20);
        D.setRandom(20);
        E.setRandom(20);
        test_tree_swap(B, C, D, E);
    }

    SUBCASE("U1")
    {
        typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(10);
        C.setRandom(10);
        D.setRandom(10);
        E.setRandom(10);
        test_tree_swap(B, C, D, E);
    }

    SUBCASE("U0")
    {
        typedef Xped::Sym::U0<> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C, D, E;
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
        typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(20);
        C.setRandom(20);
        D.setRandom(20);
        E.setRandom(20);
        test_tree_permute(B, C, D, E);
    }

    SUBCASE("U1")
    {
        typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C, D, E;
        B.setRandom(10);
        C.setRandom(10);
        D.setRandom(10);
        E.setRandom(10);
        test_tree_permute(B, C, D, E);
    }

    SUBCASE("U0")
    {
        typedef Xped::Sym::U0<> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C, D, E;
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
        typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
        typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
        typedef Xped::Sym::U0<> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
        typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
        typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
        typedef Xped::Sym::U0<> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
