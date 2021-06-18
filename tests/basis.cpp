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
#include "doctest/extensions/doctest_mpi.h"

TEST_SUITE_BEGIN("Qbasis");

MPI_TEST_CASE("Testing combine() in Qbasis.", 2)
{

    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(50);
        C.setRandom(50);
        auto BC = B.combine(C);
        CHECK(BC.fullDim() == B.fullDim() * C.fullDim());
        auto BCB = BC.combine(B);
        CHECK(BCB.fullDim() == B.fullDim() * C.fullDim() * B.fullDim());
        auto BCBC = BCB.combine(C);
        CHECK(BCBC.fullDim() == B.fullDim() * C.fullDim() * B.fullDim() * C.fullDim());
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(50);
        C.setRandom(50);
        auto BC = B.combine(C);
        CHECK(BC.fullDim() == B.fullDim() * C.fullDim());
        auto BCB = BC.combine(B);
        CHECK(BCB.fullDim() == B.fullDim() * C.fullDim() * B.fullDim());
        auto BCBC = BCB.combine(C);
        CHECK(BCBC.fullDim() == B.fullDim() * C.fullDim() * B.fullDim() * C.fullDim());
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        B.setRandom(100);
        C.setRandom(100);
        auto BC = B.combine(C);
        CHECK(BC.fullDim() == B.fullDim() * C.fullDim());
        auto BCB = BC.combine(B);
        CHECK(BCB.fullDim() == B.fullDim() * C.fullDim() * B.fullDim());
        auto BCBC = BCB.combine(C);
        CHECK(BCBC.fullDim() == B.fullDim() * C.fullDim() * B.fullDim() * C.fullDim());
    }
}

TEST_SUITE_END();
