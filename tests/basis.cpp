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
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "doctest/doctest.h"
#ifdef XPED_USE_MPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

TEST_SUITE_BEGIN("Qbasis");

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing combine() in Qbasis.", 2)
#else
TEST_CASE("Testing combine() in Qbasis.")
#endif
{
    SUBCASE("SU2")
    {
        typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
        typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
        typedef Xped::Sym::U0<> Symmetry;
        Xped::Qbasis<Symmetry, 1> B, C;
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
