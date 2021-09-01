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

// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN

#include "Util/Macros.hpp"

#include "Util/Mpi.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Interfaces/PlainInterface.hpp"

#include "Symmetry/SU2.hpp"
#include "Symmetry/U0.hpp"
#include "Symmetry/U1.hpp"

#include "MPS/Mps.hpp"
#include "MPS/MpsAlgebra.hpp"

#include "doctest/doctest.h"
#ifdef XPED_USE_MPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

#include "mps_tests.hpp"

#ifdef XPED_USE_MPI
constexpr std::size_t SU2_BASIS_SIZE = 100;
constexpr std::size_t U1_BASIS_SIZE = 100;
constexpr std::size_t U0_BASIS_SIZE = 100;
#else
constexpr std::size_t SU2_BASIS_SIZE = 5000;
constexpr std::size_t U1_BASIS_SIZE = 1000;
constexpr std::size_t U0_BASIS_SIZE = 1000;
#endif

#ifdef XPED_USE_MPI
constexpr int MPI_NUM_PROC = 2;
#endif

TEST_SUITE_BEGIN("Mps");

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing contract_L() and contract_R().", MPI_NUM_PROC)
#else
TEST_CASE("Testing contract_L() and contract_R().")
#endif
{
#ifdef XPED_USE_MPI
    util::mpi::XpedWorld test_world(test_comm);
#else
    util::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2") { perform_mps_contraction<Sym::SU2<Sym::SpinSU2>>(SU2_BASIS_SIZE, test_world); }
    SUBCASE("U1") { perform_mps_contraction<Sym::U1<Sym::SpinU1>>(U1_BASIS_SIZE, test_world); }
    SUBCASE("U0") { perform_mps_contraction<Sym::U0<>>(U0_BASIS_SIZE, test_world); }
}

TEST_SUITE_END();
