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

#include "Symmetry/SU2.hpp"
#include "Symmetry/U0.hpp"
#include "Symmetry/U1.hpp"

#include "MPS/Mps.hpp"
#include "MPS/MpsAlgebra.hpp"

#include "doctest/doctest.h"

#include "mps_tests.hpp"

constexpr std::size_t SU2_BASIS_SIZE = 2000;
constexpr std::size_t U1_BASIS_SIZE = 1000;
constexpr std::size_t U0_BASIS_SIZE = 500;

TEST_SUITE_BEGIN("Mps");

TEST_CASE("Testing contract_L() and contract_R().")
{
    SUBCASE("SU2") { perform_mps_contraction<Sym::SU2<Sym::SpinSU2>>(SU2_BASIS_SIZE); }
    SUBCASE("U1") { perform_mps_contraction<Sym::U1<Sym::SpinU1>>(U1_BASIS_SIZE); }
    SUBCASE("U0") { perform_mps_contraction<Sym::U0>(U0_BASIS_SIZE); }
}

TEST_SUITE_END();
