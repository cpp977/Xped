#ifdef _OPENMP
#    include "omp.h"
#endif

#include "Xped/Util/Macros.hpp"

#include "Xped/Util/Mpi.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "doctest/doctest.h"
#ifdef XPED_USE_MPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

#include "Xped/AD/ADTensor.hpp"
#include "Xped/IO/Json.hpp"
#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Physics/SpinBase.hpp"

#include "ad_tests.hpp"

#ifdef XPED_USE_MPI
constexpr int MPI_NUM_PROC = 2;
#endif

#ifdef XPED_USE_MPI
constexpr std::size_t SU2_BASIS_SIZE = 20;
constexpr std::size_t U1_BASIS_SIZE = 10;
constexpr std::size_t U0_BASIS_SIZE = 10;
#else
constexpr std::size_t SU2_BASIS_SIZE = 6;
constexpr std::size_t U1_BASIS_SIZE = 4;
constexpr std::size_t U0_BASIS_SIZE = 2;
#endif

TEST_SUITE_BEGIN("AD");

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for coeff().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for coeff().")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) { return t.coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return t.coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return t.coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for squareNnorm().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for squaredNorm().")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) { return (t * t.adjoint()).trace(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return (t * t.adjoint()).trace(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return (t * t.adjoint()).trace(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for norm().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for norm().")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) { return sqrt((t * t.adjoint()).trace()); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return t.norm(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return t.norm(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for maxNorm().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for maxNorm().")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) { return t.maxNorm(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return t.maxNorm(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return t.maxNorm(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for tSVD() - V.", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for tSVD() - V.")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(SU2_BASIS_SIZE, 1.e-10, trunc, false);
            return Vdag.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(U1_BASIS_SIZE, 1.e-10, trunc, false);
            return Vdag.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(U0_BASIS_SIZE, 1.e-10, trunc, false);
            return Vdag.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for tSVD() - U.", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for tSVD() - U.")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(1, 1.e-10, trunc, false);
            return U.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(1, 1.e-10, trunc, false);
            return U.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(2, 1.e-10, trunc, false);
            return U.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for tSVD() - Sigma.", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for tSVD() - Sigma.")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(SU2_BASIS_SIZE, 1.e-10, trunc, false);
            return Sigma.trace();
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(SU2_BASIS_SIZE, 1.e-10, trunc, false);
            return Sigma.trace();
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(SU2_BASIS_SIZE, 1.e-10, trunc, false);
            return Sigma.trace();
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for tSVD() - U*V.", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for tSVD() - U*V.")
#endif
{
#ifdef XPED_USE_MPI
    Xped::mpi::XpedWorld test_world(test_comm);
#else
    Xped::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(1, 1.e-10, trunc, false);
            return U.coeff(0, 0, 0) * Vdag.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(U1_BASIS_SIZE, 1.e-10, trunc, false);
            return U.coeff(0, 0, 0) * Vdag.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) {
            [[maybe_unused]] double trunc = 0.;
            auto [U, Sigma, Vdag] = t.tSVD(1, 1.e-10, trunc, false);
            return U.coeff(0, 0, 0) * Vdag.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

TEST_SUITE_END();
