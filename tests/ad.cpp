#ifdef _OPENMP
#    include "omp.h"
#endif

#include "Xped/Util/Macros.hpp"

#include "Xped/Util/Bool.hpp"
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
        auto f = [](const auto& t) { return t.coeff(t.sector().size() - 1, 0, 0); };
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
MPI_TEST_CASE("Testing backwards AD for squaredNorm().", MPI_NUM_PROC)
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
MPI_TEST_CASE("Testing backwards AD for trace().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for trace().")
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
        auto f = [](const auto& t) { return t.trace(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return t.trace(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return t.trace(); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for twist().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for twist().")
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
        auto f = [](const auto& t) { return t.twist(0).coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return t.twist(0).coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return t.twist(0).coeff(0, 0, 0); };
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
        auto f = [](const auto& t) { return t.norm(); };
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
MPI_TEST_CASE("Testing backwards AD for operator*(true,false).", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for operator*().")
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
            auto c = t.detach();
            static thread_local std::mt19937 engine(std::random_device{}());
            engine.seed(100);
            c.setRandom(engine);
            return (t * c).coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) {
            auto c = t.detach();
            static thread_local std::mt19937 engine(std::random_device{}());
            engine.seed(100);
            c.setRandom(engine);
            return (t * c).coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) {
            auto c = t.detach();
            static thread_local std::mt19937 engine(std::random_device{}());
            engine.seed(100);
            c.setRandom(engine);
            return (t * c).coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for operator*(true,ture).", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for operator*().")
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
        auto f = [](const auto& t) { return (t * t).coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return (t * t).coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return (t * t).coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for permute().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for permute().")
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
        using Scalar = double;
        auto f = [](const auto& t) {
            constexpr auto ENABLE_AD = std::decay_t<decltype(t)>::IS_AD;
            auto tdag = t.adjoint().eval().template permute<-4, 1, 2, 3, 4, 0>(Xped::Bool<ENABLE_AD>{});
            Xped::Qbasis<Symmetry> env;
            Xped::Qbasis<Symmetry> peps = t.uncoupledDomain()[0];
            env.push_back({1}, 2);
            env.push_back({2}, 2);
            env.push_back({3}, 2);
            Xped::Tensor<double, 2, 0, Symmetry, false> C1init({{env, env}}, {{}});
            static thread_local std::mt19937 engine(std::random_device{}());
            engine.seed(0);
            C1init.setRandom(engine);
            Xped::Tensor<double, 2, 0, Symmetry, ENABLE_AD> C1 = C1init;
            Xped::Tensor<double, 4, 0, Symmetry, false> T1init({{env, env, peps, peps}}, {{}});
            engine.seed(1);
            T1init.setRandom(engine);
            Xped::Tensor<double, 4, 0, Symmetry, ENABLE_AD> T1 = T1init;

            auto C1T1 = C1.template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1>(T1.adjoint().eval());
            auto T4C1T1 = T1.adjoint().eval().template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3>(C1T1.twist(0));
            auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4>(t);
            auto C4T3 = C1.template contract<std::array{1, -1}, std::array{-2, 1, -3, -4}, 1>(T1.adjoint().eval());
            auto C4T3Ad = C4T3.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, -6, 1, -7}, 3>(tdag.twist(3));
            auto left_half = T4C1T1A.template contract<std::array{1, 2, -1, 3, -2, 4, -3}, std::array{1, -4, 4, 2, 3, -5, -6}, 3>(C4T3Ad);
            // return left_half.coeff(0, 0, 0);
            auto T1C2 = T1.template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3>(C1.adjoint().eval());
            auto T1C2T2 = T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{1, -4, -5, -6}, 3>(T1);
            auto AT1C2T2 = t.adjoint().eval().twist(1).template contract<std::array{-3, -1, 1, 2, -2}, std::array{-4, 1, -5, -6, 2, -7}, 3>(T1C2T2);
            auto T3C3 = T1.template contract<std::array{1, -1, -2, -3}, std::array{-4, 1}, 3>(C1.adjoint().eval());
            auto T3C3Ad = T3C3.template contract<std::array{-1, -2, 1, -3}, std::array{-4, -5, -6, 1, -7}, 3>(t);
            auto right_half = AT1C2T2.template contract<std::array{-1, 1, -2, -3, 2, 3, 4}, std::array{-4, 1, 3, -5, 2, 4, -6}, 3>(T3C3Ad);
            auto rho_h = left_half.template contract<std::array{1, 2, -3, 3, 4, -1}, std::array{2, -4, 1, 3, 4, -2}, 2>(right_half);
            auto Id2 = Xped::Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rho_h.uncoupledCodomain(), rho_h.uncoupledDomain(), rho_h.world());
            auto norm = rho_h.template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(Id2.twist(0).twist(1)).trace();
            rho_h = rho_h * (1. / norm);
            // return tmp.coeff(tmp.sector().size() - 1, 0, 0);
            return rho_h.trace();
            // return tmp.coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd_small<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) {
            constexpr auto ENABLE_AD = std::decay_t<decltype(t)>::IS_AD;
            return t.template permute<-1, 3, 1, 2, 0>(Xped::Bool<ENABLE_AD>{}).coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) {
            constexpr auto ENABLE_AD = std::decay_t<decltype(t)>::IS_AD;
            return t.template permute<-1, 3, 1, 2, 0>(Xped::Bool<ENABLE_AD>{}).coeff(0, 0, 0);
        };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U0_BASIS_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing backwards AD for contract().", MPI_NUM_PROC)
#else
TEST_CASE("Testing backwards AD for contract().")
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
        auto f = [](const auto& t) { return t.template contract<std::array{-1, 1, 2, -2}, std::array{2, 1, -3, -4}, 2>(t).coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, SU2_BASIS_SIZE, test_world);
    }
    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto f = [](const auto& t) { return t.template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3, -4}, 2>(t).coeff(0, 0, 0); };
        Xped::internal::check_ad_vs_fd<double, Symmetry>(f, U1_BASIS_SIZE, test_world);
    }
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto f = [](const auto& t) { return t.template contract<std::array{-1, 1, 2, -2}, std::array{2, 1, -3, -4}, 2>(t).coeff(0, 0, 0); };
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
