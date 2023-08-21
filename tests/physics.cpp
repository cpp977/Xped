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
#include "Xped/Physics/FermionBase.hpp"
#include "Xped/Physics/SpinBase.hpp"
#include "Xped/Symmetry/CombSym.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "doctest/doctest.h"
#ifdef XPED_USE_MPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

TEST_SUITE_BEGIN("Physics");

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing SpinBase.", 2)
#else
TEST_CASE("Testing SpinBase.")
#endif
{
    SUBCASE("U0")
    {
        using Symmetry = Xped::Sym::U0<>;
        auto dim = 6ul;
        Xped::SpinBase<Symmetry> B(dim, 2);
        Eigen::MatrixXd Js(dim, dim);
        Js.setZero();
        auto J1 = 1.;
        auto J2 = 0.8;
        auto Bz = 0.5;
        auto Bx = -0.2;
        Js.diagonal<1>().setConstant(0.5 * J1);
        Js.diagonal<-1>().setConstant(0.5 * J1);
        Js.diagonal<2>().setConstant(0.5 * J2);
        Js.diagonal<-2>().setConstant(0.5 * J2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), B.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - Bx * B.Sx(i) - Bz * B.Sz(i); }
                if(std::abs(Js(i, j)) < 1.e-12) { continue; }
                H = H + Js(i, j) * (B.Sz(i) * B.Sz(j) + 0.5 * (B.Sp(i) * B.Sm(j) + B.Sm(i) * B.Sp(j)));
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(0)(0, 0) == doctest::Approx(-2.43062856688879));
    }

    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1>;
        auto dim = 6ul;
        Xped::SpinBase<Symmetry> B(dim, 2);
        Eigen::MatrixXd Js(dim, dim);
        Js.setZero();
        auto J1 = 1.;
        auto J2 = 0.8;
        auto Bz = 0.5;
        Js.diagonal<1>().setConstant(0.5 * J1);
        Js.diagonal<-1>().setConstant(0.5 * J1);
        Js.diagonal<2>().setConstant(0.5 * J2);
        Js.diagonal<-2>().setConstant(0.5 * J2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), B.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - Bz * B.Sz(i); }
                if(std::abs(Js(i, j)) < 1.e-12) { continue; }
                H = H + Js(i, j) * (B.Sz(i) * B.Sz(j) + 0.5 * (B.Sp(i) * B.Sm(j) + B.Sm(i) * B.Sp(j)));
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(3)(0, 0) == doctest::Approx(-2.43062856688879));
    }

    SUBCASE("Z36")
    {
        using Symmetry = Xped::Sym::ZN<Xped::Sym::SpinU1, 36>;
        auto dim = 6ul;
        Xped::SpinBase<Symmetry> B(dim, 2);
        Eigen::MatrixXd Js(dim, dim);
        Js.setZero();
        auto J1 = 1.;
        auto J2 = 0.8;
        auto Bz = 0.5;
        Js.diagonal<1>().setConstant(0.5 * J1);
        Js.diagonal<-1>().setConstant(0.5 * J1);
        Js.diagonal<2>().setConstant(0.5 * J2);
        Js.diagonal<-2>().setConstant(0.5 * J2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), B.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - Bz * B.Sz(i); }
                if(std::abs(Js(i, j)) < 1.e-12) { continue; }
                H = H + Js(i, j) * (B.Sz(i) * B.Sz(j) + 0.5 * (B.Sp(i) * B.Sm(j) + B.Sm(i) * B.Sp(j)));
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(0)(0, 0) == doctest::Approx(-2.43062856688879));
    }

    SUBCASE("SU2")
    {
        using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        auto dim = 6ul;
        Xped::SpinBase<Symmetry> B(dim, 2);
        Eigen::MatrixXd Js(dim, dim);
        Js.setZero();
        auto J1 = 1.;
        auto J2 = 0.8;
        Js.diagonal<1>().setConstant(0.5 * J1);
        Js.diagonal<-1>().setConstant(0.5 * J1);
        Js.diagonal<2>().setConstant(0.5 * J2);
        Js.diagonal<-2>().setConstant(0.5 * J2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), B.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(std::abs(Js(i, j)) < 1.e-12) { continue; }
                H = H + Js(i, j) * std::sqrt(3.) * Xped::SiteOperator<double, Symmetry>::prod(B.Sdag(i), B.S(j), Symmetry::qvacuum());
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(0)(0, 0) == doctest::Approx(-2.43062856688881));
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing FermionBase.", 2)
#else
TEST_CASE("Testing FermionBase.")
#endif
{
    SUBCASE("Z2")
    {
        using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
        auto dim = 4ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto Bz = 0.5;
        auto Bx = -0.2;
        auto mu = 3.;
        auto U = 8.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - Bx * F.Sx(i) - Bz * F.Sz(i) - mu * F.n(i) + U * F.d(i); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * (F.cdag(Xped::SPIN_INDEX::UP, i) * F.c(Xped::SPIN_INDEX::UP, j) +
                                    F.cdag(Xped::SPIN_INDEX::DN, i) * F.c(Xped::SPIN_INDEX::DN, j));
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(0)(0, 0) == doctest::Approx(-13.4172994604531));
    }

    SUBCASE("Z36")
    {
        using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>;
        auto dim = 4ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto Bz = 0.5;
        auto Bx = -0.2;
        auto mu = 3.;
        auto U = 8.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - Bx * F.Sx(i) - Bz * F.Sz(i) - mu * F.n(i) + U * F.d(i); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * (F.cdag(Xped::SPIN_INDEX::UP, i) * F.c(Xped::SPIN_INDEX::UP, j) +
                                    F.cdag(Xped::SPIN_INDEX::DN, i) * F.c(Xped::SPIN_INDEX::DN, j));
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(4)(0, 0) == doctest::Approx(-13.4172994604531));
    }

    SUBCASE("U1")
    {
        using Symmetry = Xped::Sym::U1<Xped::Sym::FChargeU1>;
        auto dim = 4ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto Bz = 0.5;
        auto Bx = -0.2;
        auto mu = 3.;
        auto U = 8.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - Bx * F.Sx(i) - Bz * F.Sz(i) - mu * F.n(i) + U * F.d(i); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * (F.cdag(Xped::SPIN_INDEX::UP, i) * F.c(Xped::SPIN_INDEX::UP, j) +
                                    F.cdag(Xped::SPIN_INDEX::DN, i) * F.c(Xped::SPIN_INDEX::DN, j));
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(4)(0, 0) == doctest::Approx(-13.4172994604531));
    }

    SUBCASE("U1xU1")
    {
        using Symmetry = Xped::Sym::Combined<Xped::Sym::U1<Xped::Sym::FChargeU1>, Xped::Sym::U1<Xped::Sym::SpinU1>>;
        auto dim = 4ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto Bz = 0.5;
        auto mu = 3.;
        auto U = 8.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - Bz * F.Sz(i) - mu * F.n(i) + U * F.d(i); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * (F.cdag(Xped::SPIN_INDEX::UP, i) * F.c(Xped::SPIN_INDEX::UP, j) +
                                    F.cdag(Xped::SPIN_INDEX::DN, i) * F.c(Xped::SPIN_INDEX::DN, j));
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(13)(0, 0) == doctest::Approx(-13.378782979745));
    }

    SUBCASE("SU2xZ2")
    {
        using Symmetry = Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
        auto dim = 4ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto mu = 3.;
        auto U = 8.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - mu * F.n(i) + U * F.d(i); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * std::sqrt(2.) * Xped::SiteOperator<double, Symmetry>::prod(F.cdag(i), F.c(j), Symmetry::qvacuum());
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(0)(0, 0) == doctest::Approx(-13.1853931833403));
    }

    SUBCASE("SU2xZ36")
    {
        using Symmetry = Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;
        auto dim = 4ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto mu = 3.;
        auto U = 8.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - mu * F.n(i) + U * F.d(i); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * std::sqrt(2.) * Xped::SiteOperator<double, Symmetry>::prod(F.cdag(i), F.c(j), Symmetry::qvacuum());
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(2)(0, 0) == doctest::Approx(-13.1853931833403));
    }

    SUBCASE("SU2xU1")
    {
        using Symmetry = Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::U1<Xped::Sym::FChargeU1>>;
        auto dim = 4ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto mu = 3.;
        auto U = 8.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - mu * F.n(i) + U * F.d(i); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * std::sqrt(2.) * Xped::SiteOperator<double, Symmetry>::prod(F.cdag(i), F.c(j), Symmetry::qvacuum());
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(2)(0, 0) == doctest::Approx(-13.1853931833403));
    }

    SUBCASE("SU2xSU2xZ2")
    {
        using Symmetry =
            Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
        auto dim = 3ul;
        Xped::FermionBase<Symmetry> F(dim);
        Eigen::MatrixXd ts(dim, dim);
        ts.setZero();
        auto t1 = 1.;
        auto t2 = 0.8;
        auto mu = 0.;
        auto U = 0.;
        ts.diagonal<1>().setConstant(-t1);
        ts.diagonal<-1>().setConstant(-t1);
        ts.diagonal<2>().setConstant(-t2);
        ts.diagonal<-2>().setConstant(-t2);
        Xped::SiteOperator<double, Symmetry> H(Symmetry::qvacuum(), F.get_basis());
        H.setZero();
        for(std::size_t i = 0; i < dim; ++i) {
            for(std::size_t j = 0; j < dim; ++j) {
                if(i == j) { H = H - (mu + 3. * U / 2.) * F.n(i) + U / 2. * F.n(i) * (F.n(i) - F.Id()); }
                if(std::abs(ts(i, j)) < 1.e-12) { continue; }
                H = H + ts(i, j) * std::sqrt(2.) * std::sqrt(2.) * Xped::SiteOperator<double, Symmetry>::prod(F.cdag(i), F.c(j), Symmetry::qvacuum());
            }
        }
        auto [Es, Us] = H.data.trim<2>().teigh();
        CHECK(Es.block(0)(0, 0) == doctest::Approx(-7.47877505967545));
    }
}

TEST_SUITE_END();
