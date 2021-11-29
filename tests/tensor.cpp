#ifdef _OPENMP
#    include "omp.h"
#endif

#ifdef XPED_USE_CYCLOPS_TENSOR_LIB
#    define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
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

#include "Xped/Util/Macros.hpp"

#include "spdlog/spdlog.h"

#include "Eigen/Core"

#include "Xped/Util/Mpi.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/FusionTree.hpp"
#include "Xped/Core/Xped.hpp"

#include "doctest/doctest.h"
#ifdef XPED_USE_MPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

#include "tensor_tests.hpp"

#ifdef XPED_USE_MPI
constexpr std::size_t SU2_TENSOR_SIZE = 3;
constexpr std::size_t U1_TENSOR_SIZE = 3;
constexpr std::size_t U0_TENSOR_SIZE = 3;
#else
constexpr std::size_t SU2_TENSOR_SIZE = 7;
constexpr std::size_t U1_TENSOR_SIZE = 5;
constexpr std::size_t U0_TENSOR_SIZE = 5;
#endif

#ifdef XPED_USE_MPI
constexpr int MPI_NUM_PROC = 2;
#endif

TEST_SUITE_BEGIN("Tensor");

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the transformation to plain Tensor.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the transformation to plain Tensor.")
#endif
{
#ifdef XPED_USE_MPI
    util::mpi::XpedWorld test_world(test_comm);
#else
    util::mpi::XpedWorld test_world;
#endif

    SUBCASE("SU2")
    {
        SPDLOG_INFO("Performing the transformation to plain for SU(2)");
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        if(test_world.rank == 0) {
            B.setRandom(SU2_TENSOR_SIZE);
            C.setRandom(SU2_TENSOR_SIZE);
        }
        util::mpi::broadcast(B, test_world.rank, 0, test_world);
        util::mpi::broadcast(C, test_world.rank, 0, test_world);
        test_tensor_transformation_to_plain(B, C, test_world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        if(test_world.rank == 0) {
            B.setRandom(U1_TENSOR_SIZE);
            C.setRandom(U1_TENSOR_SIZE);
        }
        util::mpi::broadcast(B, test_world.rank, 0, test_world);
        util::mpi::broadcast(C, test_world.rank, 0, test_world);
        test_tensor_transformation_to_plain(B, C, test_world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        if(test_world.rank == 0) {
            B.setRandom(U0_TENSOR_SIZE);
            C.setRandom(U0_TENSOR_SIZE);
        }
        util::mpi::broadcast(B, test_world.rank, 0, test_world);
        util::mpi::broadcast(C, test_world.rank, 0, test_world);
        test_tensor_transformation_to_plain(B, C, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the permutation within the domain.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the permutation within the domain.")
#endif
{
#ifdef XPED_USE_MPI
    util::mpi::XpedWorld test_world(test_comm);
#else
    util::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        test_tensor_permute_within_domain<Symmetry>(SU2_TENSOR_SIZE, test_world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        test_tensor_permute_within_domain<Symmetry>(U1_TENSOR_SIZE, test_world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        test_tensor_permute_within_domain<Symmetry>(U0_TENSOR_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the permutation within the codomain.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the permutation within the codomain.")
#endif
{
#ifdef XPED_USE_MPI
    util::mpi::XpedWorld test_world(test_comm);
#else
    util::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        test_tensor_permute_within_codomain<Symmetry>(SU2_TENSOR_SIZE, test_world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        test_tensor_permute_within_codomain<Symmetry>(U1_TENSOR_SIZE, test_world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        test_tensor_permute_within_codomain<Symmetry>(U0_TENSOR_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing the general permutation of legs.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the general permutation of legs.")
#endif
{
#ifdef XPED_USE_MPI
    util::mpi::XpedWorld test_world(test_comm);
#else
    util::mpi::XpedWorld test_world;
#endif
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        test_tensor_permute<Symmetry, -2>(SU2_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, -1>(SU2_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +0>(SU2_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +1>(SU2_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +2>(SU2_TENSOR_SIZE, test_world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        test_tensor_permute<Symmetry, -2>(U1_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, -1>(U1_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +0>(U1_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +1>(U1_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +2>(U1_TENSOR_SIZE, test_world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        test_tensor_permute<Symmetry, -2>(U0_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, -1>(U0_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +0>(U0_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +1>(U0_TENSOR_SIZE, test_world);
        test_tensor_permute<Symmetry, +2>(U0_TENSOR_SIZE, test_world);
    }
}

#ifdef XPED_USE_MPI
MPI_TEST_CASE("Testing operations with SU(2)-spin matrices.", MPI_NUM_PROC)
#else
TEST_CASE("Testing operations with SU(2)-spin matrices.")
#endif
{
#ifdef XPED_USE_MPI
    util::mpi::XpedWorld test_world(test_comm);
#else
    util::mpi::XpedWorld test_world;
#endif
    typedef Sym::SU2<Sym::SpinSU2> Symmetry;

    // loop over different S
    for(int twoS1 = 1; twoS1 < 11; twoS1++) {
        Qbasis<Symmetry, 1> B1, C, one, three, five;
        B1.push_back({twoS1 + 1}, 1);
        C.push_back({3}, 1);
        one.push_back({1}, 1);
        three.push_back({3}, 1);
        five.push_back({5}, 1);

        double S1 = 0.5 * twoS1;

        // construct the spin s matrices
        std::array<Eigen::MatrixXd, 3> pauli_vec1;
        //-1./sqrt(2.)sp
        pauli_vec1[0].resize(twoS1 + 1, twoS1 + 1);
        pauli_vec1[0].setZero();
        for(int m_ = 1; m_ < twoS1 + 1; m_++) {
            double m = S1 - m_;
            pauli_vec1[0](m_ - 1, m_) = std::sqrt(S1 * (S1 + 1) - m * (m + 1));
        }
        pauli_vec1[0] *= (-std::sqrt(0.5));

        // sz
        pauli_vec1[1].resize(twoS1 + 1, twoS1 + 1);
        pauli_vec1[1].setZero();
        for(int m_ = 0; m_ < twoS1 + 1; m_++) {
            double m = S1 - m_;
            pauli_vec1[1](m_, m_) = m;
        }

        // 1./sqrt(2.)*sm
        pauli_vec1[2].resize(twoS1 + 1, twoS1 + 1);
        pauli_vec1[2].setZero();
        for(int m_ = 0; m_ < twoS1; m_++) {
            double m = S1 - m_;
            pauli_vec1[2](m_ + 1, m_) = std::sqrt(S1 * (S1 + 1) - m * (m - 1));
        }
        pauli_vec1[2] *= std::sqrt(0.5);

        // set s to the reduced Spin operator
        Xped<double, 2, 1, Symmetry> s1({{B1, C}}, {{B1}}, test_world);
        s1.setConstant(std::sqrt(S1 * (S1 + 1.)));

        // transform to plain tensor and check against pauli_vec1
        auto tplain = s1.adjoint().eval().plainTensor();
        for(Eigen::Index k = 0;
            k < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(tplain)[2];
            k++) {
            Eigen::Matrix<double, -1, -1> pauli(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index j = 0;
                j < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(tplain)[1];
                j++)
                for(Eigen::Index i = 0;
                    i < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(tplain)[0];
                    i++) {
                    pauli(i, j) = PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::getVal<double, 3>(
                        tplain,
                        std::array<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(k)});
                }
            CHECK((pauli - pauli_vec1[k]).norm() == doctest::Approx(0.));
        }

        // build the product to QN K=0
        Xped<double, 2, 1, Symmetry> couple1({{C, one}}, {{C}}, test_world);
        couple1.setConstant(1.);
        auto prod1 = (s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple1.permute<+1, 0, 2, 1>()).permute<0, 0, 3, 1, 2>() * s1;
        // transform to plain tensor and check against S^2
        auto check1 = prod1.plainTensor();
        for(Eigen::Index j = 0;
            j < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check1)[1];
            j++) {
            Eigen::Matrix<double, -1, -1> diag(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index k = 0;
                k < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check1)[2];
                k++)
                for(Eigen::Index i = 0;
                    i < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check1)[0];
                    i++) {
                    diag(i, k) = PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::getVal<double, 3>(
                        check1,
                        std::array<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(k)});
                }
            CHECK((diag - S1 * (S1 + 1.) * Eigen::Matrix<double, -1, -1>::Identity(twoS1 + 1, twoS1 + 1)).norm() == doctest::Approx(0.));
        }

        // build the product to QN K=1
        Xped<double, 2, 1, Symmetry> couple3({{C, three}}, {{C}}, test_world);
        couple3.setConstant(1.);
        auto prod3 = (s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple3.permute<+1, 0, 2, 1>()).permute<0, 0, 3, 1, 2>() * s1;
        // transform to plain tensor and check against SxS
        auto check3 = prod3.adjoint().eval().plainTensor();
        for(Eigen::Index k = 0;
            k < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check3)[2];
            k++) {
            Eigen::Matrix<double, -1, -1> mat(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index j = 0;
                j < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check3)[1];
                j++)
                for(Eigen::Index i = 0;
                    i < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check3)[0];
                    i++) {
                    mat(i, j) = PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::getVal<double, 3>(
                        check3,
                        std::array<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(k)});
                }
            CHECK((mat - std::sqrt(0.5) * pauli_vec1[k]).norm() == doctest::Approx(0.));
        }
        // build the product to QN K=2
        Xped<double, 2, 1, Symmetry> couple5({{C, five}}, {{C}}, test_world);
        couple5.setConstant(1.);
        auto prod5 = (s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple5.permute<+1, 0, 2, 1>()).permute<0, 0, 3, 1, 2>() * s1;
        // transform to plain tensor and check against SxS
        auto check5 = prod5.adjoint().eval().plainTensor();
        for(Eigen::Index k = 0;
            k < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check5)[2];
            k++) {
            Eigen::Matrix<double, -1, -1> mat(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index j = 0;
                j < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check5)[1];
                j++)
                for(Eigen::Index i = 0;
                    i < PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::dimensions<double, 3>(check5)[0];
                    i++) {
                    mat(i, j) = PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::getVal<double, 3>(
                        check5,
                        std::array<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::Indextype>(k)});
                }
            Eigen::MatrixXd pauli(twoS1 + 1, twoS1 + 1);
            pauli.setZero();
            if(k == 0) { pauli = -std::sqrt(0.6) * (pauli_vec1[0] * pauli_vec1[0]); }
            if(k == 1) { pauli = -std::sqrt(0.3) * (pauli_vec1[0] * pauli_vec1[1] + pauli_vec1[1] * pauli_vec1[0]); }
            if(k == 2) {
                pauli = -std::sqrt(0.1) * (pauli_vec1[0] * pauli_vec1[2] + pauli_vec1[2] * pauli_vec1[0] + 2 * pauli_vec1[1] * pauli_vec1[1]);
            }
            if(k == 3) { pauli = -std::sqrt(0.3) * (pauli_vec1[2] * pauli_vec1[1] + pauli_vec1[1] * pauli_vec1[2]); }
            if(k == 4) { pauli = -std::sqrt(0.6) * (pauli_vec1[2] * pauli_vec1[2]); }
            CHECK((mat - pauli).norm() == doctest::Approx(0.));
        }

        for(int twoS2 = 1; twoS2 < 11; twoS2++) {
            Qbasis<Symmetry, 1> B2;
            B2.push_back({twoS2 + 1}, 1);
            double S2 = 0.5 * twoS2;

            Xped<double, 2, 1, Symmetry> s2({{B2, C}}, {{B2}}, test_world);
            s2.setConstant(std::sqrt(S2 * (S2 + 1.)));

            // build the outer product to QN K=0 between s1 and s2
            auto outerprod1 = (((s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple1.permute<+1, 0, 2, 1>()).permute<-1, 0, 1, 3, 2>()) *
                               s2.permute<+1, 1, 2, 0>())
                                  .permute<0, 0, 4, 2, 1, 3>();
            for(const auto& q : outerprod1.sector()) {
                CHECK(PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::rows(outerprod1(q)) *
                          PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::cols(outerprod1(q)) ==
                      1);
                double Stot = 0.5 * (q[0] - 1.);
                CHECK(PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>::getVal<double>(outerprod1(q), 0, 0) ==
                      doctest::Approx(0.5 * (Stot * (Stot + 1) - S1 * (S1 + 1.) - S2 * (S2 + 1.))));
            }
        }
    }
}
TEST_SUITE_END();
