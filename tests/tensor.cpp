#ifdef INTEL_MKL_VERSION
#    pragma message("Xped is using the intel math kernel library (MKL)")
#endif

#ifdef _OPENMP
#    pragma message("Xped is using OpenMP parallelization")
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

#include "spdlog/spdlog.h"

#include "ArgParser.h"

#include "Util/Macros.hpp"

#include "Util/Mpi.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    pragma message("Xped is using LRU cache for the output of FusionTree manipulations.")
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Interfaces/PlainInterface.hpp"

#include "Core/Qbasis.hpp"
#include "Symmetry/SU2.hpp"
#include "Symmetry/U0.hpp"
#include "Symmetry/U1.hpp"
#include "Symmetry/kind_dummies.hpp"

#include "Core/AdjointOp.hpp"
#include "Core/FusionTree.hpp"
#include "Core/Xped.hpp"

#include "doctest/doctest.h"
#ifdef XPED_USE_OPENMPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

#include "tensor_tests.hpp"

constexpr std::size_t SU2_TENSOR_SIZE = 5;
constexpr std::size_t U1_TENSOR_SIZE = 5;
constexpr std::size_t U0_TENSOR_SIZE = 5;

#ifdef XPED_USE_OPENMPI
constexpr int MPI_NUM_PROC = 4;
#endif

TEST_SUITE_BEGIN("Tensor");

#ifdef XPED_USE_OPENMPI
MPI_TEST_CASE("Testing the transformation to plain Tensor.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the transformation to plain Tensor.")
#endif
{
#ifdef XPED_USE_OPENMPI
    util::mpi::XpedWorld world(test_comm);
#else
    util::mpi::XpedWorld world;
#endif

    SUBCASE("SU2")
    {
        spdlog::get("info")->info("Performing the transformation to plain for SU(2)");
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        if(world.rank == 0) {
            B.setRandom(SU2_TENSOR_SIZE);
            C.setRandom(SU2_TENSOR_SIZE);
        }
        util::mpi::broadcast(B, world.rank, 0, world);
        util::mpi::broadcast(C, world.rank, 0, world);
        test_tensor_transformation_to_plain(B, C, world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        if(world.rank == 0) {
            B.setRandom(U1_TENSOR_SIZE);
            C.setRandom(U1_TENSOR_SIZE);
        }
        util::mpi::broadcast(B, world.rank, 0, world);
        util::mpi::broadcast(C, world.rank, 0, world);
        test_tensor_transformation_to_plain(B, C, world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        Qbasis<Symmetry, 1> B, C;
        if(world.rank == 0) {
            B.setRandom(U0_TENSOR_SIZE);
            C.setRandom(U0_TENSOR_SIZE);
        }
        util::mpi::broadcast(B, world.rank, 0, world);
        util::mpi::broadcast(C, world.rank, 0, world);
        test_tensor_transformation_to_plain(B, C, world);
    }
}

#ifdef XPED_USE_OPENMPI
MPI_TEST_CASE("Testing the permutation within the domain.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the permutation within the domain.")
#endif
{
#ifdef XPED_USE_OPENMPI
    util::mpi::XpedWorld world(test_comm);
#else
    util::mpi::XpedWorld world;
#endif
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        test_tensor_permute_within_domain<Symmetry>(SU2_TENSOR_SIZE, world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        test_tensor_permute_within_domain<Symmetry>(U1_TENSOR_SIZE, world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        test_tensor_permute_within_domain<Symmetry>(U0_TENSOR_SIZE, world);
    }
}

#ifdef XPED_USE_OPENMPI
MPI_TEST_CASE("Testing the permutation within the codomain.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the permutation within the codomain.")
#endif
{
#ifdef XPED_USE_OPENMPI
    util::mpi::XpedWorld world(test_comm);
#else
    util::mpi::XpedWorld world;
#endif
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        test_tensor_permute_within_codomain<Symmetry>(SU2_TENSOR_SIZE, world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        test_tensor_permute_within_codomain<Symmetry>(U1_TENSOR_SIZE, world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        test_tensor_permute_within_codomain<Symmetry>(U0_TENSOR_SIZE, world);
    }
}

#ifdef XPED_USE_OPENMPI
MPI_TEST_CASE("Testing the general permutation of legs.", MPI_NUM_PROC)
#else
TEST_CASE("Testing the general permutation of legs.")
#endif
{
#ifdef XPED_USE_OPENMPI
    util::mpi::XpedWorld world(test_comm);
#else
    util::mpi::XpedWorld world;
#endif
    SUBCASE("SU2")
    {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        test_tensor_permute<Symmetry, -2>(SU2_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, -1>(SU2_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +0>(SU2_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +1>(SU2_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +2>(SU2_TENSOR_SIZE, world);
    }

    SUBCASE("U1")
    {
        typedef Sym::U1<Sym::SpinU1> Symmetry;
        test_tensor_permute<Symmetry, -2>(U1_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, -1>(U1_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +0>(U1_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +1>(U1_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +2>(U1_TENSOR_SIZE, world);
    }

    SUBCASE("U0")
    {
        typedef Sym::U0<> Symmetry;
        test_tensor_permute<Symmetry, -2>(U0_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, -1>(U0_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +0>(U0_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +1>(U0_TENSOR_SIZE, world);
        test_tensor_permute<Symmetry, +2>(U0_TENSOR_SIZE, world);
    }
}

TEST_CASE("Testing operations with SU(2)-spin matrices.")
{
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
        Xped<double, 2, 1, Symmetry> s1({{B1, C}}, {{B1}});
        s1.setConstant(std::sqrt(S1 * (S1 + 1.)));

        // transform to plain tensor and check against pauli_vec1
        auto tplain = s1.adjoint().eval().plainTensor();
        for(Eigen::Index k = 0; k < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(tplain)[2]; k++) {
            Eigen::Matrix<double, -1, -1> pauli(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index j = 0; j < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(tplain)[1]; j++)
                for(Eigen::Index i = 0; i < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(tplain)[0]; i++) {
                    pauli(i, j) = PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::getVal<double, 3>(
                        tplain,
                        std::array<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(k)});
                }
            CHECK((pauli - pauli_vec1[k]).norm() == doctest::Approx(0.));
        }

        // build the product to QN K=0
        Xped<double, 2, 1, Symmetry> couple1({{C, one}}, {{C}});
        couple1.setConstant(1.);
        auto prod1 = (s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple1.permute<+1, 0, 2, 1>()).permute<0, 0, 3, 1, 2>() * s1;
        // transform to plain tensor and check against S^2
        auto check1 = prod1.plainTensor();
        for(Eigen::Index j = 0; j < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check1)[1]; j++) {
            Eigen::Matrix<double, -1, -1> diag(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index k = 0; k < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check1)[2]; k++)
                for(Eigen::Index i = 0; i < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check1)[0]; i++) {
                    diag(i, k) = PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::getVal<double, 3>(
                        check1,
                        std::array<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(k)});
                }
            CHECK((diag - S1 * (S1 + 1.) * Eigen::Matrix<double, -1, -1>::Identity(twoS1 + 1, twoS1 + 1)).norm() == doctest::Approx(0.));
        }

        // build the product to QN K=1
        Xped<double, 2, 1, Symmetry> couple3({{C, three}}, {{C}});
        couple3.setConstant(1.);
        auto prod3 = (s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple3.permute<+1, 0, 2, 1>()).permute<0, 0, 3, 1, 2>() * s1;
        // transform to plain tensor and check against SxS
        auto check3 = prod3.adjoint().eval().plainTensor();
        for(Eigen::Index k = 0; k < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check3)[2]; k++) {
            Eigen::Matrix<double, -1, -1> mat(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index j = 0; j < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check3)[1]; j++)
                for(Eigen::Index i = 0; i < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check3)[0]; i++) {
                    mat(i, j) = PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::getVal<double, 3>(
                        check3,
                        std::array<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(k)});
                }
            CHECK((mat - std::sqrt(0.5) * pauli_vec1[k]).norm() == doctest::Approx(0.));
        }
        // build the product to QN K=2
        Xped<double, 2, 1, Symmetry> couple5({{C, five}}, {{C}});
        couple5.setConstant(1.);
        auto prod5 = (s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple5.permute<+1, 0, 2, 1>()).permute<0, 0, 3, 1, 2>() * s1;
        // transform to plain tensor and check against SxS
        auto check5 = prod5.adjoint().eval().plainTensor();
        for(Eigen::Index k = 0; k < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check5)[2]; k++) {
            Eigen::Matrix<double, -1, -1> mat(twoS1 + 1, twoS1 + 1);
            for(Eigen::Index j = 0; j < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check5)[1]; j++)
                for(Eigen::Index i = 0; i < PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::dimensions<double, 3>(check5)[0]; i++) {
                    mat(i, j) = PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::getVal<double, 3>(
                        check5,
                        std::array<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype, 3>{
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(i),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(j),
                            static_cast<PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::Indextype>(k)});
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

            Xped<double, 2, 1, Symmetry> s2({{B2, C}}, {{B2}});
            s2.setConstant(std::sqrt(S2 * (S2 + 1.)));

            // build the outer product to QN K=0 between s1 and s2
            auto outerprod1 = (((s1.adjoint().eval().permute<-1, 0, 1, 2>() * couple1.permute<+1, 0, 2, 1>()).permute<-1, 0, 1, 3, 2>()) *
                               s2.permute<+1, 1, 2, 0>())
                                  .permute<0, 0, 4, 2, 1, 3>();
            for(const auto& q : outerprod1.sector()) {
                CHECK(PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::rows(outerprod1(q)) *
                          PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::cols(outerprod1(q)) ==
                      1);
                double Stot = 0.5 * (q[0] - 1.);
                CHECK(PlainInterface<M_MATRIXLIB, M_TENSORLIB, M_VECTORLIB>::getVal<double>(outerprod1(q), 0, 0) ==
                      doctest::Approx(0.5 * (Stot * (Stot + 1) - S1 * (S1 + 1.) - S2 * (S2 + 1.))));
            }
        }
    }
}

TEST_SUITE_END();
