#include <iostream>
#include <cstddef>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

#include <TextTable.h>

#include "macros.h"

using std::size_t;
using std::cout;
using std::endl;
using std::string;

#include "ArgParser.h"

#ifdef CACHE_PERMUTE_OUTPUT
#include "lru/lru.hpp"

template<std::size_t Rank, typename Symmetry> struct FusionTree;
template<std::size_t N> struct Permutation;

template<int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
struct CacheManager
{
        typedef FusionTree<CoRank, Symmetry> CoTree;
        typedef FusionTree<CoRank+shift, Symmetry> NewCoTree;
        typedef FusionTree<Rank, Symmetry> Tree;
        typedef FusionTree<Rank-shift, Symmetry> NewTree;
        typedef typename Symmetry::Scalar Scalar;
        
        typedef LRU::Cache<std::tuple<Tree, CoTree, Permutation<Rank+CoRank> >, std::unordered_map<std::pair<NewTree, NewCoTree >, Scalar> >  CacheType;
        CacheManager(std::size_t cache_size) {
                cache = CacheType(cache_size);
                cache.monitor();
        }
        CacheType cache;
};

template<int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
CacheManager<shift, Rank, CoRank, Symmetry> tree_cache(100);
#endif

#include "../src/Qbasis.hpp"
#include "../src/symmetry/kind_dummies.hpp"
#include "../src/symmetry/SU2.hpp"
#include "../src/symmetry/U1.hpp"
#include "../src/symmetry/U0.hpp"

#include "../src/FusionTree.hpp"
#include "../src/Tensor.hpp"

#include "doctest/doctest.h"

#include "tensor_tests.hpp"

TEST_SUITE_BEGIN("Tensor");

TEST_CASE("Testing the transformation to plain Tensor.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                test_tensor_transformation_to_plain(B,C);
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                test_tensor_transformation_to_plain(B,C);
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                test_tensor_transformation_to_plain(B,C);
        }
}

TEST_CASE("Testing the permutation within the domain.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(6); C.setRandom(6); D.setRandom(6), E.setRandom(6);
                test_tensor_permute_within_domain(B,C,D,E);
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(4); C.setRandom(4); D.setRandom(4), E.setRandom(4);
                test_tensor_permute_within_domain(B,C,D,E);
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(3); C.setRandom(3); D.setRandom(3), E.setRandom(3);
                test_tensor_permute_within_domain(B,C,D,E);
        }
}

TEST_CASE("Testing the permutation within the codomain.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(6); D.setRandom(6), E.setRandom(6);
                test_tensor_permute_within_codomain(B,C,D,E);
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(4); C.setRandom(4); D.setRandom(4), E.setRandom(4);
                test_tensor_permute_within_codomain(B,C,D,E);
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(3); C.setRandom(3); D.setRandom(3), E.setRandom(3);
                test_tensor_permute_within_codomain(B,C,D,E);
        }
}

TEST_CASE("Testing the general permutation of legs.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(5); C.setRandom(5);
                test_tensor_permute<-2>(B,C);
                test_tensor_permute<-1>(B,C);
                test_tensor_permute<+0>(B,C);
                test_tensor_permute<+1>(B,C);
                test_tensor_permute<+2>(B,C);                
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(4); C.setRandom(4);
                test_tensor_permute<-2>(B,C);
                test_tensor_permute<-1>(B,C);
                test_tensor_permute<+0>(B,C);
                test_tensor_permute<+1>(B,C);
                test_tensor_permute<+2>(B,C);
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(3); C.setRandom(3);
                test_tensor_permute<-2>(B,C);
                test_tensor_permute<-1>(B,C);
                test_tensor_permute<+0>(B,C);
                test_tensor_permute<+1>(B,C);
                test_tensor_permute<+2>(B,C);
        }
}

TEST_CASE("Testing operations with SU(2)-spin matrices.") {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;

        //loop over different S
        for (int twoS1 = 1; twoS1<11; twoS1++) {
                Qbasis<Symmetry,1> B1, C, one, three, five;
                B1.push_back({twoS1+1},1);
                C.push_back({3},1);
                one.push_back({1},1);
                three.push_back({3},1);
                five.push_back({5},1);

                double S1 = 0.5*twoS1;
                
                //construct the spin s matrices
                std::array<Eigen::MatrixXd,3> pauli_vec1;
                //-1./sqrt(2.)sp
                pauli_vec1[0].resize(twoS1+1,twoS1+1); pauli_vec1[0].setZero();
                for (int m_=1; m_<twoS1+1; m_++) {
                        double m = S1 - m_;
                        pauli_vec1[0](m_-1,m_) = std::sqrt(S1*(S1+1)-m*(m+1));
                }
                pauli_vec1[0] *= (-std::sqrt(0.5));
                
                //sz
                pauli_vec1[1].resize(twoS1+1,twoS1+1); pauli_vec1[1].setZero();
                for (int m_=0; m_<twoS1+1; m_++) {
                        double m = S1 - m_;
                        pauli_vec1[1](m_,m_) = m;
                }
                
                //1./sqrt(2.)*sm
                pauli_vec1[2].resize(twoS1+1,twoS1+1); pauli_vec1[2].setZero();
                for (int m_=0; m_<twoS1; m_++) {
                        double m = S1 - m_;
                        pauli_vec1[2](m_+1,m_) = std::sqrt(S1*(S1+1)-m*(m-1));
                }
                pauli_vec1[2] *= std::sqrt(0.5);
                
                
                //set s to the reduced Spin operator
                Tensor<2,1,Symmetry> s1({{B1,C}},{{B1}}); s1.setConstant(std::sqrt(S1*(S1+1.)));
                
                //transform to plain tensor and check against pauli_vec1
                auto tplain = s1.adjoint().plainTensor();
                for (Eigen::Index k=0; k<tplain.dimensions()[2]; k++) {
                        Eigen::Matrix<double,-1,-1> pauli(twoS1+1,twoS1+1);
                        for (Eigen::Index j=0; j<tplain.dimensions()[1]; j++)
                                for (Eigen::Index i=0; i<tplain.dimensions()[0]; i++) {
                                        pauli(i,j) = tplain(i,j,k);
                                }
                        CHECK((pauli-pauli_vec1[k]).norm() == doctest::Approx(0.));
                }

                //build the product to QN K=0
                Tensor<2,1,Symmetry> couple1({{C,one}},{{C}}); couple1.setConstant(1.);
                auto prod1 = (s1.adjoint().permute<-1>({{0,1,2}}) * couple1.permute<+1>({{0,2,1}})).permute<0>({{0,3,1,2}}) * s1;
                //transform to plain tensor and check against S^2
                auto check1 = prod1.plainTensor();
                for (Eigen::Index j=0; j<check1.dimensions()[1]; j++) {
                        Eigen::Matrix<double,-1,-1> diag(twoS1+1,twoS1+1);
                        for (Eigen::Index k=0; k<check1.dimensions()[2]; k++)
                                for (Eigen::Index i=0; i<check1.dimensions()[0]; i++) {
                                        diag(i,k) = check1(i,j,k);
                                }
                        CHECK((diag-S1*(S1+1.)*Eigen::Matrix<double,-1,-1>::Identity(twoS1+1,twoS1+1)).norm() == doctest::Approx(0.));
                }

                //build the product to QN K=1
                Tensor<2,1,Symmetry> couple3({{C,three}},{{C}}); couple3.setConstant(1.);
                auto prod3 = (s1.adjoint().permute<-1>({{0,1,2}}) * couple3.permute<+1>({{0,2,1}})).permute<0>({{0,3,1,2}}) * s1;
                //transform to plain tensor and check against SxS
                auto check3 = prod3.adjoint().plainTensor();
                for (Eigen::Index k=0; k<check3.dimensions()[2]; k++) {
                        Eigen::Matrix<double,-1,-1> mat(twoS1+1,twoS1+1);
                        for (Eigen::Index j=0; j<check3.dimensions()[1]; j++)
                                for (Eigen::Index i=0; i<check3.dimensions()[0]; i++) {
                                        mat(i,j) = check3(i,j,k);
                                }
                        CHECK((mat-std::sqrt(0.5)*pauli_vec1[k]).norm() == doctest::Approx(0.));
                }
                //build the product to QN K=2
                Tensor<2,1,Symmetry> couple5({{C,five}},{{C}}); couple5.setConstant(1.);
                auto prod5 = (s1.adjoint().permute<-1>({{0,1,2}}) * couple5.permute<+1>({{0,2,1}})).permute<0>({{0,3,1,2}}) * s1;
                //transform to plain tensor and check against SxS
                auto check5 = prod5.adjoint().plainTensor();
                for (Eigen::Index k=0; k<check5.dimensions()[2]; k++) {
                        Eigen::Matrix<double,-1,-1> mat(twoS1+1,twoS1+1);
                        for (Eigen::Index j=0; j<check5.dimensions()[1]; j++)
                                for (Eigen::Index i=0; i<check5.dimensions()[0]; i++) {
                                        mat(i,j) = check5(i,j,k);
                                }
                        Eigen::MatrixXd pauli(twoS1+1,twoS1+1); pauli.setZero();
                        if (k == 0) {pauli = -std::sqrt(0.6) * (pauli_vec1[0]*pauli_vec1[0]);}
                        if (k == 1) {pauli = -std::sqrt(0.3) * (pauli_vec1[0]*pauli_vec1[1] + pauli_vec1[1]*pauli_vec1[0]);}
                        if (k == 2) {pauli = -std::sqrt(0.1) * (pauli_vec1[0]*pauli_vec1[2] + pauli_vec1[2]*pauli_vec1[0] + 2*pauli_vec1[1]*pauli_vec1[1]);}
                        if (k == 3) {pauli = -std::sqrt(0.3) * (pauli_vec1[2]*pauli_vec1[1] + pauli_vec1[1]*pauli_vec1[2]);}
                        if (k == 4) {pauli = -std::sqrt(0.6) * (pauli_vec1[2]*pauli_vec1[2]);}
                        CHECK((mat-pauli).norm() == doctest::Approx(0.));
                }

                for (int twoS2 = 1; twoS2<11; twoS2++) {
                        Qbasis<Symmetry,1> B2;
                        B2.push_back({twoS2+1},1);
                        double S2 = 0.5*twoS2;

                        Tensor<2,1,Symmetry> s2({{B2,C}},{{B2}}); s2.setConstant(std::sqrt(S2*(S2+1.)));

                        //build the outer product to QN K=0 between s1 and s2
                        auto outerprod1 = (((s1.adjoint().permute<-1>({{0,1,2}}) * couple1.permute<+1>({{0,2,1}})).permute<-1>({{0,1,3,2}})) * s2.permute<+1>({{1,2,0}})).permute<0>({{0,4,2,1,3}});
                        for (const auto& q : outerprod1.sectors()) {
                                CHECK(outerprod1(q).size() == 1);
                                double Stot = 0.5*(q[0]-1.);
                                CHECK(outerprod1(q)(0,0) == doctest::Approx(0.5*(Stot*(Stot+1) - S1*(S1+1.) - S2*(S2+1.))));
                        }                
                }
        }
}

TEST_SUITE_END();
