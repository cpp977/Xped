#include <iostream>
#include <cstddef>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

#include <TextTable.h>

#define HELPERS_IO_TABLE

using std::size_t;
using std::cout;
using std::endl;
using std::string;

#include "ArgParser.h"
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
                Tensor<2,2,Symmetry> t({{B,C}},{{D,E}}); t.setRandom();
                auto tplain = t.plainTensor();
                Eigen::Tensor<double,0> norm_ = tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(3,3)}});
                double norm = norm_();
                CHECK(t.squaredNorm() == doctest::Approx(norm));
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                Tensor<2,2,Symmetry> t({{B,C}},{{D,E}}); t.setRandom();
                auto tplain = t.plainTensor();
                Eigen::Tensor<double,0> norm_ = tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(3,3)}});
                double norm = norm_();
                CHECK(t.squaredNorm() == doctest::Approx(norm));
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                Tensor<2,2,Symmetry> t({{B,C}},{{D,E}}); t.setRandom();
                auto tplain = t.plainTensor();
                Eigen::Tensor<double,0> norm_ = tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                Eigen::IndexPair<Eigen::Index>(3,3)}});
                double norm = norm_();
                CHECK(t.squaredNorm() == doctest::Approx(norm));
        }
}

TEST_CASE("Testing the SU(2)-Pauli matrices.") {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry,1> B, C;
        B.push_back({2},1);
        C.push_back({3},1);

        std::array<Eigen::Matrix<double,2,2>,3 > pauli_vec;
        //1./sqrt(2.)sp
        pauli_vec[0] << 0., 0.,
                1./std::sqrt(2.), 0.;
        //szy
        pauli_vec[1] << -0.5, 0.,
                0., 0.5;
        //-1./sqrt(2.)*sm
        pauli_vec[2] << 0., -1./std::sqrt(2.),
                0., 0.;
        
        //set t to the reduced Pauli tensor
        Tensor<2,1,Symmetry> t({{B,C}},{{B}}); t.setConstant(std::sqrt(3.)/2.);
        //transform to plain tensor and check against pauli_vec
        auto tplain = t.plainTensor();
        for (Eigen::Index j=0; j<tplain.dimensions()[1]; j++) {
                Eigen::Matrix<double,2,2> pauli;
                for (Eigen::Index k=0; k<tplain.dimensions()[2]; k++)
                        for (Eigen::Index i=0; i<tplain.dimensions()[0]; i++) {
                                pauli(i,k) = tplain(i,j,k);
                        }
                CHECK((pauli-pauli_vec[j]).norm() == doctest::Approx(0.));
        }
}

TEST_CASE("Testing the permutation within the domain.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                Tensor<4,0,Symmetry> t({{B,C,D,E}},{{}}); t.setRandom();

                Permutation<4> p(std::array<std::size_t,4>{{2,0,3,1}});
                Permutation<0> ptriv(std::array<std::size_t,0>{{}});
                auto tp = t.permute(p,ptriv);

                auto tplain = t.plainTensor();
                std::array<std::size_t,4> ptot = {{2,0,3,1}};
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(ptot);

                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;

                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                Tensor<4,0,Symmetry> t({{B,C,D,E}},{{}}); t.setRandom();

                Permutation<4> p(std::array<std::size_t,4>{{2,0,3,1}});
                Permutation<0> ptriv(std::array<std::size_t,0>{{}});
                auto tp = t.permute(p,ptriv);

                auto tplain = t.plainTensor();
                std::array<std::size_t,4> ptot = {{2,0,3,1}};
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(ptot);

                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;

                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
                Tensor<4,0,Symmetry> t({{B,C,D,E}},{{}}); t.setRandom();

                Permutation<4> p(std::array<std::size_t,4>{{2,0,3,1}});
                Permutation<0> ptriv(std::array<std::size_t,0>{{}});
                auto tp = t.permute(p,ptriv);

                auto tplain = t.plainTensor();
                std::array<std::size_t,4> ptot = {{2,0,3,1}};
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(ptot);

                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;

                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }
}

TEST_CASE("Testing the permutation within the codomain.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(5); C.setRandom(5); D.setRandom(5), E.setRandom(5);
                Tensor<0,4,Symmetry> t({{}},{{B,C,D,E}}); t.setRandom();

                Permutation<4> p(std::array<std::size_t,4>{{2,0,3,1}});
                Permutation<0> ptriv(std::array<std::size_t,0>{{}});
                auto tp = t.permute(ptriv,p);

                auto tplain = t.plainTensor();
                std::array<std::size_t,4> ptot = {{2,0,3,1}};
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(ptot);

                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;

                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(4); C.setRandom(4); D.setRandom(4), E.setRandom(4);
                Tensor<0,4,Symmetry> t({{}},{{B,C,D,E}}); t.setRandom();

                Permutation<4> p(std::array<std::size_t,4>{{2,0,3,1}});
                Permutation<0> ptriv(std::array<std::size_t,0>{{}});
                auto tp = t.permute(ptriv,p);

                auto tplain = t.plainTensor();
                std::array<std::size_t,4> ptot = {{2,0,3,1}};
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(ptot);

                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;

                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(3); C.setRandom(3); D.setRandom(3), E.setRandom(3);
                Tensor<0,4,Symmetry> t({{}},{{B,C,D,E}}); t.setRandom();

                Permutation<4> p(std::array<std::size_t,4>{{2,0,3,1}});
                Permutation<0> ptriv(std::array<std::size_t,0>{{}});
                auto tp = t.permute(ptriv,p);

                auto tplain = t.plainTensor();
                std::array<std::size_t,4> ptot = {{2,0,3,1}};
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(ptot);

                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;

                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }
}

TEST_CASE("Testing the general permutation of legs.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(5); C.setRandom(5);
                
                Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
                auto tplain = t.plainTensor();
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<+2>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<+1>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<0>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<-1>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<-2>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(4); C.setRandom(4);
                
                Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
                auto tplain = t.plainTensor();
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<+2>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<+1>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<0>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<-1>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<-2>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
        //         test_tensor_permute<-2>(B,C);
        //         test_tensor_permute<-1>(B,C);
        //         test_tensor_permute<+0>(B,C);
        //         test_tensor_permute<+1>(B,C);
        //         test_tensor_permute<+2>(B,C);
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(3); C.setRandom(3);
                Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
                auto tplain = t.plainTensor();
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<+2>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<+1>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<0>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<-1>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
                for (const auto& p : Permutation<4>::all()) {
                        auto tp = t.permute<-2>(p);
                        auto tplainp = tp.plainTensor();
                        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
                        auto check = tplainshuffle - tplainp;
                        Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                                     Eigen::IndexPair<Eigen::Index>(3,3)}});
                        CHECK(zero() == doctest::Approx(0.));
                }
        //         test_tensor_permute<-2>(B,C);
        //         test_tensor_permute<-1>(B,C);
        //         test_tensor_permute<+0>(B,C);
        //         test_tensor_permute<+1>(B,C);
        //         test_tensor_permute<+2>(B,C);
        }
}
TEST_SUITE_END();
