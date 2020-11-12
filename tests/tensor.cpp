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

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
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

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
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

                Qbasis<Symmetry,1> B,C,D,E; B.setRandom(10); C.setRandom(10); D.setRandom(10), E.setRandom(10);
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

TEST_SUITE_END();
