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

#include "doctest/doctest.h"

TEST_SUITE_BEGIN("FusionTrees");

TEST_CASE("Testing the elementary swap.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;

                Qbasis<Symmetry,1> B,C; B.setRandom(100); C.setRandom(100);
                auto BC = B.combine(C);
                auto BCB = BC.combine(B);
                auto BCBC = BCB.combine(C);

                for (std::size_t pos=0ul; pos<3; pos++) {
                        for (const auto& tree : BCBC.tree({3})) {
                                // std::cout << "swap test for tree:" << endl << tree.draw() << endl;
                                std::unordered_map<FusionTree<4, Symmetry>, Symmetry::Scalar> check;
                                auto transformed = tree.swap(pos);
                                // cout << tree.draw() << endl << "transforms to:" << endl;
                                for (const auto& [tree_p,coeff] :transformed) {
                                        auto inv = tree_p.swap(pos);
                                        for (const auto& [tree_inv,coeff2] : inv) {
                                                auto it = check.find(tree_inv);
                                                if (it == check.end()) {check.insert(std::make_pair(tree_inv, coeff*coeff2));}
                                                else (check[tree_inv] += coeff*coeff2);
                                        }
                                }
                                for (const auto& [tree_check,coeff]:check) {if (tree_check == tree) {
                                                CHECK(coeff == doctest::Approx(1.));
                                        }
                                        else {
                                                CHECK(coeff == doctest::Approx(0.));
                                        }
                                }
                        }
                }
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;

                Qbasis<Symmetry,1> B,C; B.setRandom(50); C.setRandom(50);
                auto BC = B.combine(C);
                auto BCB = BC.combine(B);
                auto BCBC = BCB.combine(C);
                for (std::size_t pos=0ul; pos<3; pos++) {
                        for (const auto& tree : BCBC.tree({1})) {
                                // std::cout << "swap test for tree:" << endl << tree.draw() << endl;
                                std::unordered_map<FusionTree<4, Symmetry>, Symmetry::Scalar> check;
                                auto transformed = tree.swap(pos);
                                // cout << tree.draw() << endl << "transforms to:" << endl;
                                for (const auto& [tree_p,coeff] :transformed) {
                                        auto inv = tree_p.swap(pos);
                                        for (const auto& [tree_inv,coeff2] : inv) {
                                                auto it = check.find(tree_inv);
                                                if (it == check.end()) {check.insert(std::make_pair(tree_inv, coeff*coeff2));}
                                                else (check[tree_inv] += coeff*coeff2);
                                        }
                                }
                                for (const auto& [tree_check,coeff]:check) {if (tree_check == tree) {
                                                CHECK(coeff == doctest::Approx(1.));
                                        }
                                        else {
                                                CHECK(coeff == doctest::Approx(0.));
                                        }
                                }
                        }
                }
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;

                Qbasis<Symmetry,1> B,C; B.setRandom(50); C.setRandom(50);
                auto BC = B.combine(C);
                auto BCB = BC.combine(B);
                auto BCBC = BCB.combine(C);

                for (std::size_t pos=0ul; pos<3; pos++) {
                        for (const auto& tree : BCBC.tree({})) {
                                // std::cout << "swap test for tree:" << endl << tree.draw() << endl;
                                std::unordered_map<FusionTree<4, Symmetry>, Symmetry::Scalar> check;
                                auto transformed = tree.swap(pos);
                                // cout << tree.draw() << endl << "transforms to:" << endl;
                                for (const auto& [tree_p,coeff] :transformed) {
                                        auto inv = tree_p.swap(pos);
                                        for (const auto& [tree_inv,coeff2] : inv) {
                                                auto it = check.find(tree_inv);
                                                if (it == check.end()) {check.insert(std::make_pair(tree_inv, coeff*coeff2));}
                                                else (check[tree_inv] += coeff*coeff2);
                                        }
                                }
                                for (const auto& [tree_check,coeff]:check) {if (tree_check == tree) {
                                                CHECK(coeff == doctest::Approx(1.));
                                        }
                                        else {
                                                CHECK(coeff == doctest::Approx(0.));
                                        }
                                }
                        }
                }
        }
}

TEST_CASE("Testing the permutation.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(100); C.setRandom(100);
                auto BC = B.combine(C);
                auto BCB = BC.combine(B);
                auto BCBC = BCB.combine(C);;
        
                for (const auto& tree : BCBC.tree({3})) {
                        std::unordered_map<FusionTree<4, Symmetry>, Symmetry::Scalar> check;
                        std::array<std::size_t,4> a = {{2ul,3ul,0ul,1ul}};
                        Permutation p(a);
                        Permutation pinv(p.pi_inv);
                        auto transformed = tree.permute(p);
                        // cout << tree.draw() << endl << "transforms to:" << endl;
                        for (const auto& [tree_p,coeff] :transformed) {
                                auto inv = tree_p.permute(pinv);
                                for (const auto& [tree_inv,coeff2] : inv) {
                                        auto it = check.find(tree_inv);
                                        if (it == check.end()) {check.insert(std::make_pair(tree_inv, coeff*coeff2));}
                                        else (check[tree_inv] += coeff*coeff2);
                                }
                        }
                        for (const auto& [tree_check,coeff]:check) {if (tree_check == tree) {
                                        CHECK(coeff == doctest::Approx(1.));
                                }
                                else {
                                        CHECK(coeff == doctest::Approx(0.));
                                }
                        }
                }
        }

        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(50); C.setRandom(50);
                auto BC = B.combine(C);
                auto BCB = BC.combine(B);
                auto BCBC = BCB.combine(C);;
        
                for (const auto& tree : BCBC.tree({3})) {
                        std::unordered_map<FusionTree<4, Symmetry>, Symmetry::Scalar> check;
                        std::array<std::size_t,4> a = {{2ul,3ul,0ul,1ul}};
                        Permutation p(a);
                        Permutation pinv(p.pi_inv);
                        auto transformed = tree.permute(p);
                        // cout << tree.draw() << endl << "transforms to:" << endl;
                        for (const auto& [tree_p,coeff] :transformed) {
                                auto inv = tree_p.permute(pinv);
                                for (const auto& [tree_inv,coeff2] : inv) {
                                        auto it = check.find(tree_inv);
                                        if (it == check.end()) {check.insert(std::make_pair(tree_inv, coeff*coeff2));}
                                        else (check[tree_inv] += coeff*coeff2);
                                }
                        }
                        for (const auto& [tree_check,coeff]:check) {if (tree_check == tree) {
                                        CHECK(coeff == doctest::Approx(1.));
                                }
                                else {
                                        CHECK(coeff == doctest::Approx(0.));
                                }
                        }
                }
        }

        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B,C; B.setRandom(50); C.setRandom(50);
                auto BC = B.combine(C);
                auto BCB = BC.combine(B);
                auto BCBC = BCB.combine(C);;
        
                for (const auto& tree : BCBC.tree({})) {
                        std::unordered_map<FusionTree<4, Symmetry>, Symmetry::Scalar> check;
                        std::array<std::size_t,4> a = {{2ul,3ul,0ul,1ul}};
                        Permutation p(a);
                        Permutation pinv(p.pi_inv);
                        auto transformed = tree.permute(p);
                        // cout << tree.draw() << endl << "transforms to:" << endl;
                        for (const auto& [tree_p,coeff] :transformed) {
                                auto inv = tree_p.permute(pinv);
                                for (const auto& [tree_inv,coeff2] : inv) {
                                        auto it = check.find(tree_inv);
                                        if (it == check.end()) {check.insert(std::make_pair(tree_inv, coeff*coeff2));}
                                        else (check[tree_inv] += coeff*coeff2);
                                }
                        }
                        for (const auto& [tree_check,coeff]:check) {if (tree_check == tree) {
                                        CHECK(coeff == doctest::Approx(1.));
                                }
                                else {
                                        CHECK(coeff == doctest::Approx(0.));
                                }
                        }
                }
        }
}

TEST_CASE("Testing the turn operation for FusionTree pairs.") {
        SUBCASE("SU2") {
                typedef Sym::SU2<Sym::SpinSU2> Symmetry;
                Qbasis<Symmetry,1> B, C; B.setRandom(100); C.setRandom(100);
                Qbasis<Symmetry,1> D, E; D.setRandom(100); E.setRandom(100);
        
                auto BC = B.combine(C);
                auto DE = B.combine(C);
                for (const auto& [q, num, plain] : BC) {
                        if (DE.NOT_PRESENT(q)) {continue;}
                        for (const auto& t1: BC.tree({q}))
                                for (const auto& t2: DE.tree({q})) {
                                        for (const auto& [trees1,coeff1] : treepair::turn<-2>(t1,t2)) {
                                                auto [t1p,t2p] = trees1;
                                                for (const auto& [trees2,coeff2] : treepair::turn<+4>(t1p,t2p)) {
                                                        auto [t1pp,t2pp] = trees2;
                                                        for (const auto& [trees3,coeff3] : treepair::turn<-2>(t1pp,t2pp)) {
                                                                auto [t1ppp,t2ppp] = trees3;
                                                                CHECK(coeff1*coeff2*coeff3 == doctest::Approx(1.));
                                                        }
                                                }               
                                        }
                                }
                }
        }
        SUBCASE("U1") {
                typedef Sym::U1<Sym::SpinU1> Symmetry;
                Qbasis<Symmetry,1> B, C; B.setRandom(100); C.setRandom(100);
                Qbasis<Symmetry,1> D, E; D.setRandom(100); E.setRandom(100);
        
                auto BC = B.combine(C);
                auto DE = B.combine(C);
                for (const auto& [q, num, plain] : BC) {
                        if (DE.NOT_PRESENT(q)) {continue;}
                        for (const auto& t1: BC.tree({q}))
                                for (const auto& t2: DE.tree({q})) {
                                        for (const auto& [trees1,coeff1] : treepair::turn<-2>(t1,t2)) {
                                                auto [t1p,t2p] = trees1;
                                                for (const auto& [trees2,coeff2] : treepair::turn<+4>(t1p,t2p)) {
                                                        auto [t1pp,t2pp] = trees2;
                                                        for (const auto& [trees3,coeff3] : treepair::turn<-2>(t1pp,t2pp)) {
                                                                auto [t1ppp,t2ppp] = trees3;
                                                                CHECK(coeff1*coeff2*coeff3 == doctest::Approx(1.));
                                                        }
                                                }               
                                        }
                                }
                }
        }
        SUBCASE("U0") {
                typedef Sym::U0 Symmetry;
                Qbasis<Symmetry,1> B, C; B.setRandom(100); C.setRandom(100);
                Qbasis<Symmetry,1> D, E; D.setRandom(100); E.setRandom(100);
        
                auto BC = B.combine(C);
                auto DE = B.combine(C);
                for (const auto& [q, num, plain] : BC) {
                        if (DE.NOT_PRESENT(q)) {continue;}
                        for (const auto& t1: BC.tree({q}))
                                for (const auto& t2: DE.tree({q})) {
                                        for (const auto& [trees1,coeff1] : treepair::turn<-2>(t1,t2)) {
                                                auto [t1p,t2p] = trees1;
                                                for (const auto& [trees2,coeff2] : treepair::turn<+4>(t1p,t2p)) {
                                                        auto [t1pp,t2pp] = trees2;
                                                        for (const auto& [trees3,coeff3] : treepair::turn<-2>(t1pp,t2pp)) {
                                                                auto [t1ppp,t2ppp] = trees3;
                                                                CHECK(coeff1*coeff2*coeff3 == doctest::Approx(1.));
                                                        }
                                                }               
                                        }
                                }
                }
        }
}
TEST_SUITE_END();
