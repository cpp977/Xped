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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

TEST_SUITE_BEGIN("FusionTrees");

TEST_CASE("Testing the elementary swap.") {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        // std::size_t pos = args.get<std::size_t>("pos",1);
        // typedef Sym::U0 Symmetry;
        Qbasis<Symmetry,1> B,C;
        B.push_back({3},1);
        B.push_back({4},1);
        B.push_back({4},3);
        C.push_back({2},1);
        C.push_back({2},1);
        C.push_back({6},1);

        auto Bsq = B.combine(C);
        auto Bcube = Bsq.combine(B);
        auto Bfourth = Bcube.combine(C);

        for (std::size_t pos=0ul; pos<3; pos++) {
                for (const auto& tree : Bfourth.tree({3})) {
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
                                        CHECK(std::abs(coeff-1.0) < 1.e-12);
                                }
                                else {
                                        CHECK(std::abs(coeff) < 1.e-12);
                                }
                        }
                }
        }
}

TEST_CASE("Testing the permutation.") {
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        // std::size_t pos = args.get<std::size_t>("pos",1);
        // typedef Sym::U0 Symmetry;
        Qbasis<Symmetry,1> B,C;
        B.push_back({3},1);
        B.push_back({4},1);
        B.push_back({4},3);
        C.push_back({2},1);
        C.push_back({2},1);
        C.push_back({6},1);

        auto Bsq = B.combine(C);
        auto Bcube = Bsq.combine(B);
        auto Bfourth = Bcube.combine(C);
        
        for (const auto& tree : Bfourth.tree({3})) {
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
                                CHECK(std::abs(coeff-1.0) < 1.e-12);
                        }
                        else {
                                CHECK(std::abs(coeff) < 1.e-12);
                        }
                }
        }
}

TEST_SUITE_END();
