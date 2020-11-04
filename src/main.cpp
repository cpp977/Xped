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
#include "Qbasis.hpp"
#include "symmetry/kind_dummies.hpp"
#include "symmetry/SU2.hpp"
#include "symmetry/U1.hpp"
#include "symmetry/U0.hpp"

#include "Tensor.hpp"

int main(int argc, char *argv[])
{
	ArgParser args(argc,argv);
        // typedef Sym::U1<Sym::SpinU1> Symmetry;
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        // typedef Sym::U0 Symmetry;
        Qbasis<Symmetry,1> B, C;
        B.push_back({1},2);
        B.push_back({2},2);
        // B.push_back({3},1);
        C.push_back({1},2);
        C.push_back({2},2);
        // C.push_back({3},1);
        cout << B << endl;
        cout << "B.dim()=" << B.dim() << ", B.fullDim()=" << B.fullDim() << endl;

        auto Bsq = B.combine(C);
        
        // auto map = Bsq.tree({})[0].swap(0);
        cout << std::boolalpha << (Bsq.fullDim() == B.fullDim()*C.fullDim()) << endl;
        // cout << Bsq << endl;
        auto Bcube = Bsq.combine(B);
        // cout << Bcube.printTrees() << endl;
        // cout << std::boolalpha << (Bcube.fullDim() == B.fullDim()*B.fullDim()*B.fullDim()) << endl;
        auto Bfourth = Bcube.combine(C);
        // cout << Bfourth.printTrees() << endl;
        // cout << std::boolalpha << (Bfourth.fullDim() == B.fullDim()*B.fullDim()*B.fullDim()*B.fullDim()) << endl;
        // cout << Bfourth << endl;

        Tensor<5,0,Symmetry> t({{B,C,B,B,C}},{{}}); t.setRandom();
        std::cout << "norm=" << t.normSquared() << std::endl;
        std::cout << t.print(false) << std::endl;
        auto tplain = t.plainTensor();
        // cout << tplain << endl;
        std::cout << "total tensor dims="; for (const auto& d:tplain.dimensions()) {std::cout << d << " ";} std::cout << endl;
        // std::cout << tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 3>{{Eigen::IndexPair<Eigen::Index>(0,0),
        //                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
        //                                                                                              Eigen::IndexPair<Eigen::Index>(2,2)}}) << endl;
        // std::cout << tplain << endl;
        std::cout << tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 5>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                           Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                           Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                           Eigen::IndexPair<Eigen::Index>(3,3),
                                                                                           Eigen::IndexPair<Eigen::Index>(4,4)}}) << endl;
        // Permutation<4> p(std::array<std::size_t,4>{{0,2,1,3}});
        // //std::cout << p.print() << std::endl << p.inverse().print() << endl;
        // auto tp = t.permute(p,Permutation<0>::Identity());
        // // std::cout << tp << std::endl;
        // auto tpp = tp.permute(p.inverse(),Permutation<0>::Identity());
        // // std::cout << tpp << std::endl;
        // cout << "norm=" << (t-tpp).norm() << endl;
        // cout << std::boolalpha << ((t-tpp).norm() < 1.e-12) << endl;

        // Tensor<2,2,Sym::SU2<Sym::SpinSU2> > t({{B,C}},{{B,C}}); t.setIdentity();
        // Tensor<2,1,Sym::SU2<Sym::SpinSU2> > tp = t.permute({1},{2,3});
        // cout << tsq << endl;
        // for (const auto& d_tree: t.domain_trees({1}))
        //         for (const auto& c_tree: t.codomain_trees({1})) {
        //                 const auto M = t(d_tree, c_tree);
        //                 auto T = t.getBlockTensor(d_tree,c_tree);
        //                 cout << "domain: " << endl << d_tree << endl <<
        //                         "codomain: " << endl << c_tree << endl <<
        //                         "block as matrix: " << endl << M << endl << 
        //                         "block as tensor:" << endl << T << endl << endl;
        //         }
}
