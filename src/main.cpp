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
        Qbasis<Symmetry,1> B, C, D; B.setRandom()
        B.push_back({1},2);
        B.push_back({3},1);
        cout << B.printTrees() << endl;
        // B.push_back({3},1);
        C.push_back({1},2);
        C.push_back({2},1);
        // C.push_back({3},1);
        D.push_back({3},1);
        D.push_back({2},1);
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

        Tensor<4,0,Symmetry> t({{B,C,D,B}},{{}}); t.setRandom();
        std::cout << "norm=" << t.normSquared() << std::endl;
        auto tplain = t.plainTensor();
        std::cout << "norm plain=" << tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                      Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                      Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                      Eigen::IndexPair<Eigen::Index>(3,3)}}) << endl;
        Permutation<4> p(std::array<std::size_t,4>{{2,0,3,1}});
        Permutation<0> ptriv(std::array<std::size_t,0>{{}});
        std::array<std::size_t,4> ptot = {{2,0,3,1}};
        auto tp = t.permute(p,ptriv);
        // std::cout << t.print(true) << std::endl;//< endl << tp << endl << endl;;

        auto tplainp = tp.plainTensor();
        Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(ptot);

        auto check = tplainshuffle - tplainp;
        // // cout << tplain << endl;
        // std::cout << "total tensor dims="; for (const auto& d:tplain.dimensions()) {std::cout << d << " ";} std::cout << endl;
        // // std::cout << tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 3>{{Eigen::IndexPair<Eigen::Index>(0,0),
        // //                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
        // //                                                                                              Eigen::IndexPair<Eigen::Index>(2,2)}}) << endl;
        // // std::cout << tplain << endl;
        std::cout << "check=" << check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                               Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                               Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                               Eigen::IndexPair<Eigen::Index>(3,3)}}) << endl;
        // std::cout << "tplain:" << endl;
        // auto it_block = tplain.data();
        // for (Eigen::Index k=0; k<tplain.dimensions()[2]; k++)        
        // {
        //         Eigen::Map<Eigen::MatrixXd> Mp(it_block, tplain.dimensions()[0], tplain.dimensions()[1]);
        //         std::cout << "[:,:," << k << "]:" << endl << std::fixed << Mp << endl;
        //         it_block += tplain.dimensions()[0] * tplain.dimensions()[1];
        // }

        // std::cout << endl << "tplain shuffle:" << endl;
        // auto it_blocks = tplainshuffle.data();
        // for (Eigen::Index k=0; k<tplainshuffle.dimensions()[2]; k++)        
        // {
        //         Eigen::Map<Eigen::MatrixXd> Mp(it_blocks, tplainshuffle.dimensions()[0], tplainshuffle.dimensions()[1]);
        //         std::cout << "[:,:," << k << "]:" << endl << std::fixed << Mp << endl;
        //         it_blocks += tplainshuffle.dimensions()[0] * tplainshuffle.dimensions()[1];
        // }

        // std::cout << endl << "tplain fancy:" << endl;
        // auto it_blockf = tplainp.data();
        // for (Eigen::Index k=0; k<tplainp.dimensions()[2]; k++)        
        // {
        //         Eigen::Map<Eigen::MatrixXd> Mp(it_blockf, tplainp.dimensions()[0], tplainp.dimensions()[1]);
        //         std::cout << "[:,:," << k << "]:" << endl << std::fixed << Mp << endl;
        //         it_blockf += tplainp.dimensions()[0] * tplainp.dimensions()[1];
        // }
        // cout << std::fixed << tplainp << endl << endl << tplainshuffle << endl << endl << tplain << endl;

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
