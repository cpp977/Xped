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
        constexpr int shift = -2;
        // typedef Sym::U0 Symmetry;
        Qbasis<Symmetry,1> B, C; B.setRandom(100); C.setRandom(100);
        Qbasis<Symmetry,1> D, E; D.setRandom(100); E.setRandom(100);
        
        auto BC = B.combine(C);
        auto DE = B.combine(C);
        for (const auto& [q, num, plain] : BC) {
                if (DE.NOT_PRESENT(q)) {continue;}
                for (const auto& t1: BC.tree({q}))
                for (const auto& t2: DE.tree({q})) {
                        cout << t1.draw() << endl << t2.draw() << endl;
                        for (const auto& [trees,coeff] : treepair::turn<shift>(t1,t2)) {
                                auto [t1p,t2p] = trees;
                                cout << t1p.draw() << endl << t2p.draw() << endl << "coeff=" << coeff << endl;
                                for (const auto& [retrees,recoeff] : treepair::turn<-shift>(t1p,t2p)) {
                                        auto [t1pp,t2pp] = retrees;
                                        cout << t1pp.draw() << endl << t2pp.draw() << endl << "coeff=" << recoeff << ", coeff*coeff=" << coeff*recoeff << endl;
                                        assert(std::abs(coeff*recoeff-1.) < 1.e-12);
                                }               
                        }
                }
        }
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
