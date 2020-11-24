#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <TextTable.h>

#define HELPERS_IO_TABLE

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "ArgParser.h"
#include "Qbasis.hpp"
#include "symmetry/SU2.hpp"
#include "symmetry/U0.hpp"
#include "symmetry/U1.hpp"
#include "symmetry/kind_dummies.hpp"

#include "Tensor.hpp"

int main(int argc, char* argv[])
{
        ArgParser args(argc, argv);
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        Qbasis<Symmetry,1> B, C, D;
        int K = args.get<int>("K",1);
        int Dloc = args.get<int>("Dloc",2);

        B.push_back({Dloc},1);
        C.push_back({3},1);
        D.push_back({K},1);
        std::array<Eigen::Matrix<double,3,3>,3 > pauli_vec;
        //1./sqrt(2.)sp
        pauli_vec[0] << 0., 0., 0.,
                std::sqrt(2.), 0., 0.,
                0., std::sqrt(2.), 0.;
        //szy
        pauli_vec[1] << 1., 0., 0.,
                        0., 0., 0.,
                        0., 0., -1.;
        //-1./sqrt(2.)*sm
        pauli_vec[2] << 0., std::sqrt(2.), 0.,
                0., 0., std::sqrt(2.),
                0., 0., 0.;
        double S=args.get<double>("S",0.5*(Dloc-1.));
        Tensor<2,1,Symmetry> id({{C,D}},{{C}}); id.setConstant(1.);
                
        //set s to the reduced Pauli tensor
        Tensor<2,1,Symmetry> s({{B,C}},{{B}}); s.setConstant(std::sqrt(S*(S+1.)));
        
        // auto test = (s.permute<0>({{0,2,1}}) * s.adjoint().permute<0>({{2,1,0}})).permute<0>({{0,3,1,2}});
        auto outerprod = (((s.adjoint().permute<-1>({{0,1,2}}) * id.permute<+1>({{0,2,1}})).permute<-1>({{0,1,3,2}})) * s.permute<+1>({{1,2,0}})).permute<0>({{0,4,2,1,3}});
        auto prod = (s.adjoint().permute<-1>({{0,1,2}}) * id.permute<+1>({{0,2,1}})).permute<0>({{0,3,1,2}}) * s;
        // auto test = ((s.adjoint().permute<-1>({{0,1,2}})) * s.permute<+1>({{1,2,0}})).permute<0>({{0,3,1,2}});
        // auto tplain = test.plainTensor();
        // Eigen::Map<Eigen::MatrixXd> mplain(tplain.data(), tplain.dimensions()[0]*tplain.dimensions()[1], tplain.dimensions()[2]*tplain.dimensions()[3]);
        // cout << "mplain:" << endl << std::fixed << mplain << endl;
        // auto test2 = test.permute<0>(p2);
        // cout << outerprod << endl;
        cout << "r=" << prod.rank() << ", cr=" << prod.corank() << endl;
        // auto tplain = prod.adjoint().plainTensor();
        // for (Eigen::Index k=0; k<tplain.dimensions()[2]; k++) {
        //         Eigen::Matrix<double,-1,-1> pauli(Dloc,Dloc);
        //         for (Eigen::Index j=0; j<tplain.dimensions()[1]; j++)
        //                 for (Eigen::Index i=0; i<tplain.dimensions()[0]; i++) {
        //                         pauli(i,j) = tplain(i,j,k);
        //                 }
        //         cout << "k=" << k << endl << std::fixed << pauli << endl;
        // }
        // cout << endl;
        // auto tplainad = prod.plainTensor();
        // for (Eigen::Index k=0; k<tplainad.dimensions()[1]; k++) {
        //         Eigen::Matrix<double,-1,-1> pauli(Dloc,Dloc);
        //         for (Eigen::Index j=0; j<tplainad.dimensions()[2]; j++)
        //                 for (Eigen::Index i=0; i<tplainad.dimensions()[0]; i++) {
        //                         pauli(i,j) = tplainad(i,k,j);
        //                 }
        //         cout << "k=" << k << endl << std::fixed << pauli << endl;
        // }
        // typedef Sym::U1<Sym::SpinU1> Symmetry;
        // // typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        // constexpr int shift = +1;

        // Qbasis<Symmetry, 1> B, C, D, E;
        // B.setRandom(10);
        // C.setRandom(10); // D.setRandom(3), E.setRandom(2);
        // // B.push_back({-2},1);
        // // B.push_back({0},1);
        // // B.push_back({+2},1);
        // // C.push_back({0},1);
        // // B.sort();
        // // C.sort();
        // Tensor<2, 1, Symmetry> t({{B, C}}, {{B}});
        // t.setRandom();
        // cout << t << endl;
        // Permutation p(std::array<std::size_t, 3>{{2, 1, 0}});
        // auto tp = t.permute<-1>(p);
        // cout << tp << endl;
        // cout << "new rank=" << tp.rank() << ", new corank=" << tp.corank() << endl;
        // auto tplain = t.plainTensor();
        // Eigen::Tensor<double, 3> tplainshuffle = tplain.shuffle(p.pi);

        // auto tplainp = tp.plainTensor();
        // auto check = tplainshuffle - tplainp;

        // Eigen::Tensor<double, 0> zero_ =
        //     check.contract(check,
        //                    Eigen::array<Eigen::IndexPair<Eigen::Index>, 3>{
        //                        {Eigen::IndexPair<Eigen::Index>(0, 0),
        //                         Eigen::IndexPair<Eigen::Index>(1, 1),
        //                         Eigen::IndexPair<Eigen::Index>(2, 2)}});
        // //                                                                                                                      Eigen::IndexPair<Eigen::Index>(3,3)}});
        // double zero = zero_();

        // std::cout << endl << "tplain:" << endl;
        // auto it_block = tplain.data();
        // for (Eigen::Index l=0; l<tplain.dimensions()[3]; l++)
        // for (Eigen::Index k=0; k<tplain.dimensions()[2]; k++)
        // {
        //         Eigen::Map<Eigen::MatrixXd> Mp(it_block, tplain.dimensions()[0],tplain.dimensions()[1]);
        //         std::cout << "[:,:," << k << "," << l << "]:" << endl << std::fixed << Mp << endl;
        //         it_block += tplain.dimensions()[0] * tplain.dimensions()[1];
        // }

        // std::cout << endl << "tplain shuffle:" << endl;
        // auto it_blocks = tplainshuffle.data();
        // // for (Eigen::Index l=0; l<tplainshuffle.dimensions()[3]; l++)
        // for (Eigen::Index k=0; k<tplainshuffle.dimensions()[2]; k++)
        // {
        //         Eigen::Map<Eigen::MatrixXd> Mp(it_blocks,
        //         tplainshuffle.dimensions()[0], tplainshuffle.dimensions()[1]);
        //         std::cout << "[:,:," << k << "]:" << endl << std::fixed << Mp <<
        //         endl; it_blocks += tplainshuffle.dimensions()[0] *
        //         tplainshuffle.dimensions()[1];
        // }

        // std::cout << endl << "tplain fancy:" << endl;
        // auto it_blockf = tplainp.data();
        // // for (Eigen::Index l=0; l<tplainp.dimensions()[3]; l++)
        // for (Eigen::Index k=0; k<tplainp.dimensions()[2]; k++)
        // {
        //         Eigen::Map<Eigen::MatrixXd> Mp(it_blockf, tplainp.dimensions()[0],
        //         tplainp.dimensions()[1]); std::cout << "[:,:," << k << "]:" << endl
        //         << std::fixed << Mp << endl; it_blockf += tplainp.dimensions()[0] *
        //         tplainp.dimensions()[1];
        // }
        // cout << "zero=" << zero << endl;
        // std::cout << endl << "tplain fancy:" << endl;
        // auto it_blockf = tplainp.data();
        // for (Eigen::Index k=0; k<tplainp.dimensions()[2]; k++)
        // {
        //         Eigen::Map<Eigen::MatrixXd> Mp(it_blockf, tplainp.dimensions()[0],
        //         tplainp.dimensions()[1]); std::cout << "[:,:," << k << "]:" << endl
        //         << std::fixed << Mp << endl; it_blockf += tplainp.dimensions()[0] *
        //         tplainp.dimensions()[1];
        // }
        // cout << std::fixed << tplainp << endl << endl << tplainshuffle << endl <<
        // endl << tplain << endl;

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
