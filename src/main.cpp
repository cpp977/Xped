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

#include "Tensor.hpp"

int main(int argc, char *argv[])
{
	ArgParser args(argc,argv);
        Qbasis<Sym::SU2<Sym::SpinSU2>,1ul> B, C;
        B.push_back({2},4);
        B.push_back({1},1);
        B.push_back({4},3);
        C.push_back({3},2);
        C.push_back({2},1);
        C.push_back({6},1);
        cout << B << endl;
        cout << "B.dim()=" << B.dim() << ", B.fullDim()=" << B.fullDim() << endl;

        // auto Bsq = B.combine(B);
        // cout << Bsq.printTrees() << endl;
        // cout << std::boolalpha << (Bsq.fullDim() == B.fullDim()*B.fullDim()) << endl;
        // auto Bcube = Bsq.combine(B);
        // cout << Bcube.printTrees() << endl;
        // cout << std::boolalpha << (Bcube.fullDim() == B.fullDim()*B.fullDim()*B.fullDim()) << endl;
        // auto Bfourth = Bcube.combine(B);
        // cout << Bfourth.printTrees() << endl;
        // cout << std::boolalpha << (Bfourth.fullDim() == B.fullDim()*B.fullDim()*B.fullDim()*B.fullDim()) << endl;
        // cout << Bfourth << endl;

        Tensor<1,2,Sym::SU2<Sym::SpinSU2>, Eigen::SparseMatrix<double> > t({{B}},{{B,C}}); t.setZero();
        auto tdag = t.adjoint();
        auto tsq = t*tdag;
        // Tensor<2,2,Sym::SU2<Sym::SpinSU2> > t({{B,C}},{{B,C}}); t.setIdentity();
        // Tensor<2,1,Sym::SU2<Sym::SpinSU2> > tp = t.permute({1},{2,3});
        cout << tsq << endl;
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
