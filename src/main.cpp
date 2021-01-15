//#define DONT_USE_BDCSVD

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <TextTable.h>

#include "macros.h"
#include "Stopwatch.h"

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "ArgParser.h"


// #define CACHE_PERMUTE_OUTPUT

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

#include "Qbasis.hpp"
#include "symmetry/SU2.hpp"
#include "symmetry/U0.hpp"
#include "symmetry/U1.hpp"
#include "symmetry/kind_dummies.hpp"

#include "Tensor.hpp"
// #include "Mps.hpp"
// #include "MpsAlgebra.hpp"

#include "interfaces/tensor_traits.hpp"

int main(int argc, char* argv[])
{
        ArgParser args(argc, argv);
        // constexpr std::size_t Rank=2;
        // tensortraits<EigenTensorLib>::Ttype<double,Rank> T1(2,3); T1.setRandom();
        // tensortraits<EigenTensorLib>::Ttype<double,Rank> T2(2,3); T2.setRandom();
        // std::array<std::pair<std::size_t,std::size_t>,1> legs = {std::pair<std::size_t,std::size_t>{0,0}};
        // auto C = tensortraits<EigenTensorLib>::contract(T1,T2,legs);
        // auto T = tensortraits<EigenTensorLib>::tensorProd(T1,T1);
        // auto Ts = tensortraits<EigenTensorLib>::shuffle(C, {{1,0}});
        // std::array<std::size_t, 2> dims = {2,3};
        // auto cmap = tensortraits<EigenTensorLib>::cMap(T.data(), dims);
        // auto map = tensortraits<EigenTensorLib>::Map(T.data(), dims);
        // map(0,0)=10;
        // std::cout << cmap(0,0) << std::endl;

        // std::cout << T << std::endl;

        // std::cout << Ts << std::endl;

        // std::cout << std::endl << C << std::endl;


        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        typedef Symmetry::qType qType;
        auto L = args.get<std::size_t>("L",10);
        auto D = args.get<int>("D",1);
        auto Minit = args.get<std::size_t>("Minit",10);
        auto Qinit = args.get<std::size_t>("Qinit",10);
        auto reps = args.get<std::size_t>("reps",10);
        // auto DIR = static_cast<DMRG::DIRECTION>(args.get<int>("DIR",0));
        auto NORM = args.get<bool>("NORM",true);
        auto SWEEP = args.get<bool>("SWEEP",true);
        auto INFO = args.get<bool>("INFO",true);
        
        qType Qtot = {D}; 
        Qbasis<Symmetry,1> qloc_; qloc_.push_back({2},1);
        std::vector<Qbasis<Symmetry,1>> qloc(L,qloc_);
        Qbasis<Symmetry,1> in; in.setRandom(Minit);
        auto out = in.combine(qloc_).forgetHistory();
        // // Stopwatch<> construct;
        // // Mps<Symmetry> Psi(L,qloc,Qtot,Minit,Qinit);
        // // cout << construct.info("Time for constructor") << endl;
        Tensor<2,1,Symmetry> T({{in, qloc_}}, {{out}}); T.setRandom();
        // for (std::size_t i=0; i<reps; i++) {
        //         auto Ap = T.template permute<1>({{0,2,1}});
        // }
        // if (INFO) {
        //         for (size_t l=0; l<=L; l++) {
        //                 cout << Psi.auxBasis(l) << endl;
        //         }
        // }
        // if (NORM) {
        //         Stopwatch<> norm;
        //         for (std::size_t i=0; i<reps; i++) {
        //                 double normSq = dot(Psi,Psi,DIR);
        //                 // std::cout << "<Psi|Psi>=" << normSq << std::endl;
        //         }
        //         cout << norm.info("Time for norm") << endl;
        // }
        // if (SWEEP) {
        //         Stopwatch<> Sweep;
        //         if (DIR == DMRG::DIRECTION::RIGHT) {
        //                 for (std::size_t l=0; l<L; l++) {
        //                         // cout << l << endl;
        //                         Psi.rightSweepStep(l, DMRG::BROOM::SVD);
        //                 }
        //         }
        //         else {
        //                 for (std::size_t l=L-1; l>0; l--) {
        //                         Psi.leftSweepStep(l, DMRG::BROOM::SVD);
        //                 }
        //                 Psi.leftSweepStep(0, DMRG::BROOM::SVD);
        //         }
        //         cout << Sweep.info("Time for sweep") << endl;
        // }
        // double trunc;
        // auto [U,S,V] = Psi.A.Ac[L/2].tSVD(10000ul, 1.e-10, trunc, false);
        // Mps<Symmetry> Phi(L,qloc,Qtot,Minit,Qinit);
        // cout << "<Phi,Psi>=" << dot(Phi,Psi) << endl;
        // cout << "<Phi,Phi>=" << dot(Phi,Phi) << endl;
        // normSq = dot(Psi,Psi,DIR);
        // std::cout << "<Psi|Psi>=" << normSq << std::endl;
        // cout << "Left: <Psi,Psi>=" << normSq << endl;
        // Psi.A.Ac[0] = Psi.A.Ac[0]*std::pow(normSq,-0.5);
        // cout << "<Psi,Psi>=" << dot(Psi,Psi,DIR) << endl;
}
