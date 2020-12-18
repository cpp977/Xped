#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <TextTable.h>

#include "macros.h"

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

int main(int argc, char* argv[])
{
        ArgParser args(argc, argv);
        typedef Sym::SU2<Sym::SpinSU2> Symmetry;
        typedef Symmetry::qType qType;
        std::vector<qType> ql(1); qType qr = Symmetry::qvacuum();
        Symmetry::reduceSilent(ql,qr);
        Qbasis<Symmetry,1> I, J, S;
        S.push_back({2},1);
        I.push_back({1},1);
        J = I.combine(S).forgetHistory();
        std::size_t L = args.get<std::size_t>("L",10);
        int Dloc = args.get<int>("Dloc",2);
        std::vector<Tensor<2,1,Symmetry> > Mps(L);
        for (std::size_t l=0; l<L; l++) {
                Mps[l] = Tensor<2,1,Symmetry>({{I,S}}, {J});
        }
        Tensor<2,1,Symmetry> A({{I,S}}, {J});
        A.setRandom();
        std::cout << A << std::endl;
        auto N = A.adjoint() * A;
        std::cout << N << std::endl;
}
