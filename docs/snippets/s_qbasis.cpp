#include <type_traits>

#include "Xped/Core/Qbasis.hpp"

int main()
{
    using Symmetry = Xped::Sym::SU2<Xped::Kind::Spin>;

    Xped::Qbasis<Symmetry> B;
    // Add two states with spin 1/2 and one state with spin 3 into the basis.
    B.push_back({2}, 2); // One has to use D=2s+1
    B.push_back({7}, 1);

    // Print the basis (it has three multiplets with a total dimension of 11 states)
    Log::debug("{}", B.print()); // Basis(SU₂, dim=3[11]): q=1/2[2], q=3[1]

    // Qbasis objects can be combined into a product basis
    auto Bsq = B.combine(B);
    Log::debug("Squared: {}",
               Bsq.print()); // Squared: Basis(SU₂, dim=23[121]): q=0[5], q=1[5], q=2[1], q=5/2[4], q=3[1], q=7/2[4], q=4[1], q=5[1], q=6[1]
    static_assert(std::is_same_v<decltype(Bsq), Qbasis<Symmetry, 2>>); // Bsq has depth 2 because it is the result of a combination of two bases
    Log::debug("{}", Bsq.printTrees()); // Bsq contains also FusionTrees (13 in total) from the combination of both bases
    /* Here are the trees for the fusion quantum S=7/2
      2 Fusion trees for Q=7/2
     1/2     3
      \     /
       \   /
         μ
         |
         |
         7/2

     3       1/2
      \     /
       \   /
         μ
         |
         |
         7/2
     */

    // This would turn Bsq into an empty basis
    Bsq.clear();
}
