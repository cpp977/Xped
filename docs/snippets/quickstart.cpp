#include <type_traits>

#include "Xped/Core/Tensor.hpp"

int main()
{
    // Make an alias for the symmetry. In this case a U1 symmetry for a spin.
    using Symmetry = Xped::Sym::U1<Xped::Sym::Spin>;

    // Initialize random basis objects which will be used to initialize tensors
    Xped::Qbasis<Symmetry, /*depth=*/1> B1;
    B1.setRandom(10);
    Xped::Qbasis<Symmetry, /*depth=*/1> B2;
    B2.setRandom(10);

    // Initialize a random rank-3 tensor with two incoming and one outgoing leg with double precision numbers constrained by the symmetry declared
    // above
    Xped::Tensor<double, /*Rank=*/2, /*CoRank=*/1, Symmetry> T({{B1, B2}}, {{B1.combine(B2).forgetHistory()}});
    T.setRandom();

    // Unary and binary coefficientwise operations:
    auto S = 3. * T; // this is an expression which is lazely evaluated
    Xped::Tensor<double, 2, 1, Symmetry> Q = S.eval() + T; //.eval() can be used to force evaluation and assignment to tensor also forces evaluation

    // makes a truncated singular value decomposition between domain and codomain keeping at most 50 singular values and discarding all singular
    // values less than 1.e-10:
    auto [U, Sigma, Vdag] = S.tSVD(50, 1.e-10);
    static_assert(std::is_same_v<decltype(U), Xped::Tensor<double, 2, 1, Symmetry>>);
    static_assert(std::is_same_v<decltype(S), Xped::Tensor<double, 1, 1, Symmetry>>);
    static_assert(std::is_same_v<decltype(Vdag), Xped::Tensor<double, 1, 1, Symmetry>>);

    // Permutation of the indices. shift specifies how much the domain is decreasing (and the codomain is increasing). The following indices
    // describes the permutation.
    auto X = T.permute</*shift1=*/+1, 2, 1, 0>(); // this gets evaluated immediately
    // This is an SVD with a different partitioning of the legs
    auto [U_, Sigma_, Vdag_] = X.tSVD(50, 1.e-10);
    static_assert(std::is_same_v<decltype(U_), Xped::Tensor<double, 1, 1, Symmetry>>);
    static_assert(std::is_same_v<decltype(S_), Xped::Tensor<double, 1, 1, Symmetry>>);
    static_assert(std::is_same_v<decltype(Vdag_), Xped::Tensor<double, 1, 2, Symmetry>>);

    // Multiplication of tensors is the contraction over the matching codomain/domain of the two tensors
    auto prod = T * T.adjoint(); // this gets also evaluated immediately

    // Arbitrary contractions are also possible:
    auto res = T.contract<std::array{-1, -2, 1}, std::array{1, -4, -3}, /*ResRank=*/2>(X);
    static_assert(std::is_same_v<decltype(res), Xped::Tensor<double, 2, 2, Symmetry>>);
}
