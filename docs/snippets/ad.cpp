#include "Xped/Core/ADTensor.hpp"

int main()
{
    using Symmetry = Xped::Sym::SU2<Xped::Sym::Fermion>;

    Xped::Qbasis<Symmetry, /*depth=*/1> B1;
    B1.setRandom(10);
    Xped::Qbasis<Symmetry, /*depth=*/1> B2;
    B2.setRandom(10);

    // This starts the autodiff procedure
    stan::math::nested_rev_autodiff nested;

    // Initialize a random rank-3 tensor for use with AD
    Xped::Tensor<double, /*Rank=*/2, /*CoRank=*/1, Symmetry, /*AD=*/true> T({{B1, B2}}, {{B1.combine(B2).forgetHistory()}});
    T.setRandom();
    std::cout << "T=\n" << T << std::endl;

    // Do some computation with T that produces a scalar output.
    // auto is important here, since the return type is stan::math::var_value<double> and not double
    auto res = ((7. * T) * (3. * T.adjoint())).norm();
    std::cout << "res=" << res.val() << std::endl;

    // This starts the backpropagation to compute the gradient of the input (in this case T)
    stan::math::grad(res.vi_);

    std::cout << "grad(T)=\n" << T.adj() << std::endl;
}
