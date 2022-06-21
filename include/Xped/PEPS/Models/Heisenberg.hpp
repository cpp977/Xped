#ifndef XPED_PEPS_HEISENBERG_HPP_
#define XPED_PEPS_HEISENBERG_HPP_

#include "Xped/Core/Tensor.hpp"

#include "Xped/Physics/SpinBase.hpp"

namespace Xped {

template <typename Symmetry>
class Heisenberg
{
public:
    static Tensor<double, 2, 2, Symmetry> twoSiteHamiltonian(double Jxy = 1., double Jz = 1.)
    {
        SpinBase<Symmetry> B(1, 2);
        Tensor<double, 2, 2, Symmetry> out = Jz * tprod(B.Sz(), B.Sz()) + 0.5 * Jxy * (tprod(B.Sp(), B.Sm()) + tprod(B.Sm(), B.Sp()));
        return out;
    }
};
// gs energy: E/N = -0.66944

template <>
class Heisenberg<Sym::SU2<Sym::SpinSU2>>
{
    using Symmetry = Sym::SU2<Sym::SpinSU2>;

public:
    static Tensor<double, 2, 2, Symmetry> twoSiteHamiltonian(double J = 1.)
    {
        Qbasis<Symmetry, 1> phys;
        phys.push_back({2}, 1);
        Qbasis<Symmetry, 1> triplet;
        triplet.push_back({3}, 1);
        auto fuse = Tensor<double, 2, 1, Symmetry>::Identity({{phys, phys}}, {{phys.combine(phys).forgetHistory()}});

        Tensor<double, 2, 1, Symmetry> S({{phys, triplet}}, {{phys}});
        S.setConstant(std::sqrt(0.75));

        Tensor<double, 2, 2, Symmetry> out = J * S.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(S.adjoint().eval());
        return out;
    }
};

} // namespace Xped
#endif
