#ifndef XPED_PEPS_HEISENBERG_HPP_
#define XPED_PEPS_HEISENBERG_HPP_

#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Symmetry>
class Heisenberg
{};
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

template <>
class Heisenberg<Sym::U1<Sym::SpinU1>>
{
    using Symmetry = Sym::U1<Sym::SpinU1>;

public:
    static Tensor<double, 2, 2, Symmetry> twoSiteHamiltonian(double Jxy = 1., double Jz = 1.)
    {
        Qbasis<Symmetry, 1> phys;
        phys.push_back({+1}, 1);
        phys.push_back({-1}, 1);

        Qbasis<Symmetry, 1> plus2;
        plus2.push_back({+2}, 1);
        auto fuse = Tensor<double, 2, 1, Symmetry>::Identity({{phys, phys}}, {{phys.combine(phys).forgetHistory()}});

        Tensor<double, 1, 1, Symmetry> Sz({{phys}}, {{phys}});
        Sz.setIdentity();
        Sz *= 0.5;
        Sz.block(1)(0, 0) *= -1;
        Tensor<double, 2, 1, Symmetry> Sp({{phys, plus2}}, {{phys}});
        Sp.setZero();
        Sp.block(0)(0, 0) = 1;
        auto Sm = Sp.adjoint().eval();
        Sz.print(std::cout);
        Sp.print(std::cout);
        Sm.print(std::cout);
        // auto SS = idxS.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(idxS.adjoint().eval());
        // auto ss = Sxid.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(Sxid.adjoint().eval());
        // auto SsxId = Sxs.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Id);
        // auto IdxSs = Id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sxs);

        // Tensor<double, 2, 2, Symmetry> out = J * ss + 0.25 * Jk * (SsxId + IdxSs) + I * SS;
        // return out;
    }
};

template <>
class Heisenberg<Sym::U0<>>
{
    using Symmetry = Sym::U0<>;

public:
    static Tensor<double, 2, 2, Symmetry> twoSiteHamiltonian(double Jxy = 1., double Jz = 1.)
    {
        Qbasis<Symmetry, 1> phys;
        phys.push_back({}, 2);

        auto fuse = Tensor<double, 2, 1, Symmetry>::Identity({{phys, phys}}, {{phys.combine(phys).forgetHistory()}});

        Tensor<double, 1, 1, Symmetry> Sz({{phys}}, {{phys}});
        Sz.setIdentity();
        Sz *= 0.5;
        Sz.block(0)(1, 1) *= -1;
        Tensor<double, 1, 1, Symmetry> Sp({{phys}}, {{phys}});
        Sp.setZero();
        Sp.block(0)(0, 1) = 1;
        auto Sm = Sp.adjoint().eval();
        // auto SS = idxS.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(idxS.adjoint().eval());
        // auto ss = Sxid.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(Sxid.adjoint().eval());
        // auto SsxId = Sxs.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Id);
        // auto IdxSs = Id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sxs);
        Tensor<double, 2, 2, Symmetry> out = Jz * Sz.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sz) +
                                             0.5 * Jxy * Sp.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sm) +
                                             0.5 * Jxy * Sm.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sp);
        return out;
    }
};

} // namespace Xped
#endif
