#ifndef XPED_KONDO_NECKLACE_HPP_
#define XPED_KONDO_NECKLACE_HPP_

#include "Xped/Core/Tensor.hpp"
#include "Xped/Physics/SpinBase.hpp"

namespace Xped {

template <typename Symmetry>
class KondoNecklace
{
public:
    static Tensor<double, 2, 2, Symmetry>
    twoSiteHamiltonian(double Jkxy = 1., double Jkz = 1., double Jxy = 1., double Jz = 1., double Ixy = 0., double Iz = 0.)
    {

        SpinBase<Symmetry> B(2, 2);
        auto SzSz = tprod(B.Sz(0), B.Sz(0));
        auto SpSm = tprod(B.Sp(0), B.Sm(0));
        auto SmSp = tprod(B.Sm(0), B.Sp(0));

        auto szsz = tprod(B.Sz(1), B.Sz(1));
        auto spsm = tprod(B.Sp(1), B.Sm(1));
        auto smsp = tprod(B.Sm(1), B.Sp(1));

        auto IdxSzsz = tprod(B.Id(), B.Sz(0) * B.Sz(1));
        auto IdxSpsm = tprod(B.Id(), B.Sp(0) * B.Sm(1));
        auto IdxSmsp = tprod(B.Id(), B.Sm(0) * B.Sp(1));

        auto SzszxId = tprod(B.Sz(0) * B.Sz(1), B.Id());
        auto SpsmxId = tprod(B.Sp(0) * B.Sm(1), B.Id());
        auto SmspxId = tprod(B.Sm(0) * B.Sp(1), B.Id());

        Tensor<double, 2, 2, Symmetry> out =
            Jz * szsz + 0.5 * Jxy * spsm + 0.5 * Jxy * smsp +
            0.25 * (Jkz * (SzszxId + IdxSzsz) + 0.5 * Jkxy * (SpsmxId + IdxSpsm) + 0.5 * Jkxy * (SmspxId + IdxSmsp)) + Iz * SzSz + 0.5 * Ixy * SpSm +
            0.5 * Ixy * SmSp;

        return out;
    }
};

template <>
class KondoNecklace<Sym::SU2<Sym::SpinSU2>>
{
    using Symmetry = Sym::SU2<Sym::SpinSU2>;

public:
    static Tensor<double, 2, 2, Symmetry> twoSiteHamiltonian(double Jk, double J = 1., double I = 0.)
    {
        SpinBase<Symmetry> B(2, 2);
        auto SS = (std::sqrt(3.) * tprod(B.Sdag(0), B.S(0))).eval();

        auto ss = (std::sqrt(3.) * tprod(B.Sdag(1), B.S(1))).eval();

        auto IdxSs = tprod(B.Id(), std::sqrt(3.) * SiteOperator<double, Symmetry>::prod(B.Sdag(0), B.S(1), Symmetry::qvacuum()));

        auto SsxId = tprod(std::sqrt(3.) * SiteOperator<double, Symmetry>::prod(B.Sdag(0), B.S(1), Symmetry::qvacuum()), B.Id());

        Tensor<double, 2, 2, Symmetry> out = J * ss + 0.25 * Jk * (SsxId + IdxSs) + I * SS;
        return out;
    }
};

} // namespace Xped
#endif
