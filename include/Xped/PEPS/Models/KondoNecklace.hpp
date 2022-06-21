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
        Qbasis<Symmetry, 1> phys;
        phys.push_back({2}, 1);
        Qbasis<Symmetry, 1> triplet;
        triplet.push_back({3}, 1);
        auto fuse = Tensor<double, 2, 1, Symmetry>::Identity({{phys, phys}}, {{phys.combine(phys).forgetHistory()}});

        Tensor<double, 2, 1, Symmetry> S({{phys, triplet}}, {{phys}});
        S.setConstant(std::sqrt(0.75));
        Tensor<double, 1, 1, Symmetry> id({{phys}}, {{phys}});
        id.setIdentity();
        auto idxS = id.template contract<std::array{-1, -4}, std::array{-2, -3, -5}, 3>(S)
                        .template contract<std::array{1, 2, -2, -3, -4}, std::array{-1, 1, 2}, 2>(fuse.adjoint().eval())
                        .template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3}, 2>(fuse);

        auto Sxid = S.template contract<std::array{-1, -3, -4}, std::array{-2, -5}, 3>(id)
                        .template contract<std::array{1, 2, -2, -3, -4}, std::array{-1, 1, 2}, 2>(fuse.adjoint().eval())
                        .template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3}, 2>(fuse);

        auto Sxs = S.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(S.adjoint().eval())
                       .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 2>(fuse.adjoint().eval())
                       .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);
        Tensor<double, 1, 1, Symmetry> Id({{phys.combine(phys).forgetHistory()}}, {{phys.combine(phys).forgetHistory()}});
        Id.setIdentity();

        auto SS = idxS.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(idxS.adjoint().eval());
        auto ss = Sxid.template contract<std::array{-1, 1, -3}, std::array{-2, -4, 1}, 2>(Sxid.adjoint().eval());
        auto SsxId = Sxs.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Id);
        auto IdxSs = Id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sxs);

        Tensor<double, 2, 2, Symmetry> out = J * ss + 0.25 * Jk * (SsxId + IdxSs) + I * SS;
        return out;
    }
};

// template <>
// class KondoNecklace<Sym::U1<Sym::SpinU1, double>>
// {
//     using Symmetry = Sym::U0<>;

// public:
//     static Tensor<double, 2, 2, Symmetry>
//     twoSiteHamiltonian(double Jkxy = 1., double Jkz = 1., double Jxy = 1., double Jz = 1., double Ixy = 0., double Iz = 0.)
//     {

//         SpinBase<Symmetry> B(2, 2);
//         auto SzSz = tprod(B.Sz(0), B.Sz(0));
//         auto SpSm = tprod(B.Sp(0), B.Sm(0));
//         auto SmSp = tprod(B.Sm(0), B.Sp(0));

//         auto szsz = tprod(B.Sz(1), B.Sz(1));
//         auto spsm = tprod(B.Sp(1), B.Sm(1));
//         auto smsp = tprod(B.Sm(1), B.Sp(1));

//         auto IdxSzsz = tprod(B.Id(), B.Sz(0) * B.Sz(1));
//         auto IdxSpsm = tprod(B.Id(), B.Sp(0) * B.Sm(1));
//         auto IdxSmsp = tprod(B.Id(), B.Sm(0) * B.Sp(1));

//         auto SzszxId = tprod(B.Sz(0) * B.Sz(1), B.Id());
//         auto SpsmxId = tprod(B.Sp(0) * B.Sm(1), B.Id());
//         auto SmspxId = tprod(B.Sm(0) * B.Sp(1), B.Id());

//         Tensor<double, 2, 2, Symmetry> out =
//             Jz * szsz + 0.5 * Jxy * spsm + 0.5 * Jxy * smsp +
//             0.25 * (Jkz * (SzszxId + IdxSzsz) + 0.5 * Jkxy * (SpsmxId + IdxSpsm) + 0.5 * Jkxy * (SmspxId + IdxSmsp)) + Iz * SzSz + 0.5 * Ixy * SpSm
//             + 0.5 * Ixy * SmSp;

//         return out;
//     }
// };

// template <>
// class KondoNecklace<Sym::U0<>>
// {
//     using Symmetry = Sym::U0<>;

// public:
//     static Tensor<double, 2, 2, Symmetry>
//     twoSiteHamiltonian(double Jkxy = 1., double Jkz = 1., double Jxy = 1., double Jz = 1., double Ixy = 0., double Iz = 0.)
//     {
//         Qbasis<Symmetry, 1> phys;
//         phys.push_back({}, 2);

//         auto fuse = Tensor<double, 2, 1, Symmetry>::Identity({{phys, phys}}, {{phys.combine(phys).forgetHistory()}});

//         Tensor<double, 1, 1, Symmetry> id({{phys}}, {{phys}});
//         id.setIdentity();

//         Tensor<double, 1, 1, Symmetry> Sz({{phys}}, {{phys}});
//         Sz.setIdentity();
//         Sz *= 0.5;
//         Sz.block(0)(1, 1) *= -1;
//         Tensor<double, 1, 1, Symmetry> Sp({{phys}}, {{phys}});
//         Sp.setZero();
//         Sp.block(0)(0, 1) = 1;
//         auto Sm = Sp.adjoint().eval();

//         auto idxSz = id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sz)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 1>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto Szxid = Sz.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(id)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 1>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto idxSp = id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sp)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 1>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto Spxid = Sp.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(id)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 1>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto idxSm = id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sm)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 1>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto Smxid = Sm.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(id)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 1>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto Szxsz = Sz.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sz)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 2>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto Spxsm = Sp.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sm)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 2>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);
//         auto Smxsp = Sm.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Sp)
//                          .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 2>(fuse.adjoint().eval())
//                          .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1>(fuse);

//         auto SzSz = idxSz.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(idxSz);
//         auto SpSm = idxSp.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(idxSm);
//         auto SmSp = idxSm.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(idxSp);

//         auto szsz = Szxid.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Szxid);
//         auto spsm = Spxid.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Smxid);
//         auto smsp = Smxid.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Spxid);

//         Tensor<double, 1, 1, Symmetry> Id({{phys.combine(phys).forgetHistory()}}, {{phys.combine(phys).forgetHistory()}});
//         Id.setIdentity();

//         auto SzszxId = Szxsz.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Id);
//         auto SpsmxId = Spxsm.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Id);
//         auto SmspxId = Smxsp.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Id);

//         auto IdxSzsz = Id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Szxsz);
//         auto IdxSpsm = Id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Spxsm);
//         auto IdxSmsp = Id.template contract<std::array{-1, -3}, std::array{-2, -4}, 2>(Smxsp);

//         Tensor<double, 2, 2, Symmetry> out =
//             Jz * szsz + 0.5 * Jxy * spsm + 0.5 * Jxy * smsp +
//             0.25 * (Jkz * (SzszxId + IdxSzsz) + 0.5 * Jkxy * (SpsmxId + IdxSpsm) + 0.5 * Jkxy * (SmspxId + IdxSmsp)) + Iz * SzSz + 0.5 * Ixy * SpSm
//             + 0.5 * Ixy * SmSp;

//         return out;
//     }
// };

} // namespace Xped
#endif
