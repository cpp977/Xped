#ifndef XPED_LABEL_DUMMIES_H
#define XPED_LABEL_DUMMIES_H

#include "functions.hpp"

// using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>

// using Symmetry = Xped::Sym::SU2<Xped::Fermion>

// using Symmetry = Xped::Sym::SU2<Xped::Spin>

// using Symmetry = Xped::Sym::U1<Xped::Fermion>

// using Symmetry = Xped::Sym::U1<Xped::Spin>

// using Symmetry = Xped::Sym::ZN<Xped::Fermion, 2>

namespace Xped {

enum class Kind
{
    Fermion,
    Boson,
    Spin
};

}

namespace Xped::Sym {

struct SpinSU2
{
    static constexpr KIND name = KIND::S;
};

struct AltSpinSU2
{
    static constexpr KIND name = KIND::Salt;
};

struct SpinU1
{
    static constexpr KIND name = KIND::M;
};

struct ChargeSU2
{
    static constexpr KIND name = KIND::T;
};

struct FChargeSU2
{
    static constexpr KIND name = KIND::FT;
};

struct ChargeU1
{
    static constexpr KIND name = KIND::N;
};

struct FChargeU1
{
    static constexpr KIND name = KIND::FN;
};

struct ChargeUp
{
    static constexpr KIND name = KIND::Nup;
};

struct ChargeDn
{
    static constexpr KIND name = KIND::Ndn;
};

struct ChargeZ2
{
    static constexpr KIND name = KIND::Z2;
};

} // namespace Xped::Sym
#endif // LABEL_DUMMIES_H
