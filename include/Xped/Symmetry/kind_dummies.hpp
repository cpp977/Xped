#ifndef LABEL_DUMMIES_H
#define LABEL_DUMMIES_H

#include "functions.hpp"

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
