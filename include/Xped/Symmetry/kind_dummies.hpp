#ifndef LABEL_DUMMIES_H
#define LABEL_DUMMIES_H

#include "functions.hpp"

namespace Xped::Sym {

struct SpinSU2
{
    static const KIND name = KIND::S;
};

struct AltSpinSU2
{
    static const KIND name = KIND::Salt;
};

struct SpinU1
{
    static const KIND name = KIND::M;
};

struct ChargeSU2
{
    static const KIND name = KIND::T;
};

struct ChargeU1
{
    static const KIND name = KIND::N;
};

struct ChargeUp
{
    static const KIND name = KIND::Nup;
};

struct ChargeDn
{
    static const KIND name = KIND::Ndn;
};

struct ChargeZ2
{
    static const KIND name = KIND::Z2;
};

} // namespace Xped::Sym
#endif // LABEL_DUMMIES_H
