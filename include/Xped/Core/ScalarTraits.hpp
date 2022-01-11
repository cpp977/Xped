#ifndef SCALAR_TRAITS_H_
#define SCALAR_TRAITS_H_

#include <complex>

namespace Xped {

template <typename Scalar>
struct ScalarTraits
{};

template <>
struct ScalarTraits<double>
{
    typedef double Real;
};

template <typename RealScalar_>
struct ScalarTraits<std::complex<RealScalar_>>
{
    typedef RealScalar_ Real;
};

} // namespace Xped
#endif
