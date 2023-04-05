#ifndef XPED_PEPS_LINEAR_ALGEBRA_HPP_
#define XPED_PEPS_LINEAR_ALGEBRA_HPP_

#include <tuple>

#include "Xped/Util/Macros.hpp"

#include "Xped/Core/Tensor.hpp"

#include "Xped/AD/ADTensor.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
std::pair<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename ScalarTraits<Scalar>::Real>>,
          TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename ScalarTraits<Scalar>::Real>>>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ENABLE_AD, CPOpts>& env, XPED_CONST Tensor<Scalar, 2, 2, Symmetry, false>& op);

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename ScalarTraits<Scalar>::Real>>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ENABLE_AD, CPOpts>& env, OneSiteObservable<Symmetry>& op);

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename ScalarTraits<Scalar>::Real>>, 4>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ENABLE_AD, CPOpts>& env, TwoSiteObservable<Symmetry>& op);

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/LinearAlgebra.cpp"
#endif

#endif
