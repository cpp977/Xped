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

template <typename Scalar,
          typename Symmetry,
          std::size_t TRank,
          bool ALL_OUT_LEGS,
          bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts,
          typename OpScalar,
          bool HERMITIAN>
TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename OneSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env, OneSiteObservable<OpScalar, Symmetry, HERMITIAN>& op);

template <typename Scalar,
          typename Symmetry,
          std::size_t TRank,
          bool ALL_OUT_LEGS,
          bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts,
          typename OpScalar,
          bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op);

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/LinearAlgebra.cpp"
#endif

#endif
