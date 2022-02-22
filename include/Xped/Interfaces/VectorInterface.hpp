#ifndef VECTORINTERFACE_H_
#define VECTORINTERFACE_H_

namespace Xped {

struct EigenVectorLib
{};

struct ArrayVectorLib
{};

struct CyclopsVectorLib
{};

#ifdef XPED_USE_ARRAY_VECTOR_LIB
#    define M_VECTORLIB ArrayVectorLib
#elif defined(XPED_USE_EIGEN_VECTOR_LIB)
#    define M_VECTORLIB EigenVectorLib
#elif defined(XPED_USE_CYCLOPS_VECTORLIB)
#    define M_VECTORLIB CyclopsVectorLib
#endif

// template <typename Library>
// struct VectorInterface
// {
// constructor (elems)
// constructor_with_zeros (elems)

// resize(elems)
// length()

// initialization
// setZero
// setRandom
// setConstant
// setIdentity

// sub(off,elems) l and r value
// print

// operations +, - , *, coefficient-wise
// };

} // namespace Xped

#if defined XPED_USE_EIGEN_VECTOR_LIB
#    include "Xped/Interfaces/VectorInterface_Eigen_impl.hpp"
#elif defined XPED_USE_ARRAY_VECTOR_LIB
#    include "Xped/Interfaces/VectorInterface_Array_impl.hpp"
#elif defined XPED_USE_CYCLOPS_VECTOR_LIB
#    include "Xped/Interfaces/VectorInterface_Cyclops_impl.hpp"
#else
#    error "You specified an unsupported plain vector library."
#endif

#endif
