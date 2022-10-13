#ifndef TENSORINTERFACE_HPP__
#define TENSORINTERFACE_HPP__

#include <string>

namespace Xped {

struct EigenTensorLib
{
    const std::string name() const { return "Eigen"; }
};

struct ArrayTensorLib
{
    const std::string name() const { return "ndarray"; }
};

struct CyclopsTensorLib
{
    const std::string name() const { return "CTF Tensor"; }
};

#ifdef XPED_USE_ARRAY_TENSOR_LIB
#    define M_TENSORLIB ArrayTensorLib
#elif defined(XPED_USE_EIGEN_TENSOR_LIB)
#    define M_TENSORLIB EigenTensorLib
#elif defined(XPED_USE_CYCLOPS_TENSOR_LIB)
#    define M_TENSORLIB CyclopsTensorLib
#endif

// template <typename Library>
// struct TensorInterface
// {
// constructor with dim array [x] [x]
// constructor with map [x] [x]

// initialization
// setZero [x] [x]
// setRandom [x] [x]
// setConstant [x] [x]

// contract [x] [x]
// shuffle [x] [x]

// lvalue methods
// shuffle ??? <-- not in Eigen
// reshape with dims [x] [x]
// slice [x] [x]

// tensorProd [x] [x]

// return dimensions [x] [x]
// };

} // namespace Xped

#include "seq/seq.h"

#if defined XPED_USE_EIGEN_TENSOR_LIB
#    define EIGEN_DENSEBASE_PLUGIN "Xped/Util/EigenPlugins.hpp"
#    include "Xped/Interfaces/TensorInterface_Eigen_impl.hpp"
#elif defined XPED_USE_ARRAY_TENSOR_LIB
#    include "Xped/Interfaces/TensorInterface_Array_impl.hpp"
#elif defined XPED_USE_CYCLOPS_TENSOR_LIB
#    include "Xped/Interfaces/TensorInterface_Cyclops_impl.hpp"
#else
#    error "You specified an unsupported plain tensor library."
#endif

#endif
