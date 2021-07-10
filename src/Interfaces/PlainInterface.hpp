#ifndef PLAIN_INTERFACE_H_
#define PLAIN_INTERFACE_H_

#ifdef XPED_USE_ARRAY_TENSOR_LIB
#    define XPED_DEFAULT_TENSORLIB ArrayTensorLib
#elif defined(XPED_USE_EIGEN_TENSOR_LIB)
#    define XPED_DEFAULT_TENSORLIB EigenTensorLib
#elif defined(XPED_USE_CYCLOPS_TENSOR_LIB)
#    define XPED_DEFAULT_TENSORLIB CyclopsTensorLib
#endif

#ifdef XPED_USE_ARRAY_MATRIX_LIB
#    define XPED_DEFAULT_MATRIXLIB ArrayMatrixLib
#elif defined(XPED_USE_EIGEN_MATRIX_LIB)
#    define XPED_DEFAULT_MATRIXLIB EigenMatrixLib
#elif defined(XPED_USE_CYCLOPS_MATRIX_LIB)
#    define XPED_DEFAULT_MATRIXLIB CyclopsMatrixLib
#endif

#ifdef XPED_USE_ARRAY_VECTOR_LIB
#    define XPED_DEFAULT_VECTORLIB ArrayVectorLib
#elif defined(XPED_USE_EIGEN_VECTOR_LIB)
#    define XPED_DEFAULT_VECTORLIB EigenVectorLib
#elif defined(XPED_USE_CYCLOPS_VECTOR_LIB)
#    define XPED_DEFAULT_VECTORLIB CyclopsVectorLib
#endif

#define XPED_DEFAULT_PLAININTERFACE PlainInterface<XPED_DEFAULT_MATRIXLIB, XPED_DEFAULT_TENSORLIB, XPED_DEFAULT_VECTORLIB>

#include "Interfaces/MatrixInterface.hpp"
#include "Interfaces/TensorInterface.hpp"
#include "Interfaces/VectorInterface.hpp"

template <typename MatrixLibrary, typename TensorLibrary, typename VectorLibrary>
struct PlainInterface
{
    // **********************************
    // ******** Matrix Interface ********
    // **********************************

    // constructor (rows,cols)
    // constructor with pointer, rows, cols <-- Interchange data with TensorLib!

    // resize(rows, cols)
    // rows(), cols()

    // initialization
    // setZero
    // setRandom
    // setConstant
    // setIdentity
    // static Identity

    // block(r,c,dr,dc) l and r value
    // leftCols
    // topRows
    // head
    // diagonal

    // svd
    // qr

    // data() <-- Interchange data with TensorLib!

    // kroneckerProduct

    // print

    // operations +, - , *, coefficient-wise

    // adjoint

    // **********************************
    // ******** Tensor Interface ********
    // **********************************

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
};

#if defined XPED_USE_EIGEN_TENSOR_LIB && defined XPED_USE_EIGEN_MATRIX_LIB && defined XPED_USE_EIGEN_VECTOR_LIB
#    include "Interfaces/PlainInterface_Eigen_impl.hpp"
#    define XPED_CONST const
#elif defined XPED_USE_ARRAY_TENSOR_LIB && defined XPED_USE_EIGEN_MATRIX_LIB && defined XPED_USE_EIGEN_VECTOR_LIB
#    include "Interfaces/PlainInterface_Eigen_Array_impl.hpp"
#    define XPED_CONST const
#elif defined XPED_USE_CYCLOPS_TENSOR_LIB && defined XPED_USE_CYCLOPS_MATRIX_LIB && defined XPED_USE_CYCLOPS_VECTOR_LIB
#    include "Interfaces/PlainInterface_Cyclops_impl.hpp"
#    define XPED_CONST
#else
#    error "You specified an invalid combination of plain matrix library, plain tensor library and plain vector library."
#endif
#endif
