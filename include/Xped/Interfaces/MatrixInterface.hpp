#ifndef MATRIXINTERFACE_H_
#define MATRIXINTERFACE_H_

namespace Xped {

struct EigenMatrixLib
{};

struct ArrayMatrixLib
{};

struct CyclopsMatrixLib
{};

#ifdef XPED_USE_ARRAY_MATRIX_LIB
#    define M_MATRIXLIB ArrayMatrixLib
#elif defined(XPED_USE_EIGEN_MATRIX_LIB)
#    define M_MATRIXLIB EigenMatrixLib
#elif defined(XPED_USE_CYCLOPS_MATRIX_LIB)
#    define M_MATRIXLIB CyclopsMatrixLib
#endif

template <typename Library>
struct MatrixInterface
{
    // constructor (rows,cols)
    // constructor_with_zeros (rows,cols)
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
};

} // namespace Xped

#if defined XPED_USE_EIGEN_MATRIX_LIB
#    include "Xped/Interfaces/MatrixInterface_Eigen_impl.hpp"
#elif defined XPED_USE_ARRAY_MATRIX_LIB
#    include "Xped/Interfaces/MatrixInterface_Array_impl.hpp"
#elif defined XPED_USE_CYCLOPS_MATRIX_LIB
#    include "Xped/Interfaces/MatrixInterface_Cyclops_impl.hpp"
#else
#    error "You specified an unsupported plain matrix library."
#endif

#endif
