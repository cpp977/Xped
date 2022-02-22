#ifndef MATRIX_INTERFACE_EIGEN_IMPL_H_
#define MATRIX_INTERFACE_EIGEN_IMPL_H_

#include "Eigen/Dense"

#include "Xped/Util/Mpi.hpp"

namespace Xped {

struct MatrixInterface
{
    // typedefs
    template <typename Scalar>
    using MType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename Scalar>
    using cMType = const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    template <typename Scalar>
    using MapMType = Eigen::Map<MType<Scalar>>;
    template <typename Scalar>
    using cMapMType = Eigen::Map<cMType<Scalar>>;

    typedef Eigen::Index MIndextype;
    // constructors
    template <typename Scalar>
    static MType<Scalar> construct(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world = mpi::getUniverse());

    template <typename Scalar>
    static MType<Scalar> construct_with_zero(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world = mpi::getUniverse());

    template <typename Scalar>
    static void resize(MType<Scalar>& M, const MIndextype& new_rows, const MIndextype& new_cols);

    // initialization
    template <typename Scalar>
    static void setZero(MType<Scalar>& M);

    template <typename Scalar>
    static void setRandom(MType<Scalar>& M);

    template <typename Scalar>
    static void setIdentity(MType<Scalar>& M);

    template <typename Scalar>
    static void setConstant(MType<Scalar>& M, const Scalar& val);

    template <typename Scalar>
    static MType<Scalar> Identity(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world = mpi::getUniverse());

    // shape
    template <typename Scalar>
    static MIndextype rows(const MType<Scalar>& M);

    template <typename Scalar>
    static MIndextype cols(const MType<Scalar>& M);

    template <typename Scalar>
    static Scalar getVal(const MType<Scalar>& M, const MIndextype& row, const MIndextype& col);

    // reduction
    template <typename Scalar>
    static Scalar trace(const MType<Scalar>& M);

    // artithmetic
    template <typename Scalar>
    static MType<Scalar> kronecker_prod(const MType<Scalar>& M1, const MType<Scalar>& M2);

    template <typename Scalar>
    static MType<Scalar> prod(const MType<Scalar>& M1, const MType<Scalar>& M2);

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres);

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod_add(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres);

    template <typename Scalar>
    static const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar, Scalar>, const MType<Scalar>, const MType<Scalar>>
    add(const MType<Scalar>& M1, const MType<Scalar>& M2);

    template <typename Scalar>
    static const Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar, Scalar>, const MType<Scalar>, const MType<Scalar>>
    difference(const MType<Scalar>& M1, const MType<Scalar>& M2);

    template <typename Scalar>
    static void scale(MType<Scalar>& M, const Scalar& val);

    template <typename Scalar, typename Derived>
    static auto unaryFunc(const Eigen::MatrixBase<Derived>& M, const std::function<Scalar(Scalar)>& func)
    {
        return M.unaryExpr(func);
    }

    template <typename Scalar, typename Derived>
    static auto diagUnaryFunc(const Eigen::MatrixBase<Derived>& M, const std::function<Scalar(Scalar)>& func)
    {
        return M.diagonal().unaryExpr(func).asDiagonal();
    }

    template <typename Scalar>
    static typename Eigen::MatrixBase<MType<Scalar>>::AdjointReturnType adjoint(const MType<Scalar>& M);

    // block
    template <typename Scalar>
    static const Eigen::Block<const MType<Scalar>>
    block(const MType<Scalar>& M, const MIndextype& row_off, const MIndextype& col_off, const MIndextype& rows, const MIndextype& cols);

    template <typename Scalar>
    static void add_to_block(MType<Scalar>& M1,
                             const MIndextype& row_off,
                             const MIndextype& col_off,
                             const MIndextype& rows,
                             const MIndextype& cols,
                             const MType<Scalar>& M2);

    template <typename Scalar>
    static void set_block(MType<Scalar>& M1,
                          const MIndextype& row_off,
                          const MIndextype& col_off,
                          const MIndextype& rows,
                          const MIndextype& cols,
                          const MType<Scalar>& M2);

    template <typename Scalar>
    static std::string print(const MType<Scalar>& M);
};

} // namespace Xped
#ifndef XPED_COMPILED_LIB
#    include "Interfaces/MatrixInterface_Eigen_impl.cpp"
#endif

#endif
