#ifndef MATRIX_INTERFACE_EIGEN_IMPL_H_
#define MATRIX_INTERFACE_EIGEN_IMPL_H_

#include "Eigen/Core"
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
    template <typename Derived>
    static void setZero(Eigen::MatrixBase<Derived>& M);

    template <typename Derived>
    static void setZero(Eigen::MatrixBase<Derived>&& M);

    template <typename Derived>
    static void setRandom(Eigen::MatrixBase<Derived>& M);

    template <typename Derived>
    static void setRandom(Eigen::MatrixBase<Derived>&& M);

    template <typename Derived>
    static void setIdentity(Eigen::MatrixBase<Derived>& M);

    template <typename Derived>
    static void setIdentity(Eigen::MatrixBase<Derived>&& M);

    template <typename Derived>
    static void setConstant(Eigen::MatrixBase<Derived>& M, const typename Derived::Scalar& val);

    template <typename Derived>
    static void setConstant(Eigen::MatrixBase<Derived>&& M, const typename Derived::Scalar& val);

    template <typename Scalar>
    static MType<Scalar> Identity(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world = mpi::getUniverse());

    // shape
    template <typename Derived>
    static MIndextype rows(const Eigen::DenseBase<Derived>& M);

    template <typename Derived>
    static MIndextype cols(const Eigen::DenseBase<Derived>& M);

    template <typename Derived>
    static typename Derived::Scalar getVal(const Eigen::DenseBase<Derived>& M, const MIndextype& row, const MIndextype& col);

    // raw data
    template <typename Scalar>
    static const Scalar* get_raw_data(const Eigen::Matrix<Scalar, -1, -1>& M)
    {
        return M.data();
    }

    template <typename Scalar>
    static Scalar* get_raw_data(Eigen::Matrix<Scalar, -1, -1>& M)
    {
        return M.data();
    }

    // reduction
    template <typename Derived>
    static typename Derived::Scalar trace(const Eigen::MatrixBase<Derived>& M);

    // artithmetic
    template <typename DerivedL, typename DerivedR>
    static MType<typename DerivedL::Scalar> kronecker_prod(const Eigen::MatrixBase<DerivedL>& M1, const Eigen::MatrixBase<DerivedR>& M2);

    template <typename DerivedL, typename DerivedR>
    static MType<typename DerivedL::Scalar> prod(const Eigen::MatrixBase<DerivedL>& M1, const Eigen::MatrixBase<DerivedR>& M2);

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres);

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod_add(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres);

    template <typename DerivedL, typename DerivedR>
    // static const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<typename Derived::Scalar, typename Derived::Scalar>,
    //                                   const Eigen::PlainObjectBase<Derived>,
    //                                   const Eigen::PlainObjectBase<Derived>>
    static auto add(const Eigen::MatrixBase<DerivedL>& M1, const Eigen::MatrixBase<DerivedR>& M2)
    {
        return (M1 + M2);
    }

    template <typename DerivedL, typename DerivedR>
    // static const Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<typename Derived::Scalar, typename Derived::Scalar>,
    //                                   const Eigen::PlainObjectBase<Derived>,
    //                                   const Eigen::PlainObjectBase<Derived>>
    static auto difference(const Eigen::MatrixBase<DerivedL>& M1, const Eigen::MatrixBase<DerivedR>& M2)
    {
        return (M1 - M2);
    }

    template <typename Derived>
    static void scale(Eigen::MatrixBase<Derived>& M, const typename Derived::Scalar& val);

    template <typename Scalar, typename Derived>
    static auto unaryFunc(const Eigen::MatrixBase<Derived>& M, const std::function<Scalar(Scalar)>& func)
    {
        return M.unaryExpr(func);
    }

    template <typename Scalar, typename Derived, typename OtherDerived>
    static auto binaryFunc(const Eigen::MatrixBase<Derived>& M_left,
                           const Eigen::MatrixBase<OtherDerived>& M_right,
                           const std::function<Scalar(Scalar, Scalar)>& func)
    {
        return M_left.binaryExpr(M_right, func);
    }

    template <typename Scalar, typename Derived>
    static auto diagUnaryFunc(const Eigen::MatrixBase<Derived>& M, const std::function<Scalar(Scalar)>& func)
    {
        return M.diagonal().unaryExpr(func).asDiagonal();
    }

    template <typename Derived>
    static auto adjoint(const Eigen::MatrixBase<Derived>& M)
    {
        return M.adjoint();
    }

    // block
    template <typename Derived>
    static auto
    block(const Eigen::MatrixBase<Derived>& M, const MIndextype& row_off, const MIndextype& col_off, const MIndextype& rows, const MIndextype& cols)
    {
        return M.block(row_off, col_off, rows, cols);
    }

    template <typename Derived>
    static void add_to_block(Eigen::MatrixBase<Derived>& M1,
                             const MIndextype& row_off,
                             const MIndextype& col_off,
                             const MIndextype& rows,
                             const MIndextype& cols,
                             const Eigen::MatrixBase<Derived>& M2);

    template <typename Derived>
    static void set_block(Eigen::MatrixBase<Derived>& M1,
                          const MIndextype& row_off,
                          const MIndextype& col_off,
                          const MIndextype& rows,
                          const MIndextype& cols,
                          const Eigen::MatrixBase<Derived>& M2);

    template <typename Derived>
    static void add_to_block(Eigen::MatrixBase<Derived>&& M1,
                             const MIndextype& row_off,
                             const MIndextype& col_off,
                             const MIndextype& rows,
                             const MIndextype& cols,
                             const Eigen::MatrixBase<Derived>& M2);

    template <typename Derived>
    static void set_block(Eigen::MatrixBase<Derived>&& M1,
                          const MIndextype& row_off,
                          const MIndextype& col_off,
                          const MIndextype& rows,
                          const MIndextype& cols,
                          const Eigen::MatrixBase<Derived>& M2);

    template <typename Derived>
    static std::pair<MType<typename Derived::Scalar>, MType<typename Derived::Scalar>> eigh(const Eigen::MatrixBase<Derived>& M);

    template <typename Derived>
    static std::string print(const Eigen::DenseBase<Derived>& M);
};

} // namespace Xped
#ifndef XPED_COMPILED_LIB
#    include "Interfaces/MatrixInterface_Eigen_impl.cpp"
#endif

#endif
