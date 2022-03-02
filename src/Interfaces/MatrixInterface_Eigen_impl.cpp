#include "Xped/Interfaces/MatrixInterface_Eigen_impl.hpp"

#include "unsupported/Eigen/KroneckerProduct"

#include "Xped/Util/Mpi.hpp"

#include "Xped/Interfaces/MatrixMultiplication.hpp"

namespace Xped {

template <typename Scalar>
using MType = MatrixInterface::MType<Scalar>;

using MIndextype = MatrixInterface::MIndextype;

// constructors
template <typename Scalar>
MType<Scalar> MatrixInterface::construct(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world)
{
    return MType<Scalar>(rows, cols);
}

template <typename Scalar>
MType<Scalar> MatrixInterface::construct_with_zero(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world)
{
    MType<Scalar> mat(rows, cols);
    mat.setZero();
    return mat;
}

template <typename Scalar>
void MatrixInterface::resize(MType<Scalar>& M, const MIndextype& new_rows, const MIndextype& new_cols)
{
    M.resize(new_rows, new_cols);
}

// initialization
template <typename Derived>
void MatrixInterface::setZero(Eigen::PlainObjectBase<Derived>& M)
{
    M.setZero();
}

template <typename Derived>
void MatrixInterface::setRandom(Eigen::PlainObjectBase<Derived>& M)
{
    M.setRandom();
}

template <typename Derived>
void MatrixInterface::setIdentity(Eigen::PlainObjectBase<Derived>& M)
{
    M.setIdentity();
}

template <typename Derived>
void MatrixInterface::setConstant(Eigen::PlainObjectBase<Derived>& M, const typename Derived::Scalar& val)
{
    M.setConstant(val);
}

template <typename Scalar>
MType<Scalar> MatrixInterface::Identity(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world)
{
    return Eigen::MatrixXd::Identity(rows, cols);
}

// shape
template <typename Derived>
MIndextype MatrixInterface::rows(const Eigen::PlainObjectBase<Derived>& M)
{
    return M.rows();
}

template <typename Derived>
MIndextype MatrixInterface::cols(const Eigen::PlainObjectBase<Derived>& M)
{
    return M.cols();
}

template <typename Derived>
typename Derived::Scalar MatrixInterface::getVal(const Eigen::PlainObjectBase<Derived>& M, const MIndextype& row, const MIndextype& col)
{
    return M(row, col);
}

// reduction
template <typename Derived>
typename Derived::Scalar MatrixInterface::trace(const Eigen::PlainObjectBase<Derived>& M)
{
    return M.trace();
}

// artithmetic
template <typename DerivedL, typename DerivedR>
MType<typename DerivedL::Scalar> MatrixInterface::kronecker_prod(const Eigen::MatrixBase<DerivedL>& M1, const Eigen::MatrixBase<DerivedR>& M2)
{
    return Eigen::kroneckerProduct(M1, M2);
}

template <typename DerivedL, typename DerivedR>
MType<typename DerivedL::Scalar> MatrixInterface::prod(const Eigen::MatrixBase<DerivedL>& M1, const Eigen::MatrixBase<DerivedR>& M2)
{
    return M1 * M2;
}

template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
void MatrixInterface::optimal_prod(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres)
{
    std::vector<std::size_t> cost(2);
    cost = internal::mult_cost(std::array<MIndextype, 2>{M1.rows(), M1.cols()},
                               std::array<MIndextype, 2>{M2.rows(), M2.cols()},
                               std::array<MIndextype, 2>{M3.rows(), M3.cols()});
    std::size_t opt_mult = std::min_element(cost.begin(), cost.end()) - cost.begin();

    if(opt_mult == 0) {
        MType<Scalar> Mtmp = M1 * M2;
        Mres.noalias() = scale * Mtmp * M3;
    } else if(opt_mult == 1) {
        MType<Scalar> Mtmp = M2 * M3;
        Mres.noalias() = scale * M1 * Mtmp;
    }
}

template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
void MatrixInterface::optimal_prod_add(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres)
{
    std::vector<std::size_t> cost(2);
    cost = internal::mult_cost(std::array<MIndextype, 2>{M1.rows(), M1.cols()},
                               std::array<MIndextype, 2>{M2.rows(), M2.cols()},
                               std::array<MIndextype, 2>{M3.rows(), M3.cols()});
    std::size_t opt_mult = std::min_element(cost.begin(), cost.end()) - cost.begin();

    if(opt_mult == 0) {
        MType<Scalar> Mtmp = M1 * M2;
        Mres.noalias() += scale * Mtmp * M3;
    } else if(opt_mult == 1) {
        MType<Scalar> Mtmp = M2 * M3;
        Mres.noalias() += scale * M1 * Mtmp;
    }
}

template <typename Derived>
// const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<typename Derived::Scalar, typename Derived::Scalar>,
//                            const Eigen::PlainObjectBase<Derived>,
//                            const Eigen::PlainObjectBase<Derived>>
auto MatrixInterface::add(const Eigen::PlainObjectBase<Derived>& M1, const Eigen::PlainObjectBase<Derived>& M2)
{
    return (M1 + M2);
}

template <typename Derived>
// const Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<typename Derived::Scalar, typename Derived::Scalar>,
//                            const Eigen::PlainObjectBase<Derived>,
//                            const Eigen::PlainObjectBase<Derived>>
auto MatrixInterface::difference(const Eigen::PlainObjectBase<Derived>& M1, const Eigen::PlainObjectBase<Derived>& M2)
{
    return (M1 - M2);
}

template <typename Derived>
void MatrixInterface::scale(Eigen::PlainObjectBase<Derived>& M, const typename Derived::Scalar& val)
{
    M = (val * M);
}

// block
template <typename Derived>
void MatrixInterface::add_to_block(Eigen::MatrixBase<Derived>& M1,
                                   const MIndextype& row_off,
                                   const MIndextype& col_off,
                                   const MIndextype& rows,
                                   const MIndextype& cols,
                                   const Eigen::MatrixBase<Derived>& M2)
{
    M1.block(row_off, col_off, rows, cols) += M2;
}

template <typename Derived>
void MatrixInterface::set_block(Eigen::MatrixBase<Derived>& M1,
                                const MIndextype& row_off,
                                const MIndextype& col_off,
                                const MIndextype& rows,
                                const MIndextype& cols,
                                const Eigen::MatrixBase<Derived>& M2)
{
    M1.block(row_off, col_off, rows, cols) = M2;
}

template <typename Derived>
std::string MatrixInterface::print(const Eigen::PlainObjectBase<Derived>& M)
{
    std::stringstream ss;
    ss << M;
    return ss.str();
}

} // namespace Xped
