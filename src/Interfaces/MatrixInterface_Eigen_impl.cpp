#include "Xped/Interfaces/MatrixInterface.hpp"

#include "Xped/Interfaces/MatrixInterface_Eigen_impl.hpp"

#include "unsupported/Eigen/KroneckerProduct"

#include "Xped/Util/Mpi.hpp"

#include "Xped/Interfaces/MatrixMultiplication.hpp"

namespace Xped {

template <typename Scalar>
using MType = MatrixInterface<EigenMatrixLib>::MType<Scalar>;

using MIndextype = MatrixInterface<EigenMatrixLib>::MIndextype;

// constructors
template <typename Scalar>
MType<Scalar> MatrixInterface<EigenMatrixLib>::construct(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world)
{
    return MType<Scalar>(rows, cols);
}

template <typename Scalar>
MType<Scalar> MatrixInterface<EigenMatrixLib>::construct_with_zero(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world)
{
    MType<Scalar> mat(rows, cols);
    mat.setZero();
    return mat;
}

template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::resize(MType<Scalar>& M, const MIndextype& new_rows, const MIndextype& new_cols)
{
    M.resize(new_rows, new_cols);
}

// initialization
template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::setZero(MType<Scalar>& M)
{
    M.setZero();
}

template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::setRandom(MType<Scalar>& M)
{
    M.setRandom();
}

template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::setIdentity(MType<Scalar>& M)
{
    M.setIdentity();
}

template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::setConstant(MType<Scalar>& M, const Scalar& val)
{
    M.setConstant(val);
}

template <typename Scalar>
MType<Scalar> MatrixInterface<EigenMatrixLib>::Identity(const MIndextype& rows, const MIndextype& cols, mpi::XpedWorld& world)
{
    return Eigen::MatrixXd::Identity(rows, cols);
}

// shape
template <typename Scalar>
MIndextype MatrixInterface<EigenMatrixLib>::rows(const MType<Scalar>& M)
{
    return M.rows();
}

template <typename Scalar>
MIndextype MatrixInterface<EigenMatrixLib>::cols(const MType<Scalar>& M)
{
    return M.cols();
}

template <typename Scalar>
Scalar MatrixInterface<EigenMatrixLib>::getVal(const MType<Scalar>& M, const MIndextype& row, const MIndextype& col)
{
    return M(row, col);
}

// reduction
template <typename Scalar>
Scalar MatrixInterface<EigenMatrixLib>::trace(const MType<Scalar>& M)
{
    return M.trace();
}

// artithmetic
template <typename Scalar>
MType<Scalar> MatrixInterface<EigenMatrixLib>::kronecker_prod(const MType<Scalar>& M1, const MType<Scalar>& M2)
{
    return MType<Scalar>(Eigen::kroneckerProduct(M1, M2));
}

template <typename Scalar>
MType<Scalar> MatrixInterface<EigenMatrixLib>::prod(const MType<Scalar>& M1, const MType<Scalar>& M2)
{
    return MType<Scalar>(M1 * M2);
}

template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
void MatrixInterface<EigenMatrixLib>::optimal_prod(const Scalar& scale,
                                                   const MatrixExpr1& M1,
                                                   const MatrixExpr2& M2,
                                                   const MatrixExpr3& M3,
                                                   MatrixExprRes& Mres)
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
void MatrixInterface<EigenMatrixLib>::optimal_prod_add(const Scalar& scale,
                                                       const MatrixExpr1& M1,
                                                       const MatrixExpr2& M2,
                                                       const MatrixExpr3& M3,
                                                       MatrixExprRes& Mres)
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

template <typename Scalar>
const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar, Scalar>, const MType<Scalar>, const MType<Scalar>>
MatrixInterface<EigenMatrixLib>::add(const MType<Scalar>& M1, const MType<Scalar>& M2)
{
    return (M1 + M2);
}

template <typename Scalar>
const Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar, Scalar>, const MType<Scalar>, const MType<Scalar>>
MatrixInterface<EigenMatrixLib>::difference(const MType<Scalar>& M1, const MType<Scalar>& M2)
{
    return (M1 - M2);
}

template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::scale(MType<Scalar>& M, const Scalar& val)
{
    M = (val * M);
}

template <typename Scalar>
typename Eigen::MatrixBase<MType<Scalar>>::AdjointReturnType MatrixInterface<EigenMatrixLib>::adjoint(const MType<Scalar>& M)
{
    return M.adjoint();
}

// block
template <typename Scalar>
const Eigen::Block<const MType<Scalar>> MatrixInterface<EigenMatrixLib>::block(const MType<Scalar>& M,
                                                                               const MIndextype& row_off,
                                                                               const MIndextype& col_off,
                                                                               const MIndextype& rows,
                                                                               const MIndextype& cols)
{
    return M.block(row_off, col_off, rows, cols);
}

template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::add_to_block(MType<Scalar>& M1,
                                                   const MIndextype& row_off,
                                                   const MIndextype& col_off,
                                                   const MIndextype& rows,
                                                   const MIndextype& cols,
                                                   const MType<Scalar>& M2)
{
    M1.block(row_off, col_off, rows, cols) += M2;
}

template <typename Scalar>
void MatrixInterface<EigenMatrixLib>::set_block(MType<Scalar>& M1,
                                                const MIndextype& row_off,
                                                const MIndextype& col_off,
                                                const MIndextype& rows,
                                                const MIndextype& cols,
                                                const MType<Scalar>& M2)
{
    M1.block(row_off, col_off, rows, cols) = M2;
}

template <typename Scalar>
std::string MatrixInterface<EigenMatrixLib>::print(const MType<Scalar>& M)
{
    std::stringstream ss;
    ss << M;
    return ss.str();
}

} // namespace Xped
