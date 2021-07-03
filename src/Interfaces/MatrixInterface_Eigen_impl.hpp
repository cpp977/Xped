
#ifndef MATRIX_INTERFACE_EIGEN_IMPL_H_
#define MATRIX_INTERFACE_EIGEN_IMPL_H_

#include "Eigen/Dense"
#include "unsupported/Eigen/KroneckerProduct"

#include "Util/Mpi.hpp"

#include "Interfaces/MatrixMultiplication.hpp"

template <>
struct MatrixInterface<EigenMatrixLib>
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
    static MType<Scalar> construct(const MIndextype& rows, const MIndextype& cols, util::mpi::XpedWorld world = util::mpi::Universe)
    {
        return MType<Scalar>(rows, cols);
    }

    template <typename Scalar>
    static MType<Scalar> construct_with_zero(const MIndextype& rows, const MIndextype& cols, util::mpi::XpedWorld world = util::mpi::Universe)
    {
        MType<Scalar> mat(rows, cols);
        mat.setZero();
        return mat;
    }

    template <typename Scalar>
    static void resize(MType<Scalar>& M, const MIndextype& new_rows, const MIndextype& new_cols)
    {
        M.resize(new_rows, new_cols);
    }

    // initialization
    template <typename Scalar>
    static void setZero(MType<Scalar>& M)
    {
        M.setZero();
    }

    template <typename Scalar>
    static void setRandom(MType<Scalar>& M)
    {
        M.setRandom();
    }

    template <typename Scalar>
    static void setIdentity(MType<Scalar>& M)
    {
        M.setIdentity();
    }

    template <typename Scalar>
    static void setConstant(MType<Scalar>& M, const Scalar& val)
    {
        M.setConstant(val);
    }

    template <typename Scalar>
    static MType<Scalar> Identity(const MIndextype& rows, const MIndextype& cols)
    {
        return Eigen::MatrixXd::Identity(rows, cols);
    }

    // shape
    template <typename Scalar>
    static MIndextype rows(const MType<Scalar>& M)
    {
        return M.rows();
    }

    template <typename Scalar>
    static MIndextype cols(const MType<Scalar>& M)
    {
        return M.cols();
    }

    template <typename Scalar>
    static Scalar getVal(const MType<Scalar>& M, const MIndextype& row, const MIndextype& col)
    {
        return M(row, col);
    }

    // reduction
    template <typename Scalar>
    static Scalar trace(const MType<Scalar>& M)
    {
        return M.trace();
    }

    // artithmetic
    template <typename Scalar>
    static MType<Scalar> kronecker_prod(const MType<Scalar>& M1, const MType<Scalar>& M2)
    {
        return MType<Scalar>(Eigen::kroneckerProduct(M1, M2));
    }

    template <typename Scalar>
    static MType<Scalar> prod(const MType<Scalar>& M1, const MType<Scalar>& M2)
    {
        return MType<Scalar>(M1 * M2);
    }

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres)
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
    static void optimal_prod_add(const Scalar& scale, const MatrixExpr1& M1, const MatrixExpr2& M2, const MatrixExpr3& M3, MatrixExprRes& Mres)
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
    static auto add(const MType<Scalar>& M1, const MType<Scalar>& M2)
    {
        return (M1 + M2);
    }

    template <typename Scalar>
    static auto difference(const MType<Scalar>& M1, const MType<Scalar>& M2)
    {
        return (M1 - M2);
    }

    template <typename Scalar>
    static void scale(MType<Scalar>& M, const Scalar& val)
    {
        M = (val * M);
    }

    template <typename Scalar>
    static auto adjoint(const MType<Scalar>& M)
    {
        return M.adjoint();
    }

    // block
    template <typename Scalar>
    static auto block(const MType<Scalar>& M, const MIndextype& row_off, const MIndextype& col_off, const MIndextype& rows, const MIndextype& cols)
    {
        return M.block(row_off, col_off, rows, cols);
    }

    template <typename Scalar>
    static void add_to_block(MType<Scalar>& M1,
                             const MIndextype& row_off,
                             const MIndextype& col_off,
                             const MIndextype& rows,
                             const MIndextype& cols,
                             const MType<Scalar>& M2)
    {
        M1.block(row_off, col_off, rows, cols) += M2;
    }

    template <typename Scalar>
    static void set_block(MType<Scalar>& M1,
                          const MIndextype& row_off,
                          const MIndextype& col_off,
                          const MIndextype& rows,
                          const MIndextype& cols,
                          const MType<Scalar>& M2)
    {
        M1.block(row_off, col_off, rows, cols) = M2;
    }

    template <typename Scalar>
    static std::string print(const MType<Scalar>& M)
    {
        std::stringstream ss;
        ss << M;
        return ss.str();
    }
};

#endif
