#include "Xped/Interfaces/MatrixInterface_Eigen_impl.hpp"

#include "unsupported/Eigen/KroneckerProduct"

#include "Xped/Util/Mpi.hpp"
#include "Xped/Core/ScalarTraits.hpp"

#include "Xped/Interfaces/MatrixMultiplication.hpp"

namespace Xped {

template <typename Scalar>
using MType = MatrixInterface::MType<Scalar>;

using MIndextype = MatrixInterface::MIndextype;

// constructors
template <typename Scalar>
MType<Scalar> MatrixInterface::construct(const MIndextype& rows, const MIndextype& cols, const mpi::XpedWorld&)
{
    return MType<Scalar>(rows, cols);
}

template <typename Scalar>
MType<Scalar> MatrixInterface::construct_with_zero(const MIndextype& rows, const MIndextype& cols, const mpi::XpedWorld&)
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
void MatrixInterface::setZero(Eigen::MatrixBase<Derived>& M)
{
    M.setZero();
}

template <typename Derived>
void MatrixInterface::setZero(Eigen::MatrixBase<Derived>&& M)
{
    M.setZero();
}

template <typename Derived>
void MatrixInterface::setRandom(Eigen::MatrixBase<Derived>& M, std::mt19937& engine)
{
    using Scalar = typename Derived::Scalar;
    std::uniform_real_distribution<typename ScalarTraits<Scalar>::Real> distribution(-1., 1.);
    M = M.NullaryExpr(M.rows(), M.cols(), [&engine, &distribution]() {
        if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
            return std::complex(distribution(engine), distribution(engine));
        } else {
            return distribution(engine);
        }
    });
    // M.setRandom();
}

template <typename Derived>
void MatrixInterface::setRandom(Eigen::MatrixBase<Derived>&& M, std::mt19937& engine)
{
    Eigen::MatrixBase<Derived>& tmp_M = M;
    return setRandom(tmp_M, engine);
}

template <typename Derived>
void MatrixInterface::setIdentity(Eigen::MatrixBase<Derived>& M)
{
    M.setIdentity();
}

template <typename Derived>
void MatrixInterface::setIdentity(Eigen::MatrixBase<Derived>&& M)
{
    M.setIdentity();
}

template <typename Derived>
void MatrixInterface::setConstant(Eigen::MatrixBase<Derived>& M, const typename Derived::Scalar& val)
{
    M.setConstant(val);
}

template <typename Derived>
void MatrixInterface::setConstant(Eigen::MatrixBase<Derived>&& M, const typename Derived::Scalar& val)
{
    M.setConstant(val);
}

template <typename Derived>
void MatrixInterface::setVal(Eigen::DenseBase<Derived>& M, const MIndextype& row, const MIndextype& col, typename Derived::Scalar val)
{
    M(row, col) = val;
}

template <typename Scalar>
MType<Scalar> MatrixInterface::Identity(const MIndextype& rows, const MIndextype& cols, const mpi::XpedWorld&)
{
    return Eigen::MatrixXd::Identity(rows, cols);
}

// shape
template <typename Derived>
MIndextype MatrixInterface::rows(const Eigen::DenseBase<Derived>& M)
{
    return M.rows();
}

template <typename Derived>
MIndextype MatrixInterface::cols(const Eigen::DenseBase<Derived>& M)
{
    return M.cols();
}

template <typename Derived>
typename Derived::Scalar MatrixInterface::getVal(const Eigen::DenseBase<Derived>& M, const MIndextype& row, const MIndextype& col)
{
    return M(row, col);
}

// reduction
template <typename Derived>
typename Derived::Scalar MatrixInterface::trace(const Eigen::MatrixBase<Derived>& M)
{
    return M.trace();
}

template <typename Derived>
typename Derived::RealScalar MatrixInterface::maxNorm(const Eigen::MatrixBase<Derived>& M)
{
    return M.template lpNorm<Eigen::Infinity>();
}

template <typename Derived>
typename Derived::RealScalar MatrixInterface::maxCoeff(const Eigen::MatrixBase<Derived>& M, MIndextype& maxrow, MIndextype& maxcol)
{
    return M.array().abs().maxCoeff(&maxrow, &maxcol);
}

// artithmetic
template <typename DerivedL, typename DerivedR>
MType<std::common_type_t<typename DerivedL::Scalar, typename DerivedR::Scalar>> MatrixInterface::kronecker_prod(const Eigen::MatrixBase<DerivedL>& M1,
                                                                                                                const Eigen::MatrixBase<DerivedR>& M2)
{
    return Eigen::kroneckerProduct(M1, M2);
}

template <typename DerivedL, typename DerivedR>
MType<std::common_type_t<typename DerivedL::Scalar, typename DerivedR::Scalar>> MatrixInterface::prod(const Eigen::MatrixBase<DerivedL>& M1,
                                                                                                      const Eigen::MatrixBase<DerivedR>& M2)
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
void MatrixInterface::scale(Eigen::MatrixBase<Derived>& M, const typename Derived::Scalar& val)
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
void MatrixInterface::add_to_block(Eigen::MatrixBase<Derived>&& M1,
                                   const MIndextype& row_off,
                                   const MIndextype& col_off,
                                   const MIndextype& rows,
                                   const MIndextype& cols,
                                   const Eigen::MatrixBase<Derived>& M2)
{
    M1.block(row_off, col_off, rows, cols) += M2;
}

template <typename Derived>
void MatrixInterface::set_block(Eigen::MatrixBase<Derived>&& M1,
                                const MIndextype& row_off,
                                const MIndextype& col_off,
                                const MIndextype& rows,
                                const MIndextype& cols,
                                const Eigen::MatrixBase<Derived>& M2)
{
    M1.block(row_off, col_off, rows, cols) = M2;
}

template <typename Derived>
std::pair<MType<typename Derived::Scalar>, MType<typename Derived::Scalar>> MatrixInterface::qr(const Eigen::MatrixBase<Derived>& M)
{
    Eigen::HouseholderQR<MType<typename Derived::Scalar>> Quirinus;
    Quirinus.compute(M);
    return std::make_pair(Quirinus.householderQ(), Quirinus.matrixQR().template triangularView<Eigen::Upper>());
}

template <typename Derived>
std::string MatrixInterface::print(const Eigen::DenseBase<Derived>& M)
{
    std::stringstream ss;
    ss << M;
    return ss.str();
}

} // namespace Xped

#if __has_include("MatrixInterface_Eigen_impl.gen.cpp")
#    include "MatrixInterface_Eigen_impl.gen.cpp"
#endif
