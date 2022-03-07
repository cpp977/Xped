#ifndef XPED_HELPER_H_
#define XPED_HELPER_H_

#include "Qbasis.hpp"

namespace Xped {

namespace internal {

template <std::size_t Rank1, std::size_t Rank2, typename Symmetry, typename AllocationPolicy>
std::pair<Qbasis<Symmetry, Rank2 + Rank1, AllocationPolicy>, std::array<Qbasis<Symmetry, 1, AllocationPolicy>, 0>>
build_FusionTree_Helper(const Qbasis<Symmetry, Rank2, AllocationPolicy>& coupled,
                        const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank1>& uncoupled);

template <std::size_t Rank, typename Symmetry, typename AllocationPolicy>
Qbasis<Symmetry, Rank, AllocationPolicy> build_FusionTree(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& uncoupled);

} // namespace internal

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/TensorHelper.cpp"
#endif
// template <typename MatrixType_>
// MatrixType_ zero_init()
// {
//     if constexpr(std::is_same<MatrixType_, Eigen::MatrixXd>::value) {
//         return Eigen::MatrixXd::Zero(1, 1);
//     } else if constexpr(std::is_same<MatrixType_, Eigen::SparseMatrix<double>>::value) {
//         Eigen::SparseMatrix<double> M(1, 1);
//         return M;
//     } else if constexpr(std::is_same<MatrixType_, Eigen::DiagonalMatrix<double, -1>>::value) {
//         Eigen::DiagonalMatrix<double, -1> M(1);
//         M.diagonal() << 0.;
//         return M;
//     }
// }

// template <typename MatrixType_>
// string print_matrix(const MatrixType_& mat)
// {
//     std::stringstream ss;
//     if constexpr(std::is_same<MatrixType_, Eigen::MatrixXd>::value) {
//         ss << mat;
//         return ss.str();
//     } else if constexpr(std::is_same<MatrixType_, Eigen::SparseMatrix<double>>::value) {
//         ss << mat;
//         return ss.str();
//     } else if constexpr(std::is_same<MatrixType_, Eigen::DiagonalMatrix<double, -1>>::value) {
//         ss << mat.toDenseMatrix();
//         return ss.str();
//     }
// }

#endif
