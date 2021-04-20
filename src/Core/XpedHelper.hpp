#ifndef XPED_HELPER_H_
#define XPED_HELPER_H_

#include "FusionTree.hpp"
#include "Qbasis.hpp"

namespace util {

template <std::size_t Rank1, std::size_t Rank2, typename Symmetry>
std::pair<Qbasis<Symmetry, Rank2 + Rank1>, std::array<Qbasis<Symmetry, 1>, 0>>
build_FusionTree_Helper(const Qbasis<Symmetry, Rank2>& coupled, const std::array<Qbasis<Symmetry, 1>, Rank1>& uncoupled)
{
    if constexpr(Rank1 == 0) {
        return std::make_pair(coupled, uncoupled);
    } else if constexpr(Rank1 == 1) {
        std::array<Qbasis<Symmetry, 1>, 0> trivial;
        return std::make_pair(coupled.combine(uncoupled[0]), trivial);
    } else {
        std::array<Qbasis<Symmetry, 1>, Rank1 - 1> new_uncoupled;
        std::copy(uncoupled.begin() + 1, uncoupled.end(), new_uncoupled.begin());
        return build_FusionTree_Helper(coupled.combine(uncoupled[0]), new_uncoupled);
    }
}

template <std::size_t Rank, typename Symmetry>
Qbasis<Symmetry, Rank> build_FusionTree(const std::array<Qbasis<Symmetry, 1>, Rank>& uncoupled)
{
    if constexpr(Rank == 0) {
        Qbasis<Symmetry, 0> tmp;
        tmp.push_back(Symmetry::qvacuum(), 1);
        return tmp;
    } else {
        std::array<Qbasis<Symmetry, 1>, Rank - 1> basis_domain_shrinked;
        std::copy(uncoupled.begin() + 1, uncoupled.end(), basis_domain_shrinked.begin());
        auto [domain_, discard] = util::build_FusionTree_Helper(uncoupled[0], basis_domain_shrinked);
        return domain_;
    }
}

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
} // namespace util

#endif
