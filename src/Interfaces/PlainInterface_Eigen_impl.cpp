#include "Xped/Interfaces/PlainInterface_Eigen_impl.hpp"

#include <Eigen/SVD>

namespace Xped {

using Indextype = Eigen::Index;

template <typename Scalar>
using MType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template <typename Scalar>
using cMType = const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar, std::size_t Rank>
using TType = Eigen::Tensor<Scalar, Rank>;

template <typename Scalar>
using VType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar, std::size_t Rank>
void PlainInterface::set_block_from_tensor(MType<Scalar>& M,
                                           const Indextype& row,
                                           const Indextype& col,
                                           const Indextype& rows,
                                           const Indextype& cols,
                                           const Scalar& scale,
                                           const TType<Scalar, Rank>& T)
{
    M.block(row, col, rows, cols) = scale * Eigen::Map<cMType<Scalar>>(T.data(), rows, cols);
}

template <typename Scalar, std::size_t Rank>
void PlainInterface::add_to_block_from_tensor(MType<Scalar>& M,
                                              const Indextype& row,
                                              const Indextype& col,
                                              const Indextype& rows,
                                              const Indextype& cols,
                                              const Scalar& scale,
                                              const TType<Scalar, Rank>& T)
{
    M.block(row, col, rows, cols) += scale * Eigen::Map<cMType<Scalar>>(T.data(), rows, cols);
}

template <typename Scalar, std::size_t Rank>
TType<Scalar, Rank> PlainInterface::tensor_from_matrix_block(const MType<Scalar>& M,
                                                             const Indextype& row,
                                                             const Indextype& col,
                                                             const Indextype& rows,
                                                             const Indextype& cols,
                                                             const std::array<Indextype, Rank>& dims)
{
    MType<Scalar> submatrix = M.block(row, col, rows, cols);
    cMapTType<Scalar, Rank> tensorview = cMap(submatrix.data(), dims);
    return construct<Scalar, Rank>(tensorview);
}

template <typename Scalar>
void PlainInterface::diagonal_head_matrix_to_vector(VType<Scalar>& V, const MType<Scalar>& M, const Indextype& n_elems)
{
    V = M.diagonal().head(n_elems);
}

template <typename Scalar>
std::tuple<MType<Scalar>, VType<Scalar>, MType<Scalar>> PlainInterface::svd(const MType<Scalar>& M)
{
#ifdef XPED_DONT_USE_BDCSVD
    Eigen::JacobiSVD<MType<Scalar>> Jack; // standard SVD
#else
    Eigen::BDCSVD<MType<Scalar>> Jack; // "Divide and conquer" SVD (only available in Eigen)
#endif

    Jack.compute(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return std::make_tuple(Jack.matrixU(), Jack.singularValues(), Jack.matrixV().adjoint());
}

template <typename Scalar>
MType<Scalar> PlainInterface::vec_to_diagmat(const VType<Scalar>& V)
{
    return V.matrix().asDiagonal();
}

} // namespace Xped
