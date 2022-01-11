#ifndef PLAIN_INTERFACE_EIGEN_IMPL_H_
#define PLAIN_INTERFACE_EIGEN_IMPL_H_

#include <Eigen/Core>
#include <Eigen/SVD>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Xped {

template <>
struct PlainInterface<EigenMatrixLib, ArrayTensorLib, EigenVectorLib>
    : public MatrixInterface<EigenMatrixLib>, public TensorInterface<ArrayTensorLib>, public VectorInterface<EigenVectorLib>
{
    using Indextype = Eigen::Index;
    using MatrixInterface<EigenMatrixLib>::construct;
    using MatrixInterface<EigenMatrixLib>::construct_with_zero;
    using MatrixInterface<EigenMatrixLib>::setZero;
    using MatrixInterface<EigenMatrixLib>::setRandom;
    using MatrixInterface<EigenMatrixLib>::setConstant;
    using MatrixInterface<EigenMatrixLib>::scale;
    using MatrixInterface<EigenMatrixLib>::getVal;
    using MatrixInterface<EigenMatrixLib>::add;
    using MatrixInterface<EigenMatrixLib>::difference;
    using MatrixInterface<EigenMatrixLib>::print;

    using TensorInterface<ArrayTensorLib>::construct;
    using TensorInterface<ArrayTensorLib>::setZero;
    using TensorInterface<ArrayTensorLib>::setRandom;
    using TensorInterface<ArrayTensorLib>::setConstant;
    using TensorInterface<ArrayTensorLib>::getVal;
    using TensorInterface<ArrayTensorLib>::print;

    using VectorInterface<EigenVectorLib>::construct;
    using VectorInterface<EigenVectorLib>::construct_with_zero;
    using VectorInterface<EigenVectorLib>::setZero;
    using VectorInterface<EigenVectorLib>::setRandom;
    using VectorInterface<EigenVectorLib>::setConstant;
    using VectorInterface<EigenVectorLib>::scale;
    using VectorInterface<EigenVectorLib>::print;

    template <typename Scalar, std::size_t Rank>
    static void set_block_from_tensor(MType<Scalar>& M,
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
    static void add_to_block_from_tensor(MType<Scalar>& M,
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
    static TType<Scalar, Rank> tensor_from_matrix_block(const MType<Scalar>& M,
                                                        const Indextype& row,
                                                        const Indextype& col,
                                                        const Indextype& rows,
                                                        const Indextype& cols,
                                                        const std::array<Indextype, Rank>& dims)
    {
        MType<Scalar> submatrix = M.block(row, col, rows, cols);
        nda::dim<-9, -9, 1> first_dim;
        first_dim.set_extent(dims[0]);
        std::array<nda::dim<-9, -9, -9>, Rank - 1> shape_data;
        for(std::size_t i = 1; i < Rank; i++) {
            shape_data[i - 1].set_extent(dims[i]);
            shape_data[i - 1].set_stride(std::accumulate(dims.begin(), dims.begin() + i, 1ul, std::multiplies<Scalar>()));
        }
        auto dims_tuple = std::tuple_cat(std::make_tuple(first_dim), TensorInterface<ArrayTensorLib>::as_tuple(shape_data));

        nda::dense_shape<Rank> block_shape(dims_tuple);

        cMapTType<Scalar, Rank> tensorview(submatrix.data(), block_shape);

        // cMapTType<Scalar, Rank> tensorview = cMap(submatrix.data(), dims);
        return construct<Scalar, Rank>(tensorview);
    }

    template <typename Scalar>
    static void diagonal_head_matrix_to_vector(VType<Scalar>& V, const MType<Scalar>& M, const Indextype& n_elems)
    {
        V = M.diagonal().head(n_elems);
    }

    template <typename Scalar>
    static std::tuple<MType<Scalar>, VType<Scalar>, MType<Scalar>> svd(const MType<Scalar>& M)
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
    static MType<Scalar> vec_to_diagmat(const VType<Scalar>& V)
    {
        return V.matrix().asDiagonal();
    }
};

} // namespace Xped

#endif
