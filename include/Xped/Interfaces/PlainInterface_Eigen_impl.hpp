#ifndef PLAIN_INTERFACE_EIGEN_IMPL_H_
#define PLAIN_INTERFACE_EIGEN_IMPL_H_

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

template <>
struct PlainInterface<EigenMatrixLib, EigenTensorLib, EigenVectorLib>
    : public MatrixInterface<EigenMatrixLib>, public TensorInterface<EigenTensorLib>, public VectorInterface<EigenVectorLib>
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

    using TensorInterface<EigenTensorLib>::construct;
    using TensorInterface<EigenTensorLib>::setZero;
    using TensorInterface<EigenTensorLib>::setRandom;
    using TensorInterface<EigenTensorLib>::setConstant;
    using TensorInterface<EigenTensorLib>::getVal;
    using TensorInterface<EigenTensorLib>::print;

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
                                      const TType<Scalar, Rank>& T);

    template <typename Scalar, std::size_t Rank>
    static void add_to_block_from_tensor(MType<Scalar>& M,
                                         const Indextype& row,
                                         const Indextype& col,
                                         const Indextype& rows,
                                         const Indextype& cols,
                                         const Scalar& scale,
                                         const TType<Scalar, Rank>& T);

    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> tensor_from_matrix_block(const MType<Scalar>& M,
                                                        const Indextype& row,
                                                        const Indextype& col,
                                                        const Indextype& rows,
                                                        const Indextype& cols,
                                                        const std::array<Indextype, Rank>& dims);

    template <typename Scalar>
    static void diagonal_head_matrix_to_vector(VType<Scalar>& V, const MType<Scalar>& M, const Indextype& n_elems);

    template <typename Scalar>
    static std::tuple<MType<Scalar>, VType<Scalar>, MType<Scalar>> svd(const MType<Scalar>& M);

    template <typename Scalar>
    static MType<Scalar> vec_to_diagmat(const VType<Scalar>& V);
};

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/PlainInterface_Eigen_impl.cpp"
#endif

#endif
