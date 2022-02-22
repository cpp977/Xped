#ifndef PLAIN_INTERFACE_EIGEN_IMPL_H_
#define PLAIN_INTERFACE_EIGEN_IMPL_H_

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Xped {

struct PlainInterface : public MatrixInterface, public TensorInterface, public VectorInterface
{
    using Indextype = Eigen::Index;
    using MatrixInterface::add;
    using MatrixInterface::construct;
    using MatrixInterface::construct_with_zero;
    using MatrixInterface::difference;
    using MatrixInterface::getVal;
    using MatrixInterface::print;
    using MatrixInterface::scale;
    using MatrixInterface::setConstant;
    using MatrixInterface::setRandom;
    using MatrixInterface::setZero;

    using TensorInterface::construct;
    using TensorInterface::getVal;
    using TensorInterface::print;
    using TensorInterface::setConstant;
    using TensorInterface::setRandom;
    using TensorInterface::setZero;

    using VectorInterface::construct;
    using VectorInterface::construct_with_zero;
    using VectorInterface::print;
    using VectorInterface::scale;
    using VectorInterface::setConstant;
    using VectorInterface::setRandom;
    using VectorInterface::setZero;

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

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/PlainInterface_Eigen_impl.cpp"
#endif

#endif
