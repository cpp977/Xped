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
    using MatrixInterface::get_raw_data;
    using MatrixInterface::getVal;
    using MatrixInterface::print;
    using MatrixInterface::scale;
    using MatrixInterface::setConstant;
    using MatrixInterface::setRandom;
    using MatrixInterface::setVal;
    using MatrixInterface::setZero;

    using TensorInterface::construct;
    using TensorInterface::get_raw_data;
    using TensorInterface::getVal;
    using TensorInterface::print;
    using TensorInterface::setConstant;
    using TensorInterface::setRandom;
    using TensorInterface::setVal;
    using TensorInterface::setZero;

    using VectorInterface::construct;
    using VectorInterface::construct_with_zero;
    using VectorInterface::print;
    using VectorInterface::scale;
    using VectorInterface::setConstant;
    using VectorInterface::setRandom;
    using VectorInterface::setZero;

    template <std::size_t Rank, typename Derived, typename Scalar>
    static void set_block_from_tensor(Eigen::MatrixBase<Derived>& M,
                                      const Indextype& row,
                                      const Indextype& col,
                                      const Indextype& rows,
                                      const Indextype& cols,
                                      const Scalar& scale,
                                      const TType<Scalar, Rank>& T);

    template <std::size_t Rank, typename Derived, typename Scalar>
    static void add_to_block_from_tensor(Eigen::MatrixBase<Derived>& M,
                                         const Indextype& row,
                                         const Indextype& col,
                                         const Indextype& rows,
                                         const Indextype& cols,
                                         const Scalar& scale,
                                         const TType<Scalar, Rank>& T);

    template <std::size_t Rank, typename Derived, typename Scalar>
    static void set_block_from_tensor(Eigen::MatrixBase<Derived>&& M,
                                      const Indextype& row,
                                      const Indextype& col,
                                      const Indextype& rows,
                                      const Indextype& cols,
                                      const Scalar& scale,
                                      const TType<Scalar, Rank>& T);

    template <std::size_t Rank, typename Derived, typename Scalar>
    static void add_to_block_from_tensor(Eigen::MatrixBase<Derived>&& M,
                                         const Indextype& row,
                                         const Indextype& col,
                                         const Indextype& rows,
                                         const Indextype& cols,
                                         const Scalar& scale,
                                         const TType<Scalar, Rank>& T);

    template <std::size_t Rank, typename Derived>
    static TType<typename Derived::Scalar, Rank> tensor_from_matrix_block(const Eigen::MatrixBase<Derived>& M,
                                                                          const Indextype& row,
                                                                          const Indextype& col,
                                                                          const Indextype& rows,
                                                                          const Indextype& cols,
                                                                          const std::array<Indextype, Rank>& dims);

    template <typename Derived>
    static void diagonal_head_matrix_to_vector(VType<typename Derived::Scalar>& V, const Eigen::MatrixBase<Derived>& M, const Indextype& n_elems);

    template <typename Scalar>
    static void vec_diff(const Eigen::Matrix<Scalar, -1, 1>& vec, MType<Scalar>& res);

    template <typename Scalar>
    static void vec_add(const Eigen::Matrix<Scalar, -1, 1>& vec, MType<Scalar>& res);

    template <typename Derived>
    static std::tuple<MType<typename Derived::Scalar>, VType<typename Derived::Scalar>, MType<typename Derived::Scalar>>
    svd(const Eigen::MatrixBase<Derived>& M);

    template <typename Scalar>
    static MType<Scalar> vec_to_diagmat(const VType<Scalar>& V);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/PlainInterface_Eigen_impl.cpp"
#endif

#endif
