#ifndef PLAIN_INTERFACE_CYCLOPS_IMPL_H_
#define PLAIN_INTERFACE_CYCLOPS_IMPL_H_

#include "ctf.hpp"

namespace Xped {

struct PlainInterface : public MatrixInterface, public TensorInterface, public VectorInterface
{
    // typedefs
    using Indextype = int;
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

    template <std::size_t Rank, typename Scalar>
    static void set_block_from_tensor(MType<Scalar>& M,
                                      const Indextype& row_off,
                                      const Indextype& col_off,
                                      const Indextype& rows,
                                      const Indextype& cols,
                                      const Scalar& scale,
                                      const TType<Scalar, Rank>& T);

    template <std::size_t Rank, typename Scalar>
    static void add_to_block_from_tensor(MType<Scalar>& M,
                                         const Indextype& row_off,
                                         const Indextype& col_off,
                                         const Indextype& rows,
                                         const Indextype& cols,
                                         const Scalar& scale,
                                         const TType<Scalar, Rank>& T);

    template <std::size_t Rank, typename Scalar>
    static TType<Scalar, Rank> tensor_from_matrix_block(const MType<Scalar>& M,
                                                        const Indextype& row_off,
                                                        const Indextype& col_off,
                                                        const Indextype& rows,
                                                        const Indextype& cols,
                                                        const std::array<Indextype, Rank>& dims);

    template <typename MT>
    static void diagonal_head_matrix_to_vector(VType<typename ctf_traits<MT>::Scalar>& V, MT&& M, const Indextype& n_elems);

    template <typename MT>
    static std::tuple<MType<typename ctf_traits<MT>::Scalar>, VType<typename ctf_traits<MT>::Scalar>, MType<typename ctf_traits<MT>::Scalar>>
    svd(MT&& M);

    template <typename Scalar, typename VT>
    static MType<Scalar> vec_to_diagmat(VT&& V);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/PlainInterface_Cyclops_impl.cpp"
#endif

#endif
