#ifndef PLAIN_INTERFACE_CYCLOPS_IMPL_H_
#define PLAIN_INTERFACE_CYCLOPS_IMPL_H_

#include "ctf.hpp"

namespace Xped {

template <>
struct PlainInterface<CyclopsMatrixLib, CyclopsTensorLib, CyclopsVectorLib>
    : public MatrixInterface<CyclopsMatrixLib>, public TensorInterface<CyclopsTensorLib>, public VectorInterface<CyclopsVectorLib>
{
    // typedefs
    using Indextype = int;
    using MatrixInterface<CyclopsMatrixLib>::construct;
    using MatrixInterface<CyclopsMatrixLib>::construct_with_zero;
    using MatrixInterface<CyclopsMatrixLib>::setZero;
    using MatrixInterface<CyclopsMatrixLib>::setRandom;
    using MatrixInterface<CyclopsMatrixLib>::setConstant;
    using MatrixInterface<CyclopsMatrixLib>::scale;
    using MatrixInterface<CyclopsMatrixLib>::getVal;
    using MatrixInterface<CyclopsMatrixLib>::add;
    using MatrixInterface<CyclopsMatrixLib>::difference;
    using MatrixInterface<CyclopsMatrixLib>::print;

    using TensorInterface<CyclopsTensorLib>::construct;
    using TensorInterface<CyclopsTensorLib>::setZero;
    using TensorInterface<CyclopsTensorLib>::setRandom;
    using TensorInterface<CyclopsTensorLib>::setConstant;
    using TensorInterface<CyclopsTensorLib>::getVal;
    using TensorInterface<CyclopsTensorLib>::print;

    using VectorInterface<CyclopsVectorLib>::construct;
    using VectorInterface<CyclopsVectorLib>::construct_with_zero;
    using VectorInterface<CyclopsVectorLib>::setZero;
    using VectorInterface<CyclopsVectorLib>::setRandom;
    using VectorInterface<CyclopsVectorLib>::setConstant;
    using VectorInterface<CyclopsVectorLib>::scale;
    using VectorInterface<CyclopsVectorLib>::print;

    template <typename Scalar, std::size_t Rank>
    static void set_block_from_tensor(MType<Scalar>& M,
                                      const Indextype& row_off,
                                      const Indextype& col_off,
                                      const Indextype& rows,
                                      const Indextype& cols,
                                      const Scalar& scale,
                                      const TType<Scalar, Rank>& T);

    template <typename Scalar, std::size_t Rank>
    static void add_to_block_from_tensor(MType<Scalar>& M,
                                         const Indextype& row_off,
                                         const Indextype& col_off,
                                         const Indextype& rows,
                                         const Indextype& cols,
                                         const Scalar& scale,
                                         const TType<Scalar, Rank>& T);

    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> tensor_from_matrix_block(const MType<Scalar>& M,
                                                        const Indextype& row_off,
                                                        const Indextype& col_off,
                                                        const Indextype& rows,
                                                        const Indextype& cols,
                                                        const std::array<Indextype, Rank>& dims);

    template <typename Scalar, typename MT>
    static void diagonal_head_matrix_to_vector(VType<Scalar>& V, MT&& M, const Indextype& n_elems);

    template <typename Scalar, typename MT>
    static std::tuple<MType<Scalar>, VType<Scalar>, MType<Scalar>> svd(MT&& M);

    template <typename Scalar, typename VT>
    static MType<Scalar> vec_to_diagmat(VT&& V);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/PlainInterface_Cyclops_impl.cpp"
#endif

#endif
