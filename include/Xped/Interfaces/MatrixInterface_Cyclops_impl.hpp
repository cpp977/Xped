#ifndef MATRIX_INTERFACE_CYCLOPS_IMPL_H_
#define MATRIX_INTERFACE_CYCLOPS_IMPL_H_

#include "ctf.hpp"

namespace Xped {

template <typename T>
struct ctf_traits
{};

template <typename Scalar_>
struct ctf_traits<CTF::Tensor<Scalar_>>
{
    using Scalar = Scalar_;
};

template <typename Scalar_>
struct ctf_traits<CTF::Matrix<Scalar_>>
{
    using Scalar = Scalar_;
};

template <typename Scalar_>
struct ctf_traits<CTF::Tensor<Scalar_>&>
{
    using Scalar = Scalar_;
};

template <typename Scalar_>
struct ctf_traits<CTF::Matrix<Scalar_>&>
{
    using Scalar = Scalar_;
};

struct MatrixInterface
{
    // typedefs
    template <typename Scalar>
    using MType = CTF::Matrix<Scalar>;
    template <typename Scalar>
    using cMType = const CTF::Matrix<Scalar>;

    template <typename Scalar>
    using MapMType = CTF::Matrix<Scalar>;
    template <typename Scalar>
    using cMapMType = const CTF::Matrix<Scalar>;

    typedef int MIndextype;

    // constructors
    template <typename Scalar>
    static MType<Scalar> construct(const MIndextype& rows, const MIndextype& cols, CTF::World& world);

    template <typename Scalar>
    static MType<Scalar> construct_with_zero(const MIndextype& rows, const MIndextype& cols, CTF::World& world);

    template <typename Scalar>
    static void resize(MType<Scalar>& M, const MIndextype& new_rows, const MIndextype& new_cols);

    // initialization
    template <typename Scalar>
    static void setZero(MType<Scalar>& M);

    template <typename Scalar>
    static void setRandom(MType<Scalar>& M);

    template <typename Scalar>
    static void setIdentity(MType<Scalar>& M);

    template <typename Scalar>
    static void setConstant(MType<Scalar>& M, const Scalar& val);

    template <typename Scalar>
    static void setVal(MType<Scalar>& M, const MIndextype row, const MIndextype col, const Scalar& val);

    template <typename Scalar>
    static MType<Scalar> Identity(const MIndextype& rows, const MIndextype& cols, CTF::World& world);

    // shape
    template <typename Scalar>
    static MIndextype rows(const MType<Scalar>& M);

    template <typename Scalar>
    static MIndextype cols(const MType<Scalar>& M);

    // reduction
    template <typename MT>
    static typename ctf_traits<MT>::Scalar trace(MT&& M);

    template <typename MT>
    static typename ctf_traits<MT>::Scalar maxNorm(MT&& M);

    template <typename MT>
    static typename ctf_traits<MT>::Scalar maxCoeff(MT&& M, MIndextype& maxrow, MIndextype& maxcol);

    template <typename Scalar>
    static Scalar getVal(const MType<Scalar>& M, const MIndextype& row, const MIndextype& col);

    template <typename MT1, typename MT2>
    static MType<typename ctf_traits<MT1>::Scalar> kronecker_prod(MT1&& M1, MT2&& M2);

    template <typename MT1, typename MT2>
    static MType<typename ctf_traits<MT1>::Scalar> prod(MT1&& M1, MT2&& M2);

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod(const Scalar& scale, MatrixExpr1&& M1, MatrixExpr2&& M2, MatrixExpr3&& M3, MatrixExprRes& Mres);

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod_add(const Scalar& scale, MatrixExpr1&& M1, MatrixExpr2&& M2, MatrixExpr3&& M3, MatrixExprRes& Mres);

    template <typename MT>
    static MType<typename ctf_traits<MT>::Scalar> msqrt(MT&& M);

    template <typename MT>
    static MType<typename ctf_traits<MT>::Scalar> mexp(MT&& M);

    template <typename MT1, typename MT2>
    static MType<typename ctf_traits<MT1>::Scalar> add(MT1&& M1, MT2&& M2);

    template <typename MT1, typename MT2>
    static MType<typename ctf_traits<MT1>::Scalar> difference(MT1&& M1, MT2&& M2);

    template <typename Scalar>
    static void scale(MType<Scalar>& M, const Scalar& val);

    template <typename Scalar, typename MT>
    static MType<Scalar> unaryFunc(MT&& M, const std::function<Scalar(Scalar)>& func);

    template <typename Scalar, typename MT>
    static MType<Scalar> diagUnaryFunc(MT&& M, const std::function<Scalar(Scalar)>& func);

    template <typename Scalar, typename MTL, typename MTR>
    static MType<Scalar> diagBinaryFunc(MTL&& M_left, MTR&& M_right, const std::function<Scalar(Scalar, Scalar)>& func);

    template <typename Scalar, typename MTL, typename MTR>
    static MType<Scalar> binaryFunc(MTL&& M_left, MTR&& M_right, const std::function<Scalar(Scalar, Scalar)>& func);

    template <typename MT>
    static MType<typename ctf_traits<MT>::Scalar> adjoint(MT&& M);

    // template <typename Scalar>
    // static auto adjoint(CTF::Tensor<Scalar>& M)
    // {
    //     MType<Scalar> N(M.lens[1], M.lens[0]);
    //     N["ij"] = M["ji"];
    //     return N;
    // }

    template <typename Scalar>
    static MType<Scalar>
    block(const MType<Scalar>& M, const MIndextype& row_off, const MIndextype& col_off, const MIndextype& rows, const MIndextype& cols);

    template <typename MT>
    static std::pair<MType<typename ctf_traits<MT>::Scalar>, MType<typename ctf_traits<MT>::Scalar>> eigh(MT&& M);

    template <typename MT>
    static std::pair<MType<typename ctf_traits<MT>::Scalar>, MType<typename ctf_traits<MT>::Scalar>> qr(MT&& M);

    template <typename Scalar>
    static void add_to_block(MType<Scalar>& M1,
                             const MIndextype& row_off,
                             const MIndextype& col_off,
                             const MIndextype& rows,
                             const MIndextype& cols,
                             const MType<Scalar>& M2);

    template <typename MT>
    static void print(MT&& M);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/MatrixInterface_Cyclops_impl.cpp"
#endif

#endif
