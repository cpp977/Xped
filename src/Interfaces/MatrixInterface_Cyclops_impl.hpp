#ifndef MATRIX_INTERFACE_CYCLOPS_IMPL_H_
#define MATRIX_INTERFACE_CYCLOPS_IMPL_H_

#include "spdlog/spdlog.h"

#include "ctf.hpp"

#include "Interfaces/MatrixMultiplication.hpp"

// #define XPED_CONST
// #define XPED_REF &&

template <>
struct MatrixInterface<CyclopsMatrixLib>
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
    static MType<Scalar> construct(const MIndextype& rows, const MIndextype& cols, CTF::World& world)
    {
        return MType<Scalar>(rows, cols, world);
    }

    template <typename Scalar>
    static MType<Scalar> construct_with_zero(const MIndextype& rows, const MIndextype& cols, CTF::World& world)
    {
        SPDLOG_INFO("Entering construct with zero");
        SPDLOG_INFO("rows={}, cols={}", rows, cols);
        return MType<Scalar>(rows, cols, world);
    }

    template <typename Scalar>
    static void resize(MType<Scalar>& M, const MIndextype& new_rows, const MIndextype& new_cols)
    {
        M = CTF::Matrix<Scalar>(new_rows, new_cols, *M.wrld);
    }

    // initialization
    template <typename Scalar>
    static void setZero(MType<Scalar>& M)
    {
        M["ij"] = Scalar(0.);
    }

    template <typename Scalar>
    static void setRandom(MType<Scalar>& M)
    {
        M.fill_random(-1., 1.);
    }

    template <typename Scalar>
    static void setIdentity(MType<Scalar>& M)
    {
        M["ii"] = Scalar(1.);
    }

    template <typename Scalar>
    static void setConstant(MType<Scalar>& M, const Scalar& val)
    {
        M["ij"] = val;
    }

    template <typename Scalar>
    static MType<Scalar> Identity(const MIndextype& rows, const MIndextype& cols, CTF::World& world)
    {
        SPDLOG_TRACE("Begin of Identity()");
        SPDLOG_TRACE("rows: " + std::to_string(rows) + ", cols: " + std::to_string(cols));
        MType<Scalar> M(rows, cols, world);
        SPDLOG_TRACE("Constructor passed.");
        M["ii"] = Scalar(1.);
        SPDLOG_TRACE("Set diagonal to 1..");
        return M;
    }

    // shape
    template <typename Scalar>
    static MIndextype rows(const MType<Scalar>& M)
    {
        return M.nrow;
    }

    template <typename Scalar>
    static MIndextype cols(const MType<Scalar>& M)
    {
        return M.ncol;
    }

    // reduction
    template <typename Scalar, typename MT>
    static Scalar trace(MT&& M)
    {
        Scalar out = M["ii"];
        return out;
    }

    template <typename Scalar>
    static Scalar getVal(const MType<Scalar>& M, const MIndextype& row, const MIndextype& col)
    {
        int64_t global_idx = 1 * row + M.nrow * col;
        Scalar out = 0.;
        int64_t nvals;
        Scalar* data;
        M.get_all_data(&nvals, &data);
        out = data[global_idx];
        delete[] data;
        return out;
    }

    template <typename Scalar, typename MT1, typename MT2>
    static MType<Scalar> kronecker_prod(MT1&& M1, MT2&& M2)
    {
        assert(*M1.wrld == *M2.wrld and "Tensors needs to live on the same world for kroneckerProd().");
        std::array<int64_t, 4> dims = {M2.nrow, M1.nrow, M2.ncol, M1.ncol};
        CTF::Tensor<Scalar> tmp(4, dims.data(), *M1.wrld);
        tmp["kilj"] = M1["ij"] * M2["kl"];
        MType<Scalar> res(M1.nrow * M2.nrow, M1.ncol * M2.ncol, *M1.wrld);
        int64_t nvals;
        int64_t* indices;
        Scalar* data;
        tmp.get_local_data(&nvals, &indices, &data);
        res.write(nvals, indices, data);
        free(indices);
        delete[] data;
        return res;
    }

    template <typename Scalar, typename MT1, typename MT2>
    static MType<Scalar> prod(MT1&& M1, MT2&& M2)
    {
        assert(*M1.wrld == *M2.wrld and "Tensors needs to live on the same world for prod().");
        MType<Scalar> res(M1.nrow, M2.ncol, *M1.wrld);
        res["ik"] = M1["ij"] * M2["jk"];
        return res;
    }

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod(const Scalar& scale, MatrixExpr1&& M1, MatrixExpr2&& M2, MatrixExpr3&& M3, MatrixExprRes& Mres)
    {
        std::vector<std::size_t> cost(2);
        cost = internal::mult_cost(
            std::array<int64_t, 2>{M1.nrow, M1.ncol}, std::array<int64_t, 2>{M2.nrow, M2.ncol}, std::array<int64_t, 2>{M3.nrow, M3.ncol});
        std::size_t opt_mult = std::min_element(cost.begin(), cost.end()) - cost.begin();

        if(opt_mult == 0) {
            MType<Scalar> Mtmp = prod<Scalar>(M1, M2);
            Mres["ij"] = scale * Mtmp["ik"] * M3["kj"];
        } else if(opt_mult == 1) {
            MType<Scalar> Mtmp = prod<Scalar>(M2, M3);
            Mres["ij"] = scale * M1["ik"] * Mtmp["kj"];
        }
    }

    template <typename Scalar, typename MatrixExpr1, typename MatrixExpr2, typename MatrixExpr3, typename MatrixExprRes>
    static void optimal_prod_add(const Scalar& scale, MatrixExpr1&& M1, MatrixExpr2&& M2, MatrixExpr3&& M3, MatrixExprRes& Mres)
    {
        std::vector<std::size_t> cost(2);
        cost = internal::mult_cost(
            std::array<int64_t, 2>{M1.nrow, M1.ncol}, std::array<int64_t, 2>{M2.nrow, M2.ncol}, std::array<int64_t, 2>{M3.nrow, M3.ncol});
        std::size_t opt_mult = std::min_element(cost.begin(), cost.end()) - cost.begin();

        if(opt_mult == 0) {
            MType<Scalar> Mtmp = prod<Scalar>(M1, M2);
            Mres["ij"] = Mres["ij"] + scale * Mtmp["ik"] * M3["kj"];
        } else if(opt_mult == 1) {
            MType<Scalar> Mtmp = prod<Scalar>(M2, M3);
            Mres["ij"] = Mres["ij"] + scale * M1["ik"] * Mtmp["kj"];
        }
    }

    template <typename Scalar, typename MT1, typename MT2>
    static auto add(MT1&& M1, MT2&& M2)
    {
        assert(*M1.wrld == *M2.wrld and "Tensors needs to live on the same world for add().");
        MType<Scalar> res(M1.nrow, M2.ncol, *M1.wrld);
        res["ij"] = M1["ij"] + M2["ij"];
        return res;
    }

    template <typename Scalar, typename MT1, typename MT2>
    static auto difference(MT1&& M1, MT2&& M2)
    {
        assert(*M1.wrld == *M2.wrld and "Tensors needs to live on the same world for difference().");
        MType<Scalar> res(M1.nrow, M2.ncol, *M1.wrld);
        res["ij"] = M1["ij"] - M2["ij"];
        return res;
    }

    template <typename Scalar>
    static void scale(MType<Scalar>& M, const Scalar& val)
    {
        M.scale(val, "ij");
    }

    template <typename Scalar, typename MT>
    static auto adjoint(MT&& M)
    {
        MType<Scalar> N(M.ncol, M.nrow, *M.wrld);
        N["ij"] = M["ji"];
        return N;
    }

    // template <typename Scalar>
    // static auto adjoint(CTF::Tensor<Scalar>& M)
    // {
    //     MType<Scalar> N(M.lens[1], M.lens[0]);
    //     N["ij"] = M["ji"];
    //     return N;
    // }

    template <typename Scalar>
    static MType<Scalar>
    block(const MType<Scalar>& M, const MIndextype& row_off, const MIndextype& col_off, const MIndextype& rows, const MIndextype& cols)
    {
        SPDLOG_INFO("Entering MatrixInterface::block()");
        SPDLOG_INFO("Extract block of mat with dims=({},{})", M.nrow, M.ncol);
        std::array<MIndextype, 2> offsets = {row_off, col_off};
        std::array<MIndextype, 2> ends = {row_off + rows, col_off + cols};
        SPDLOG_INFO("Prepared the dims: offs=({},{}), ends=({},{})", offsets[0], offsets[1], ends[0], ends[1]);
        // {
        //     volatile int i = 0;
        //     char hostname[256];
        //     gethostname(hostname, sizeof(hostname));
        //     printf("PID %d on %s ready for attach\n", getpid(), hostname);
        //     fflush(stdout);
        //     while(0 == i) sleep(5);
        // }
        CTF::Matrix<Scalar> out = (M.slice(offsets.data(), ends.data()));
        SPDLOG_INFO("Leaving MatrixInterface::block()");
        return out;
    }

    template <typename Scalar>
    static void add_to_block(MType<Scalar>& M1,
                             const MIndextype& row_off,
                             const MIndextype& col_off,
                             const MIndextype& rows,
                             const MIndextype& cols,
                             const MType<Scalar>& M2)
    {
        SPDLOG_INFO("Entering MatrixInterface::add_to_block().");
        SPDLOG_INFO("Add to block of mat with dims=({},{})", M1.nrow, M1.ncol);
        SPDLOG_INFO("Add mat with dims=({},{})", M2.nrow, M2.ncol);
        std::array<MIndextype, 2> offsets = {row_off, col_off};
        std::array<MIndextype, 2> ends = {row_off + rows, col_off + cols};
        std::array<MIndextype, 2> offsets_M2 = {0, 0};
        std::array<MIndextype, 2> ends_M2 = {static_cast<MIndextype>(M2.nrow), static_cast<MIndextype>(M2.ncol)};
        M1.slice(offsets.data(), ends.data(), Scalar(1.), M2, offsets_M2.data(), ends_M2.data(), Scalar(1.));
        SPDLOG_INFO("Leaving MatrixInterface::add_to_block().");
    }

    template <typename Scalar, typename MT>
    static void print(MT&& M)
    {
        M.print_matrix();
    }
};

#endif
