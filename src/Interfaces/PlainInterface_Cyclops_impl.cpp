#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Interfaces/PlainInterface_Cyclops_impl.hpp"

#include "spdlog/spdlog.h"

// typedefs
using Indextype = int;

template <typename Scalar, std::size_t Rank>
using TType = CTF::Tensor<Scalar>;
template <typename Scalar, std::size_t Rank>
using cTType = const CTF::Tensor<Scalar>;

template <typename Scalar, std::size_t Rank>
using MapTType = CTF::Tensor<Scalar>;
template <typename Scalar, std::size_t Rank>
using cMapTType = const CTF::Tensor<Scalar>;

template <typename Scalar>
using MType = CTF::Matrix<Scalar>;
template <typename Scalar>
using cMType = const CTF::Matrix<Scalar>;

template <typename Scalar>
using MapMType = CTF::Matrix<Scalar>;
template <typename Scalar>
using cMapMType = const CTF::Matrix<Scalar>;

template <typename Scalar>
using VType = CTF::Vector<Scalar>;

template <typename Scalar, std::size_t Rank>
void PlainInterface<CyclopsMatrixLib, CyclopsTensorLib, CyclopsVectorLib>::set_block_from_tensor(MType<Scalar>& M,
                                                                                                 const Indextype& row_off,
                                                                                                 const Indextype& col_off,
                                                                                                 const Indextype& rows,
                                                                                                 const Indextype& cols,
                                                                                                 const Scalar& scale,
                                                                                                 const TType<Scalar, Rank>& T)
{
    SPDLOG_INFO("Entering PlainInterface::set_block_from_tensor().");
    SPDLOG_INFO("Set block of mat with dims=({},{})", M.nrow, M.ncol);
    SPDLOG_INFO("Block dims=({},{})", rows, cols);
    SPDLOG_INFO("Block offs=({},{})", row_off, col_off);
    SPDLOG_INFO("Set from tensor with rank={}", Rank);
    std::array<Indextype, 2> mat_dims = {rows, cols};
    auto Tmat = reshape<Scalar, Rank, 2>(T, mat_dims);
    const std::array<Indextype, 2> offsets = {row_off, col_off};
    const std::array<Indextype, 2> ends = {row_off + rows, col_off + cols};
    const std::array<Indextype, 2> offsets_Tmat = {0, 0};
    // std::array<Indextype, Rank> ends_T;
    // for(std::size_t r = 0; r < Rank; r++) { SPDLOG_INFO("Tensor dim r={} is {}", r, T.lens[r]); }
    // std::copy(T.lens, T.lens + Rank, std::begin(ends_T));
    // std::fill(offsets_T.begin(), offsets_T.end(), 0);
    M.slice(offsets.data(), ends.data(), Scalar(0.), Tmat, offsets_Tmat.data(), mat_dims.data(), scale);
    SPDLOG_INFO("Leaving PlainInterface::set_block_from_tensor().");
}

template <typename Scalar, std::size_t Rank>
void PlainInterface<CyclopsMatrixLib, CyclopsTensorLib, CyclopsVectorLib>::add_to_block_from_tensor(MType<Scalar>& M,
                                                                                                    const Indextype& row_off,
                                                                                                    const Indextype& col_off,
                                                                                                    const Indextype& rows,
                                                                                                    const Indextype& cols,
                                                                                                    const Scalar& scale,
                                                                                                    const TType<Scalar, Rank>& T)
{
    SPDLOG_INFO("Entering PlainInterface::add_to_block_from_tensor().");
    SPDLOG_INFO("Add to block of mat with dims=({},{})", M.nrow, M.ncol);
    std::array<Indextype, 2> mat_dims = {rows, cols};
    auto Tmat = reshape<Scalar, Rank, 2>(T, mat_dims);
    const std::array<Indextype, 2> offsets = {row_off, col_off};
    const std::array<Indextype, 2> ends = {row_off + rows, col_off + cols};
    const std::array<Indextype, 2> offsets_Tmat = {0, 0};
    // std::array<Indextype, Rank> ends_T;
    // std::copy(T.lens, T.lens + Rank, std::begin(ends_T));
    // std::fill(offsets_T.begin(), offsets_T.end(), 0);
    M.slice(offsets.data(), ends.data(), Scalar(1.), Tmat, offsets_Tmat.data(), mat_dims.data(), scale);
    SPDLOG_INFO("Leaving PlainInterface::add_to_block_from_tensor().");
}

template <typename Scalar, std::size_t Rank>
TType<Scalar, Rank>
PlainInterface<CyclopsMatrixLib, CyclopsTensorLib, CyclopsVectorLib>::tensor_from_matrix_block(const MType<Scalar>& M,
                                                                                               const Indextype& row_off,
                                                                                               const Indextype& col_off,
                                                                                               const Indextype& rows,
                                                                                               const Indextype& cols,
                                                                                               const std::array<Indextype, Rank>& dims)
{
    std::array<Indextype, 2> offsets = {row_off, col_off};
    std::array<Indextype, 2> ends = {row_off + rows, col_off + cols};
    MType<Scalar> submatrix = M.slice(offsets.data(), ends.data());
    TType<Scalar, Rank> T(Rank, dims.data(), *M.wrld);
    int64_t nvals;
    int64_t* indices;
    Scalar* data;
    submatrix.get_local_data(&nvals, &indices, &data);
    T.write(nvals, indices, data);
    free(indices);
    delete[] data;
    return T;
}

template <typename Scalar, typename MT>
void PlainInterface<CyclopsMatrixLib, CyclopsTensorLib, CyclopsVectorLib>::diagonal_head_matrix_to_vector(VType<Scalar>& V,
                                                                                                          MT&& M,
                                                                                                          const Indextype& n_elems)
{
    SPDLOG_INFO("Entering PlainInterface::diagonal_head_matrix_to_vector().");
    assert(n_elems <= M.nrow);
    assert(n_elems <= M.ncol);
    SPDLOG_INFO("Got mat with size=({},{}) and cut size={}", M.nrow, M.ncol, n_elems);
    CTF::Vector<Scalar> tmp(M.nrow, *M.wrld);
    SPDLOG_INFO("Constructed vector of size={}", M.nrow);
    tmp["i"] = M["ii"];
    SPDLOG_INFO("Set the elements of the tmp vec.");
    std::array<int, 1> offsets = {0};
    std::array<int, 1> ends = {n_elems};
    V = tmp.slice(offsets.data(), ends.data());
    SPDLOG_INFO("Leaving PlainInterface::diagonal_head_matrix_to_vector().");
}

template <typename Scalar, typename MT>
std::tuple<MType<Scalar>, VType<Scalar>, MType<Scalar>> PlainInterface<CyclopsMatrixLib, CyclopsTensorLib, CyclopsVectorLib>::svd(MT&& M)
{
    MType<Scalar> U, Vdag;
    CTF::Vector<Scalar> S;
    M.svd(U, S, Vdag);
    // MType<Scalar> S(S_tmp.len, S_tmp.len);
    // S["ii"] = S_tmp["i"];

    return std::make_tuple(U, S, Vdag);
}

template <typename Scalar, typename VT>
MType<Scalar> PlainInterface<CyclopsMatrixLib, CyclopsTensorLib, CyclopsVectorLib>::vec_to_diagmat(VT&& V)
{
    CTF::Matrix<Scalar> mat(V.len, V.len, *V.wrld);
    mat["ii"] = V["i"];
    return mat;
}
