#ifndef PLAIN_INTERFACE_CYCLOPS_IMPL_H_
#define PLAIN_INTERFACE_CYCLOPS_IMPL_H_
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
                                      const TType<Scalar, Rank>& T)
    {
        spdlog::get("info")->info("Entering PlainInterface::set_block_from_tensor().");
        spdlog::get("info")->info("Set block of mat with dims=({},{})", M.nrow, M.ncol);
        spdlog::get("info")->info("Block dims=({},{})", rows, cols);
        spdlog::get("info")->info("Block offs=({},{})", row_off, col_off);
        spdlog::get("info")->info("Set from tensor with rank={}", Rank);
        std::array<Indextype, 2> mat_dims = {rows, cols};
        auto Tmat = reshape<Scalar, Rank, 2>(T, mat_dims);
        const std::array<Indextype, 2> offsets = {row_off, col_off};
        const std::array<Indextype, 2> ends = {row_off + rows, col_off + cols};
        const std::array<Indextype, 2> offsets_Tmat = {0, 0};
        // std::array<Indextype, Rank> ends_T;
        // for(std::size_t r = 0; r < Rank; r++) { spdlog::get("info")->info("Tensor dim r={} is {}", r, T.lens[r]); }
        // std::copy(T.lens, T.lens + Rank, std::begin(ends_T));
        // std::fill(offsets_T.begin(), offsets_T.end(), 0);
        M.slice(offsets.data(), ends.data(), Scalar(0.), Tmat, offsets_Tmat.data(), mat_dims.data(), scale);
        spdlog::get("info")->info("Leaving PlainInterface::set_block_from_tensor().");
    }

    template <typename Scalar, std::size_t Rank>
    static void add_to_block_from_tensor(MType<Scalar>& M,
                                         const Indextype& row_off,
                                         const Indextype& col_off,
                                         const Indextype& rows,
                                         const Indextype& cols,
                                         const Scalar& scale,
                                         const TType<Scalar, Rank>& T)
    {
        spdlog::get("info")->info("Entering PlainInterface::add_to_block_from_tensor().");
        spdlog::get("info")->info("Add to block of mat with dims=({},{})", M.nrow, M.ncol);
        std::array<Indextype, 2> mat_dims = {rows, cols};
        auto Tmat = reshape<Scalar, Rank, 2>(T, mat_dims);
        const std::array<Indextype, 2> offsets = {row_off, col_off};
        const std::array<Indextype, 2> ends = {row_off + rows, col_off + cols};
        const std::array<Indextype, 2> offsets_Tmat = {0, 0};
        // std::array<Indextype, Rank> ends_T;
        // std::copy(T.lens, T.lens + Rank, std::begin(ends_T));
        // std::fill(offsets_T.begin(), offsets_T.end(), 0);
        M.slice(offsets.data(), ends.data(), Scalar(1.), Tmat, offsets_Tmat.data(), mat_dims.data(), scale);
        spdlog::get("info")->info("Leaving PlainInterface::add_to_block_from_tensor().");
    }

    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> tensor_from_matrix_block(const MType<Scalar>& M,
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
    static void diagonal_head_matrix_to_vector(VType<Scalar>& V, MT&& M, const Indextype& n_elems)
    {
        spdlog::get("info")->info("Entering PlainInterface::diagonal_head_matrix_to_vector().");
        assert(n_elems <= M.nrow);
        assert(n_elems <= M.ncol);
        spdlog::get("info")->info("Got mat with size=({},{}) and cut size={}", M.nrow, M.ncol, n_elems);
        CTF::Vector<Scalar> tmp(M.nrow);
        spdlog::get("info")->info("Constructed vector of size={}", M.nrow);
        tmp["i"] = M["ii"];
        std::array<int, 1> offsets = {0};
        std::array<int, 1> ends = {n_elems};
        V = tmp.slice(offsets.data(), ends.data());
        spdlog::get("info")->info("Leaving PlainInterface::diagonal_head_matrix_to_vector().");
    }

    template <typename Scalar, typename MT>
    static std::tuple<MType<Scalar>, VType<Scalar>, MType<Scalar>> svd(MT&& M)
    {
        MType<Scalar> U, Vdag;
        CTF::Vector<Scalar> S;
        M.svd(U, S, Vdag);
        // MType<Scalar> S(S_tmp.len, S_tmp.len);
        // S["ii"] = S_tmp["i"];

        return std::make_tuple(U, S, Vdag);
    }

    template <typename Scalar, typename VT>
    static MType<Scalar> vec_to_diagmat(VT&& V)
    {
        CTF::Matrix<Scalar> mat(V.len, V.len);
        mat["ii"] = V["i"];
        return mat;
    }
};
#endif
