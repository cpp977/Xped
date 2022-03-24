#ifndef TENSOR_INTERFACE_ARRAY_IMPL_H_
#define TENSOR_INTERFACE_ARRAY_IMPL_H_

#include <array.h>
#include <ein_reduce.h>

#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/Random.hpp"

namespace Xped {

namespace util {

template <typename T, T... Nums>
void print(seq::iseq<T, Nums...> seq)
{
    std::array<T, sizeof...(Nums)> s = {Nums...};
    std::cout << "seq=";
    for(auto x : s) { std::cout << x << " "; }
    std::cout << std::endl;
}

} // namespace util

template <typename Index, Index oldVal, Index newVal, typename S>
using seq_replace = seq::insert<seq::index_of<oldVal, S>, newVal, seq::remove<oldVal, S>>;

struct TensorInterface
{
    template <typename element_t, std::size_t N, size_t... Is>
    static nda::internal::tuple_of_n<element_t, N> as_tuple(std::array<element_t, N> const& arr, std::index_sequence<Is...>)
    {
        return std::make_tuple(arr[Is]...);
    }

    template <typename element_t, std::size_t N>
    static nda::internal::tuple_of_n<element_t, N> as_tuple(std::array<element_t, N> const& arr)
    {
        return as_tuple(arr, std::make_index_sequence<N>{});
    }

    // typedefs
    template <typename Scalar, std::size_t Rank>
    using TType = nda::dense_array<Scalar, Rank>;
    template <typename Scalar, std::size_t Rank>
    using cTType = const nda::dense_array<Scalar, Rank>;

    template <typename Scalar, std::size_t Rank>
    using MapTType = nda::dense_array_ref<Scalar, Rank>;
    template <typename Scalar, std::size_t Rank>
    using cMapTType = nda::const_dense_array_ref<Scalar, Rank>;

    // template<typename Scalar, std::size_t Rank> using Indextype = typename nda::dense_array<Scalar,Rank>::shape_type::index_type;
    using Indextype = nda::index_t;

    // constructors
    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> construct(const std::array<Indextype, Rank>& dims, mpi::XpedWorld& world = mpi::getUniverse())
    {
        return TType<Scalar, Rank>(as_tuple(dims));
    }

    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> construct(const MapTType<Scalar, Rank>& map)
    {
        TType<Scalar, Rank> out(map.shape());
        nda::copy(map, out);
        return out;
    }

    template <typename Scalar, std::size_t Rank>
    static cTType<Scalar, Rank> construct(const cMapTType<Scalar, Rank>& map)
    {
        TType<Scalar, Rank> tmp(map.shape());
        nda::copy(map, tmp);
        const cTType<Scalar, Rank> out(std::move(tmp));
        return out;
    }

    // map constructors
    template <typename Scalar, std::size_t Rank>
    static cMapTType<Scalar, Rank> cMap(const Scalar* data, const std::array<Indextype, Rank>& dims)
    {
        return cMapTType<Scalar, Rank>(data, as_tuple(dims));
    }

    template <typename Scalar, std::size_t Rank>
    static MapTType<Scalar, Rank> Map(Scalar* data, const std::array<Indextype, Rank>& dims)
    {
        return MapTType<Scalar, Rank>(data, as_tuple(dims));
    }

    // initialization
    template <typename Scalar, std::size_t Rank>
    static void setZero(TType<Scalar, Rank>& T)
    {
        T.for_each_value([](Scalar& d) { d = 0.; });
    }

    template <typename Scalar, std::size_t Rank>
    static void setRandom(TType<Scalar, Rank>& T)
    {
        T.for_each_value([](Scalar& d) { d = random::threadSafeRandUniform<Scalar, Scalar>(-1., 1.); });
    }

    template <typename Scalar, std::size_t Rank>
    static void setConstant(TType<Scalar, Rank>& T, const Scalar& val)
    {
        T.for_each_value([val](Scalar& d) { d = val; });
    }

    template <typename Scalar, int Rank>
    static void setVal(TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index, const Scalar& val)
    {
        T(as_tuple(index)) = val;
    }

    template <typename Scalar, int Rank>
    static Scalar getVal(const TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index)
    {
        if constexpr(Rank == 0) {
            return T();
        } else {
            return T(as_tuple(index));
        }
    }

    // raw data
    template <typename Scalar, int Rank>
    static const Scalar* get_raw_data(const TType<Scalar, Rank>& T)
    {
        return T.data();
    }

    template <typename Scalar, int Rank>
    static Scalar* get_raw_data(TType<Scalar, Rank>& T)
    {
        return T.data();
    }

    // shape info
    template <typename Scalar, std::size_t Rank>
    static std::array<Indextype, Rank> dimensions(const TType<Scalar, Rank>& T)
    {
        std::array<Indextype, Rank> out;
        auto tmp = nda::internal::tuple_to_array<nda::dim<>>(T.shape().dims());
        for(std::size_t d = 0; d < Rank; d++) { out[d] = tmp[d].extent(); }
        return out;
    }

    // tensorProd
    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> tensorProd(const TType<Scalar, Rank>& T1, const TType<Scalar, Rank>& T2)
    {
        std::array<Indextype, 2 * Rank> tmp_dims;
        std::array<Indextype, Rank> out_dims;
        auto dim1 = dimensions<Scalar, Rank>(T1);
        auto dim2 = dimensions<Scalar, Rank>(T2);
        for(std::size_t i = 0; i < Rank; i++) {
            tmp_dims[i] = dim2[i];
            tmp_dims[i + Rank] = dim1[i];
            out_dims[i] = dim1[i] * dim2[i];
        }
        TType<Scalar, 2 * Rank> tmp(as_tuple(tmp_dims));

        tmp = contract<Scalar, Rank, Rank>(T2, T1);

        TType<Scalar, Rank> res(as_tuple(out_dims));

        if constexpr(Rank == 1) {
            res = reshape<Scalar, 2 * Rank, Rank>(shuffle<Scalar, 2 * Rank>(tmp, seq::make<Indextype, 2>{}), out_dims);
        } else {
            seq::zip<seq::make<Indextype, Rank>, seq::make<Indextype, Rank, Rank>> shuffle_dims{};
            res = reshape<Scalar, 2 * Rank, Rank>(shuffle<Scalar, 2 * Rank>(tmp, shuffle_dims), out_dims);
        }
        return res;

        // cMapTType<Scalar, Rank> T2m(T2.data(), T2.shape());
        // typedef TType<Scalar, Rank> TensorType;
        // typedef Indextype Index;
        // std::array<Index, Rank> dims;
        // std::array<Index, Rank> dim1_array;
        // std::size_t tmp = 0;
        // for(const auto& d : nda::internal::tuple_to_array<nda::dim<>>(T1.shape().dims())) {
        //     dim1_array[tmp] = d.extent();
        //     tmp++;
        // }
        // std::array<Index, Rank> dim2_array;
        // tmp = 0;
        // for(const auto& d : nda::internal::tuple_to_array<nda::dim<>>(T2.shape().dims())) {
        //     dim2_array[tmp] = d.extent();
        //     tmp++;
        // }

        // for(Index i = 0; i < T1.rank(); i++) { dims[i] = dim1_array[i] * dim2_array[i]; }

        // TensorType res(as_tuple(dims));
        // res.for_each_value([](Scalar& d) { d = 0.; });
        // std::array<Index, Rank> extents = dim2_array;

        // std::vector<std::size_t> vec_dims;
        // for(const auto& d : dim1_array) { vec_dims.push_back(d); }
        // NestedLoopIterator Nelly(T1.rank(), vec_dims);

        // for(std::size_t i = Nelly.begin(); i != Nelly.end(); i++) {
        //     std::array<Index, Rank> indices;
        //     for(Index j = 0; j < T1.rank(); j++) { indices[j] = Nelly(j); }
        //     std::array<Index, Rank> offsets;
        //     for(Index i = 0; i < T1.rank(); i++) { offsets[i] = indices[i] * dim2_array[i]; }

        //     std::array<nda::interval<>, Rank> slices;
        //     for(std::size_t r = 0; r < Rank; r++) { slices[r] = nda::interval<>(offsets[r], extents[r]); }
        //     nda::dense_shape<Rank> new_s(as_tuple(slices));
        //     new_s.resolve();
        //     T2m.set_shape(new_s);
        //     nda::copy(T2m, res(as_tuple(slices)));
        //     res(as_tuple(slices)).for_each_value([&T1, &indices](Scalar& val) { val = val * T1(as_tuple(indices)); });
        //     ++Nelly;
        // }
        // return res;
    }

    template <typename Scalar, std::size_t Rank, Indextype... Is, typename Expr1, typename Expr2>
    static void addScale_helper(const Expr1& src, Expr2& dst, const Scalar& scale, seq::iseq<Indextype, Is...> S)
    {
        nda::ein_reduce(nda::ein<Is...>(dst) += nda::ein<>(scale) * nda::ein<Is...>(src));
    }

    template <typename Scalar, std::size_t Rank, typename Expr1, typename Expr2>
    static void addScale(const Expr1& src, Expr2& dst, const Scalar& scale)
    {
        return addScale_helper<Scalar, Rank>(src, dst, scale, seq::make<Indextype, Rank>());
    }

    // methods rvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is1, Indextype... Is2, Indextype... Ist>
    static auto contract_helper(const TType<Scalar, Rank1>& T1,
                                const TType<Scalar, Rank2>& T2,
                                seq::iseq<Indextype, Is1...> S1,
                                seq::iseq<Indextype, Is2...> S2,
                                seq::iseq<Indextype, Ist...> St)
    {
        return nda::make_ein_sum<Scalar, Ist...>(nda::ein<Is1...>(T1) * nda::ein<Is2...>(T2));
    }

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
    static TType<Scalar, Rank1 + Rank2 - sizeof...(Is)> contract(const TType<Scalar, Rank1>& T1, const TType<Scalar, Rank2>& T2)
    {
        static_assert(sizeof...(Is) % 2 == 0);
        constexpr Indextype Ncon = sizeof...(Is) / 2;
        static_assert(Ncon <= 4, "Contractions of more than 4 legs is currently not implemented for ArrayTensorLib");
        static_assert(Rank1 + Rank2 >= Ncon);

        constexpr Indextype Rank1i = static_cast<Indextype>(Rank1);
        constexpr Indextype Rank2i = static_cast<Indextype>(Rank2);

        using seq1 = seq::make<Indextype, Rank1i>; // 0, 1, ..., Rank1-1
        using seq2 = seq::make<Indextype, Rank2i, Rank1i>; // Rank1, Rank+1, ..., Rank1+Rank2-1
        using seqC = seq::iseq<Indextype, Is...>;

        if constexpr(Ncon == 0) {
            return contract_helper<Scalar, Rank1, Rank2>(T1, T2, seq1{}, seq2{}, seq::concat<seq1, seq2>{});
        } else {
            using mod_seq1 = seq_replace<Indextype, seq::at<0, seqC>, 100, seq1>;
            using mod_seq2 = seq_replace<Indextype, seq::at<1, seqC> + Rank1i, 100, seq2>;
            using rem_seq1 = seq::remove<seq::at<0, seqC>, seq1>;
            using rem_seq2 = seq::remove<seq::at<1, seqC> + Rank1i, seq2>;
            if constexpr(Ncon == 1) {
                return contract_helper<Scalar, Rank1, Rank2>(T1, T2, mod_seq1{}, mod_seq2{}, seq::concat<rem_seq1, rem_seq2>{});
            } else {
                using mod_seq11 = seq_replace<Indextype, seq::at<2, seqC>, 101, mod_seq1>;
                using mod_seq22 = seq_replace<Indextype, seq::at<3, seqC> + Rank1i, 101, mod_seq2>;
                using rem_seq11 = seq::remove<seq::at<2, seqC>, rem_seq1>;
                using rem_seq22 = seq::remove<seq::at<3, seqC> + Rank1i, rem_seq2>;
                if constexpr(Ncon == 2) {
                    return contract_helper<Scalar, Rank1, Rank2>(T1, T2, mod_seq11{}, mod_seq22{}, seq::concat<rem_seq11, rem_seq22>{});
                } else {
                    using mod_seq111 = seq_replace<Indextype, seq::at<4, seqC>, 102, mod_seq11>;
                    using mod_seq222 = seq_replace<Indextype, seq::at<5, seqC> + Rank1i, 102, mod_seq22>;
                    using rem_seq111 = seq::remove<seq::at<4, seqC>, rem_seq11>;
                    using rem_seq222 = seq::remove<seq::at<5, seqC> + Rank1i, rem_seq22>;
                    if constexpr(Ncon == 3) {
                        return contract_helper<Scalar, Rank1, Rank2>(T1, T2, mod_seq111{}, mod_seq222{}, seq::concat<rem_seq111, rem_seq222>{});
                    } else {
                        using mod_seq1111 = seq_replace<Indextype, seq::at<6, seqC>, 103, mod_seq111>;
                        using mod_seq2222 = seq_replace<Indextype, seq::at<7, seqC> + Rank1i, 103, mod_seq222>;
                        using rem_seq1111 = seq::remove<seq::at<6, seqC>, rem_seq111>;
                        using rem_seq2222 = seq::remove<seq::at<7, seqC> + Rank1i, rem_seq222>;
                        if constexpr(Ncon == 4) {
                            return contract_helper<Scalar, Rank1, Rank2>(
                                T1, T2, mod_seq1111{}, mod_seq2222{}, seq::concat<rem_seq1111, rem_seq2222>{});
                        }
                    }
                }
            }
        }
    }

    template <typename Expr, Indextype... p>
    static auto shuffle_view(const Expr& T)
    {
        // static_assert(Rank == sizeof...(p));
        return nda::transpose<p...>(T);
    }

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(const TType<Scalar, Rank>& T)
    {
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> perm = {p...};
        std::array<Indextype, Rank> dims;
        for(std::size_t r = 0; r < Rank; r++) { dims[r] = T.shape().dim(perm[r]).extent(); }

        nda::dense_shape<Rank> shp(as_tuple(dims));
        return make_copy(nda::transpose<p...>(T), shp);
    }

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(const cMapTType<Scalar, Rank>& T)
    {
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> perm = {p...};
        std::array<Indextype, Rank> dims;
        for(std::size_t r = 0; r < Rank; r++) { dims[r] = T.shape().dim(perm[r]).extent(); }

        nda::dense_shape<Rank> shp(as_tuple(dims));
        return make_copy(nda::transpose<p...>(T), shp);
    }

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(const TType<Scalar, Rank>& T, seq::iseq<Indextype, p...> s)
    {
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> perm = {p...};
        std::array<Indextype, Rank> dims;
        for(std::size_t r = 0; r < Rank; r++) { dims[r] = T.shape().dim(perm[r]).extent(); }

        nda::dense_shape<Rank> shp(as_tuple(dims));
        return make_copy(nda::transpose<p...>(T), shp);
    }

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static auto shuffle_view(const cMapTType<Scalar, Rank>& T, seq::iseq<Indextype, p...> s)
    {
        static_assert(Rank == sizeof...(p));
        return nda::transpose<p...>(T);
    }

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static TType<Scalar, Rank2> reshape(const TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims)
    {
        cMapTType<Scalar, Rank2> map(T.data(), as_tuple(dims));
        return construct<Scalar, Rank2>(map);
    }

    // methods lvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static auto slice(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents)
    {
        std::array<nda::interval<>, Rank1> slices;
        for(std::size_t r = 0; r < Rank1; r++) { slices[r] = nda::interval<>(offsets[r], extents[r]); }
        return T(as_tuple(slices));
    }

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static void setSubTensor(TType<Scalar, Rank1>& T,
                             const std::array<Indextype, Rank2>& offsets,
                             const std::array<Indextype, Rank2>& extents,
                             const TType<Scalar, Rank1>& S)
    {
        cMapTType<Scalar, Rank1> Sm = S;
        std::array<nda::interval<>, Rank1> slices;
        for(std::size_t r = 0; r < Rank1; r++) { slices[r] = nda::interval<>(offsets[r], extents[r]); }
        nda::dense_shape<Rank1> new_s(as_tuple(slices));
        new_s.resolve();
        Sm.set_shape(new_s);
        nda::copy(Sm, T(as_tuple(slices)));
        // return T(as_tuple(slices));
    }

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static auto reshape(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims)
    {
        MapTType<Scalar, Rank2> map(T.data(), as_tuple(dims));
        return map;
    }

    template <typename Scalar, int Rank>
    static std::string print(const TType<Scalar, Rank>& T)
    {
        std::stringstream ss;
        ss << "Tensor";
        return ss.str();
    }
};

} // namespace Xped

#endif
