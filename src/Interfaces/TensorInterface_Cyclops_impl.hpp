#ifndef TENSOR_INTERFACE_CYCLOPS_IMPL_H_
#define TENSOR_INTERFACE_CYCLOPS_IMPL_H_

#include <ctf.hpp>

template <typename Index, Index oldVal, Index newVal, typename S>
using seq_replace = seq::insert<seq::index_of<oldVal, S>, newVal, seq::remove<oldVal, S>>;

template <>
struct TensorInterface<CyclopsTensorLib>
{
    // typedefs
    template <typename Scalar, std::size_t Rank>
    using TType = CTF::Tensor<Scalar>;
    template <typename Scalar, std::size_t Rank>
    using cTType = const CTF::Tensor<Scalar>;

    template <typename Scalar, std::size_t Rank>
    using MapTType = CTF::Tensor<Scalar>;
    template <typename Scalar, std::size_t Rank>
    using cMapTType = const CTF::Tensor<Scalar>;

    using Indextype = int;

    // utility functions
    static constexpr char idx(const Indextype& i)
    {
        if(i == 0) { return 'i'; }
        if(i == 1) { return 'j'; }
        if(i == 2) { return 'k'; }
        if(i == 3) { return 'l'; }
        if(i == 4) { return 'm'; }
        if(i == 5) { return 'n'; }
        if(i == 6) { return 'o'; }
        if(i == 100) { return 'z'; }
        if(i == 101) { return 'y'; }
        if(i == 102) { return 'x'; }
        if(i == 103) { return 'w'; }
        if(i == 104) { return 'v'; }
        if(i == 105) { return 'u'; }
        if(i == 106) { return 't'; }
        assert(false and "Invalid usage of idx() for CTF tensors.");
        return 'e';
    }

    template <std::size_t Rank>
    static constexpr std::array<char, Rank> get_idx(std::size_t shift = 0)
    {
        std::array<char, Rank> out;
        for(std::size_t i = 0; i < Rank; i++) { out[i] = idx(i + shift); }
        return out;
    }

    // constructors
    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> construct(const std::array<Indextype, Rank>& dims, CTF::World world = CTF::get_universe())
    {
        return TType<Scalar, Rank>(Rank, dims.data());
    }

    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> construct(const MapTType<Scalar, Rank>& map)
    {
        return TType<Scalar, Rank>(map);
    }

    // template <typename Scalar, int Rank>
    // static cTType<Scalar, Rank> construct(const cMapTType<Scalar, Rank>& map)
    // {
    //     return cTType<Scalar, Rank>(map);
    // }

    // map constructors
    // template <typename Scalar, std::size_t Rank>
    // static cMapTType<Scalar, Rank> cMap(const Scalar* data, const std::array<Indextype, Rank>& dims)
    // {
    //     MapTType<Scalar, Rank> out(Rank, dims.data());
    //     int64_t nvals;
    //     int64_t* indices;
    //     Scalar* new_data;
    //     out.get_local_data(&nvals, &indices, &new_data);
    //     for(int i = 0; i < nvals; i++) { new_data[i] = data[indices[i]]; }
    //     out.write(nvals, indices, new_data);
    //     free(indices);
    //     delete[] new_data;
    //     return out;
    // }

    // template <typename Scalar, std::size_t Rank>
    // static MapTType<Scalar, Rank> Map(Scalar* data, const std::array<Indextype, Rank>& dims)
    // {
    //     MapTType<Scalar, Rank> out(Rank, dims.data());
    //     int64_t nvals;
    //     int64_t* indices;
    //     Scalar* new_data;
    //     out.get_local_data(&nvals, &indices, &new_data);
    //     for(int i = 0; i < nvals; i++) { new_data[i] = data[indices[i]]; }
    //     out.write(nvals, indices, new_data);
    //     free(indices);
    //     delete[] new_data;
    //     return out;
    // }

    // initialization
    template <typename Scalar, int Rank>
    static void setZero(TType<Scalar, Rank>& T)
    {
        T[get_idx<Rank>().data()] = 0.;
    }

    template <typename Scalar, int Rank>
    static void setRandom(TType<Scalar, Rank>& T)
    {
        T.fill_random(-1.0, 1.0);
    }

    template <typename Scalar, int Rank>
    static void setConstant(TType<Scalar, Rank>& T, const Scalar& val)
    {
        T[get_idx<Rank>().data()] = val;
    }

    template <typename Scalar, int Rank>
    static void setVal(TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index, const Scalar& val)
    {
        int64_t global_idx = 0;
        for(std::size_t i = 0; i < Rank; i++) {
            int64_t factor = 1;
            for(std::size_t j = 0; j < i; j++) { factor *= T.lens[j]; }
            global_idx += index[i] * factor;
        }

        int64_t nvals;
        int64_t* indices;
        Scalar* data;
        T.get_local_data(&nvals, &indices, &data);
        for(int i = 0; i < nvals; i++) {
            if(indices[i] == global_idx) { data[i] = val; }
        }
        T.write(nvals, indices, data);
        free(indices);
        delete[] data;
    }

    template <typename Scalar, int Rank>
    static Scalar getVal(const TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index)
    {
        if constexpr(Rank == 0) {
            Scalar out = 0.;

            int64_t nvals;
            Scalar* data;
            T.get_all_data(&nvals, &data);
            out = data[0];
            delete[] data;
            return out;
        }
        int64_t global_idx = 0;
        for(std::size_t i = 0; i < Rank; i++) {
            int64_t factor = 1;
            for(std::size_t j = 0; j < i; j++) { factor *= T.lens[j]; }
            global_idx += index[i] * factor;
        }

        Scalar out = 0.;

        int64_t nvals;
        int64_t* indices;
        Scalar* data;
        T.get_local_data(&nvals, &indices, &data);
        for(int i = 0; i < nvals; i++) {
            if(indices[i] == global_idx) { out = data[i]; }
        }
        free(indices);
        delete[] data;
        return out;
    }

    // raw data
    template <typename Scalar, int Rank>
    static Scalar* get_raw_data(const TType<Scalar, Rank>& T)
    {
        int64_t nvals;
        int64_t* indices;
        Scalar* data;
        T.get_local_data(&nvals, &indices, &data);
        free(indices);
        return data;
    }

    // shape info
    template <typename Scalar, int Rank>
    static std::array<Indextype, Rank> dimensions(const TType<Scalar, Rank>& T)
    {
        std::array<Indextype, Rank> out;
        for(std::size_t i = 0; i < Rank; i++) { out[i] = T.lens[i]; }
        return out;
    }

    // tensorProd
    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> tensorProd(TType<Scalar, Rank>& T1, TType<Scalar, Rank>& T2)
    {
        std::array<Indextype, 2 * Rank> tmp_dims;
        std::array<Indextype, Rank> out_dims;
        for(std::size_t i = 0; i < Rank; i++) {
            tmp_dims[2 * i] = T2.lens[i];
            tmp_dims[2 * i + 1] = T1.lens[i];
            out_dims[i] = T1.lens[i] * T2.lens[i];
        }
        TType<Scalar, 2 * Rank> tmp(2 * Rank, tmp_dims.data(), *T1.wrld);
        auto T2_idx = get_idx<Rank>();
        auto T1_idx = get_idx<Rank>(100);
        std::array<char, 2 * Rank> tot_idx;
        for(std::size_t i = 0; i < Rank; i++) {
            tot_idx[2 * i] = T2_idx[i];
            tot_idx[2 * i + 1] = T1_idx[i];
        }
        tmp[tot_idx.data()] = T2[T2_idx.data()] * T1[T1_idx.data()];

        TType<Scalar, Rank> out(Rank, out_dims.data(), *T1.wrld);
        int64_t nvals;
        int64_t* indices;
        Scalar* data;
        tmp.get_local_data(&nvals, &indices, &data);
        out.write(nvals, indices, data);
        free(indices);
        delete[] data;
        return out;
        // typedef Indextype Index;
        // std::array<Index, Rank> dims;
        // for(Index i = 0; i < Rank; i++) { dims[i] = T1.lens[i] * T2.lens[i]; }
        // TType<Scalar, Rank> res(Rank, dims.data());
        // std::array<Index, Rank> extents;
        // std::copy(std::begin(T2.lens), std::end(T2.lens), std::begin(extents));

        // std::vector<std::size_t> vec_dims;
        // for(const auto& d : T1.lens) { vec_dims.push_back(d); }
        // NestedLoopIterator Nelly(Rank, vec_dims);

        // for(std::size_t i = Nelly.begin(); i != Nelly.end(); i++) {
        //     std::array<Index, Rank> indices;
        //     for(Index j = 0; j < Rank; j++) { indices[j] = Nelly(j); }
        //     std::array<Index, Rank> offsets;
        //     for(Index i = 0; i < T1.rank(); i++) { offsets[i] = indices[i] * T2.lens[i]; }
        //     std::array<Indextype, Rank> ends;
        //     for(Index i = 0; i < Rank; i++) { ends[i] = offsets[i] + extents[i]; }
        //     int64_t global_idx = 0;
        //     for(std::size_t i = 0; i < Rank; i++) {
        //         global_idx += indices[i] * std::accumulate(std::begin(T1.lens), std::begin(T1.lens) + i, 1, std::multiplies<int64_t>);
        //     }
        //     std::vector<int64_t> global_idxs(1, global_idx);
        //     auto Tval = T1[global_idxs];
        //     int64_t nvals;
        //     int64_t* indices;
        //     Scalar* data;
        //     Tval.get_local_data(&nvals, &indices, &data);
        //     Scalar* val = *data;
        //     free(indices);
        //     delete[] data;
        //     res.slice(offsets.data(), end.data()) = T2.scale(get_idx<Rank>().data(), val);
        //     ++Nelly;
        // }
        // return res;
    }

    template <typename Scalar, std::size_t Rank, typename Expr1, typename Expr2>
    static void addScale(const Expr1& src, Expr2& dst, const Scalar& scale)
    {
        dst[get_idx<Rank>().data()] += src.scale[scale, get_idx<Rank>().data()];
    }

    // methods rvalue

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is1, Indextype... Is2, Indextype... Ist>
    static auto contract_helper(TType<Scalar, Rank1>& T1,
                                TType<Scalar, Rank2>& T2,
                                seq::iseq<Indextype, Is1...> S1,
                                seq::iseq<Indextype, Is2...> S2,
                                seq::iseq<Indextype, Ist...> St)
    {
        char idx_T1[Rank1] = {idx(Is1)...};
        char idx_T2[Rank2] = {idx(Is2)...};
        char idx_res[sizeof...(Ist)] = {idx(Ist)...};
        std::array<Indextype, sizeof...(Ist)> res_i = {Ist...};

        int lens[sizeof...(Ist)];
        for(std::size_t i = 0; i < sizeof...(Ist); i++) {
            if(res_i[i] < Rank1) {
                lens[i] = T1.lens[res_i[i]];
            } else {
                lens[i] = T2.lens[res_i[i] - Rank1];
            }
        }

        TType<Scalar, sizeof...(Ist)> res(sizeof...(Ist), lens, *T1.wrld);

        res[idx_res] = T1[idx_T1] * T2[idx_T2];
        return res;
    }

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
    static TType<Scalar, Rank1 + Rank2 - sizeof...(Is)> contract(TType<Scalar, Rank1>& T1, TType<Scalar, Rank2>& T2)
    {
        static_assert(sizeof...(Is) % 2 == 0);
        constexpr Indextype Ncon = sizeof...(Is) / 2;
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

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(TType<Scalar, Rank>& T)
    {
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> perm = {p...};
        std::array<Indextype, Rank> out_dims;
        for(std::size_t i = 0; i < Rank; i++) { out_dims[i] = T.lens[perm[i]]; }

        char perm_idx[Rank] = {idx(p)...};
        auto id_idx = get_idx<Rank>();

        TType<Scalar, Rank> out(Rank, out_dims.data(), *T.wrld);
        out[perm_idx] = T[id_idx.data()];
        return out;
    }

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(TType<Scalar, Rank>& T, seq::iseq<Indextype, p...> s)
    {
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> perm = {p...};
        std::array<Indextype, Rank> out_dims;
        for(std::size_t i = 0; i < Rank; i++) { out_dims[i] = T.lens[perm[i]]; }

        char perm_idx[Rank] = {idx(p)...};
        auto id_idx = get_idx<Rank>();

        TType<Scalar, Rank> out(Rank, out_dims.data(), *T.wrld);
        out[perm_idx] = T[id_idx.data()];
        return out;
    }

    template <typename Expr, Indextype... p>
    static auto shuffle_view(const Expr& T)
    {
        assert(false and "Shuffle view is not supported with CTF tensor lib");
    }

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static TType<Scalar, Rank2> reshape(const TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims)
    {
        TType<Scalar, Rank2> out(Rank2, dims.data(), *T.wrld);
        int64_t nvals;
        int64_t* indices;
        Scalar* new_data;
        T.get_local_data(&nvals, &indices, &new_data);
        out.write(nvals, indices, new_data);
        free(indices);
        delete[] new_data;
        return out;
    }

    // methods lvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static void setSubTensor(TType<Scalar, Rank1>& T,
                             const std::array<Indextype, Rank2>& offsets,
                             const std::array<Indextype, Rank2>& extents,
                             const TType<Scalar, Rank1>& S)
    {
        static_assert(Rank1 == Rank2);

        std::array<Indextype, Rank2> ends;
        for(std::size_t i = 0; i < Rank2; i++) { ends[i] = offsets[i] + extents[i]; }
        std::array<Indextype, Rank2> offsets_S;
        std::array<Indextype, Rank2> ends_S;
        for(std::size_t i = 0; i < Rank2; i++) {
            offsets_S[i] = 0;
            ends_S[i] = S.lens[i];
        }
        T.slice(offsets.data(), ends.data(), 0., S, offsets_S.data(), ends_S.data(), 1.);
    }

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static auto slice(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents)
    {
        std::array<Indextype, Rank2> ends;
        for(std::size_t i = 0; i < Rank2; i++) { ends[i] = offsets[i] + extents[i]; }
        return T.slice(offsets.data(), ends.data());
    }

    template <typename Scalar, std::size_t Rank>
    static std::string print(const TType<Scalar, Rank>& T)
    {
        return T.print();
    }
};

#endif