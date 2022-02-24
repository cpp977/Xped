#ifndef TENSOR_INTERFACE_CYCLOPS_IMPL_H_
#define TENSOR_INTERFACE_CYCLOPS_IMPL_H_

#include <ctf.hpp>

namespace Xped {

struct TensorInterface
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
    static TType<Scalar, Rank> construct(const std::array<Indextype, Rank>& dims, CTF::World& world);

    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> construct(const MapTType<Scalar, Rank>& map);

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
    static void setZero(TType<Scalar, Rank>& T);

    template <typename Scalar, int Rank>
    static void setRandom(TType<Scalar, Rank>& T);

    template <typename Scalar, int Rank>
    static void setConstant(TType<Scalar, Rank>& T, const Scalar& val);

    template <typename Scalar, int Rank>
    static void setVal(TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index, const Scalar& val);

    template <typename Scalar, int Rank>
    static Scalar getVal(const TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index);

    // raw data
    template <typename Scalar, int Rank>
    static Scalar* get_raw_data(const TType<Scalar, Rank>& T);

    // shape info
    template <typename Scalar, int Rank>
    static std::array<Indextype, Rank> dimensions(const TType<Scalar, Rank>& T);

    // tensorProd
    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> tensorProd(TType<Scalar, Rank>& T1, TType<Scalar, Rank>& T2);

    template <typename Scalar, std::size_t Rank, typename Expr1, typename Expr2>
    static void addScale(const Expr1& src, Expr2& dst, const Scalar& scale);

    // methods rvalue

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is1, Indextype... Is2, Indextype... Ist>
    static TType<Scalar, sizeof...(Ist)> contract_helper(TType<Scalar, Rank1>& T1,
                                                         TType<Scalar, Rank2>& T2,
                                                         seq::iseq<Indextype, Is1...> S1,
                                                         seq::iseq<Indextype, Is2...> S2,
                                                         seq::iseq<Indextype, Ist...> St);

    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
    static TType<Scalar, Rank1 + Rank2 - sizeof...(Is)> contract(TType<Scalar, Rank1>& T1, TType<Scalar, Rank2>& T2);

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(TType<Scalar, Rank>& T);

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(TType<Scalar, Rank>& T, seq::iseq<Indextype, p...> s);

    template <typename Expr, Indextype... p>
    static Expr shuffle_view(const Expr& T);

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static TType<Scalar, Rank2> reshape(const TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims);

    // methods lvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static void setSubTensor(TType<Scalar, Rank1>& T,
                             const std::array<Indextype, Rank2>& offsets,
                             const std::array<Indextype, Rank2>& extents,
                             const TType<Scalar, Rank1>& S);

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static TType<Scalar, Rank1>
    slice(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents);

    template <typename Scalar, std::size_t Rank>
    static std::string print(const TType<Scalar, Rank>& T);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/TensorInterface_Cyclops_impl.cpp"
#endif

#endif
