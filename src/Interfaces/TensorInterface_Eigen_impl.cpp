#include "Xped/Interfaces/TensorInterface.hpp"

#include "Xped/Interfaces/TensorInterface_Eigen_impl.hpp"

namespace Xped {

template <typename Scalar, std::size_t Rank>
using TType = Eigen::Tensor<Scalar, Rank>;
template <typename Scalar, std::size_t Rank>
using cTType = const Eigen::Tensor<Scalar, Rank>;

template <typename Scalar, std::size_t Rank>
using MapTType = Eigen::TensorMap<TType<Scalar, Rank>>;
template <typename Scalar, std::size_t Rank>
using cMapTType = Eigen::TensorMap<cTType<Scalar, Rank>>;

using Indextype = Eigen::Index;

// constructors
template <typename Scalar, std::size_t Rank>
TType<Scalar, Rank> TensorInterface::construct(const std::array<Indextype, Rank>& dims, const mpi::XpedWorld&)
{
    return TType<Scalar, Rank>(dims);
}

template <typename Scalar, int Rank>
TType<Scalar, Rank> TensorInterface::construct(const MapTType<Scalar, Rank>& map)
{
    return TType<Scalar, Rank>(map);
}

template <typename Scalar, int Rank>
cTType<Scalar, Rank> TensorInterface::construct(const cMapTType<Scalar, Rank>& map)
{
    return cTType<Scalar, Rank>(map);
}

// map constructors
template <typename Scalar, std::size_t Rank>
cMapTType<Scalar, Rank> TensorInterface::cMap(const Scalar* data, const std::array<Indextype, Rank>& dims)
{
    return cMapTType<Scalar, Rank>(data, dims);
}

template <typename Scalar, std::size_t Rank>
MapTType<Scalar, Rank> TensorInterface::Map(Scalar* data, const std::array<Indextype, Rank>& dims)
{
    return MapTType<Scalar, Rank>(data, dims);
}

// initialization
template <typename Scalar, int Rank>
void TensorInterface::setZero(TType<Scalar, Rank>& T)
{
    T.setZero();
}

template <typename Scalar, int Rank>
void TensorInterface::setRandom(TType<Scalar, Rank>& T)
{
    T.setRandom();
}

template <typename Scalar, int Rank>
void TensorInterface::setConstant(TType<Scalar, Rank>& T, const Scalar& val)
{
    T.setConstant(val);
}

template <typename Scalar, int Rank>
void TensorInterface::setVal(TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index, const Scalar& val)
{
    T(index) = val;
}

template <typename Scalar, int Rank>
Scalar TensorInterface::getVal(const TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index)
{
    if constexpr(Rank == 0) {
        return T(0);
    } else {
        return T(index);
    }
}

// raw data
template <typename Scalar, int Rank>
const Scalar* TensorInterface::get_raw_data(const TType<Scalar, Rank>& T)
{
    return T.data();
}

template <typename Scalar, int Rank>
Scalar* TensorInterface::get_raw_data(TType<Scalar, Rank>& T)
{
    return T.data();
}

// shape info
template <typename Scalar, int Rank>
std::array<Indextype, Rank> TensorInterface::dimensions(const TType<Scalar, Rank>& T)
{
    std::array<Indextype, Rank> out;
    auto dims = T.dimensions();
    std::copy(dims.cbegin(), dims.cend(), out.begin());
    return out;
}

// tensorProd
template <typename Scalar, int Rank>
TType<Scalar, Rank> TensorInterface::tensorProd(const TType<Scalar, Rank>& T1, const TType<Scalar, Rank>& T2)
{
    std::array<Indextype, 2 * Rank> tmp_dims;
    std::array<Indextype, Rank> out_dims;
    auto dim1 = T1.dimensions();
    auto dim2 = T2.dimensions();
    for(std::size_t i = 0; i < Rank; i++) {
        tmp_dims[i] = dim2[i];
        tmp_dims[i + Rank] = dim1[i];
        out_dims[i] = dim1[i] * dim2[i];
    }
    TType<Scalar, 2 * Rank> tmp(tmp_dims);

    std::array<Eigen::IndexPair<Eigen::Index>, 0> con{};
    tmp = T2.contract(T1, con);

    TType<Scalar, Rank> res(out_dims);
    std::array<Indextype, 2 * Rank> shuffle_dims;
    for(std::size_t i = 0; i < Rank; i++) {
        shuffle_dims[2 * i] = i;
        shuffle_dims[2 * i + 1] = Rank + i;
    }
    res = tmp.shuffle(shuffle_dims).reshape(out_dims);

    return res;

    // typedef TType<Scalar, Rank> TensorType;
    // typedef Indextype Index;
    // std::array<Index, TensorType::NumIndices> dims;
    // for(Index i = 0; i < T1.rank(); i++) { dims[i] = T1.dimensions()[i] * T2.dimensions()[i]; }
    // TensorType res(dims);
    // res.setZero();
    // std::array<Index, TensorType::NumIndices> extents = T2.dimensions();

    // std::vector<std::size_t> vec_dims;
    // for(const auto& d : T1.dimensions()) { vec_dims.push_back(d); }
    // NestedLoopIterator Nelly(T1.rank(), vec_dims);

    // for(std::size_t i = Nelly.begin(); i != Nelly.end(); i++) {
    //     std::array<Index, TensorType::NumIndices> indices;
    //     for(Index j = 0; j < T1.rank(); j++) { indices[j] = Nelly(j); }
    //     std::array<Index, TensorType::NumIndices> offsets;
    //     for(Index i = 0; i < T1.rank(); i++) { offsets[i] = indices[i] * T2.dimensions()[i]; }

    //     res.slice(offsets, extents) = T1(indices) * T2;
    //     ++Nelly;
    // }
    // return res;
}

template <typename Scalar, std::size_t Rank, typename Expr1, typename Expr2>
void TensorInterface::addScale(const Expr1& src, Expr2& dst, const Scalar& scale)
{
    dst += scale * src;
}

// methods rvalue
template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
TType<Scalar, Rank1 + Rank2 - sizeof...(Is)> TensorInterface::contract(const TType<Scalar, Rank1>& T1, const TType<Scalar, Rank2>& T2)
{
    static_assert(sizeof...(Is) % 2 == 0);
    constexpr Indextype Ncon = sizeof...(Is) / 2;
    static_assert(Rank1 + Rank2 >= Ncon);
    static_assert(Ncon <= 5, "Contractions of more than 5 legs is currently not implemented for EigenTensorLib");

    using seqC = seq::iseq<Indextype, Is...>;
    if constexpr(Ncon == 0) {
        constexpr std::array<Eigen::IndexPair<Indextype>, Ncon> con{};
        return T1.contract(T2, con);
    } else if constexpr(Ncon == 1) {
        // constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>)};
        constexpr Eigen::IndexPairList<Eigen::type2indexpair<seq::at<0, seqC>, seq::at<1, seqC>>> con;
        return T1.contract(T2, con);
    } else if constexpr(Ncon == 2) {
        // constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>),
        //                                                                               Eigen::IndexPair<Indextype >(seq::at<2, seqC>, seq::at<3,
        //                                                                               seqC>)};
        constexpr Eigen::IndexPairList<Eigen::type2indexpair<seq::at<0, seqC>, seq::at<1, seqC>>,
                                       Eigen::type2indexpair<seq::at<2, seqC>, seq::at<3, seqC>>>
            con;
        return T1.contract(T2, con);
    } else if constexpr(Ncon == 3) {
        // constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>),
        //                                                                               Eigen::IndexPair<Indextype >(seq::at<2, seqC>, seq::at<3,
        //                                                                               seqC>), Eigen::IndexPair<Indextype >(seq::at<4, seqC>,
        //                                                                               seq::at<5, seqC>)};
        constexpr Eigen::IndexPairList<Eigen::type2indexpair<seq::at<0, seqC>, seq::at<1, seqC>>,
                                       Eigen::type2indexpair<seq::at<2, seqC>, seq::at<3, seqC>>,
                                       Eigen::type2indexpair<seq::at<4, seqC>, seq::at<5, seqC>>>
            con;
        return T1.contract(T2, con);
    } else if constexpr(Ncon == 4) {
        // constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {
        //         Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>),
        //         Eigen::IndexPair<Indextype >(seq::at<2, seqC>, seq::at<3, seqC>),
        //         Eigen::IndexPair<Indextype >(seq::at<4, seqC>, seq::at<5, seqC>),
        //         Eigen::IndexPair<Indextype >(seq::at<6, seqC>, seq::at<7, seqC>)
        // };
        constexpr Eigen::IndexPairList<Eigen::type2indexpair<seq::at<0, seqC>, seq::at<1, seqC>>,
                                       Eigen::type2indexpair<seq::at<2, seqC>, seq::at<3, seqC>>,
                                       Eigen::type2indexpair<seq::at<4, seqC>, seq::at<5, seqC>>,
                                       Eigen::type2indexpair<seq::at<6, seqC>, seq::at<7, seqC>>>
            con;
        return T1.contract(T2, con);
    } else /*if constexpr(Ncon == 5)*/ {
        constexpr Eigen::IndexPairList<Eigen::type2indexpair<seq::at<0, seqC>, seq::at<1, seqC>>,
                                       Eigen::type2indexpair<seq::at<2, seqC>, seq::at<3, seqC>>,
                                       Eigen::type2indexpair<seq::at<4, seqC>, seq::at<5, seqC>>,
                                       Eigen::type2indexpair<seq::at<6, seqC>, seq::at<7, seqC>>,
                                       Eigen::type2indexpair<seq::at<8, seqC>, seq::at<9, seqC>>>
            con;
        return T1.contract(T2, con);
    }
}

template <typename Scalar, std::size_t Rank, Indextype... p>
TType<Scalar, Rank> TensorInterface::shuffle(const TType<Scalar, Rank>& T)
{
    static_assert(Rank == sizeof...(p));
    std::array<Indextype, Rank> dims = {p...};
    return TType<Scalar, Rank>(T.shuffle(dims));
}

template <typename Scalar, std::size_t Rank, Indextype... p>
TType<Scalar, Rank> TensorInterface::shuffle(const TType<Scalar, Rank>& T, seq::iseq<Indextype, p...> s)
{
    static_assert(Rank == sizeof...(p));
    std::array<Indextype, Rank> dims = {p...};
    return TType<Scalar, Rank>(T.shuffle(dims));
}

template <typename Expr, Indextype... p>
const Eigen::TensorShufflingOp<const std::array<Indextype, Eigen::internal::traits<Expr>::NumDimensions>, const Expr>
TensorInterface::shuffle_view(const Expr& T)
{
    constexpr Indextype Rank = Eigen::internal::traits<Expr>::NumDimensions;
    static_assert(Rank == sizeof...(p));
    std::array<Indextype, Rank> dims = {p...};
    return T.shuffle(dims);
}

// template<typename Scalar, std::size_t Rank, Indextype... p>
//  auto shuffle_view(const cMapTType<Scalar,Rank>& T)
// {
//         static_assert(Rank == sizeof...(p));
//         std::array<Indextype, Rank> dims = {p...};
//         return T.shuffle(dims);
// }

template <typename Scalar, int Rank1, std::size_t Rank2>
TType<Scalar, Rank2> TensorInterface::reshape(const TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims)
{
    return T.reshape(dims);
}

// methods lvalue
template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
void TensorInterface::setSubTensor(TType<Scalar, Rank1>& T,
                                   const std::array<Indextype, Rank2>& offsets,
                                   const std::array<Indextype, Rank2>& extents,
                                   const TType<Scalar, Rank1>& S)
{
    T.slice(offsets, extents) = S;
}

template <typename Scalar, int Rank1, std::size_t Rank2>
const Eigen::TensorSlicingOp<const std::array<Indextype, Rank2>, const std::array<Indextype, Rank2>, const TType<Scalar, Rank1>>
TensorInterface::slice(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents)
{
    return T.slice(offsets, extents);
}

template <typename Scalar, int Rank1, std::size_t Rank2>
const Eigen::TensorReshapingOp<const std::array<Indextype, Rank2>, const TType<Scalar, Rank1>>
TensorInterface::reshape(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims)
{
    return T.reshape(dims);
}

template <typename Scalar, int Rank>
std::string TensorInterface::print(const TType<Scalar, Rank>& T)
{
    std::stringstream ss;
    ss << T;
    return ss.str();
}

} // namespace Xped
