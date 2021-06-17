#ifndef TENSOR_INTERFACE_EIGEN_IMPL_H_
#define TENSOR_INTERFACE_EIGEN_IMPL_H_

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

template <>
struct TensorInterface<EigenTensorLib>
{
    // typedefs
    template <typename Scalar, std::size_t Rank>
    using TType = Eigen::Tensor<Scalar, Rank>;
    template <typename Scalar, std::size_t Rank>
    using cTType = const Eigen::Tensor<Scalar, Rank>;

    template <typename Scalar, std::size_t Rank>
    using MapTType = Eigen::TensorMap<TType<Scalar, Rank>>;
    template <typename Scalar, std::size_t Rank>
    using cMapTType = Eigen::TensorMap<cTType<Scalar, Rank>>;

    // template<typename Scalar, std::size_t Rank> using Indextype = typename Eigen::Tensor<Scalar,Rank>::Index;
    using Indextype = Eigen::Index;

    // constructors
    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank> construct(const std::array<Indextype, Rank>& dims)
    {
        return TType<Scalar, Rank>(dims);
    }

    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> construct(const MapTType<Scalar, Rank>& map)
    {
        return TType<Scalar, Rank>(map);
    }

    template <typename Scalar, int Rank>
    static cTType<Scalar, Rank> construct(const cMapTType<Scalar, Rank>& map)
    {
        return cTType<Scalar, Rank>(map);
    }

    // map constructors
    template <typename Scalar, std::size_t Rank>
    static cMapTType<Scalar, Rank> cMap(const Scalar* data, const std::array<Indextype, Rank>& dims)
    {
        return cMapTType<Scalar, Rank>(data, dims);
    }

    template <typename Scalar, std::size_t Rank>
    static MapTType<Scalar, Rank> Map(Scalar* data, const std::array<Indextype, Rank>& dims)
    {
        return MapTType<Scalar, Rank>(data, dims);
    }

    // initialization
    template <typename Scalar, int Rank>
    static void setZero(TType<Scalar, Rank>& T)
    {
        T.setZero();
    }

    template <typename Scalar, int Rank>
    static void setRandom(TType<Scalar, Rank>& T)
    {
        T.setRandom();
    }

    template <typename Scalar, int Rank>
    static void setConstant(TType<Scalar, Rank>& T, const Scalar& val)
    {
        T.setConstant(val);
    }

    template <typename Scalar, int Rank>
    static void setVal(TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index, const Scalar& val)
    {
        T(index) = val;
    }

    template <typename Scalar, int Rank>
    static Scalar getVal(const TType<Scalar, Rank>& T, const std::array<Indextype, Rank>& index)
    {
        if constexpr(Rank == 0) {
            return T(0);
        } else {
            return T(index);
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
    template <typename Scalar, int Rank>
    static std::array<Indextype, Rank> dimensions(const TType<Scalar, Rank>& T)
    {
        std::array<Indextype, Rank> out;
        auto dims = T.dimensions();
        std::copy(dims.cbegin(), dims.cend(), out.begin());
        return out;
    }

    // tensorProd
    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> tensorProd(const TType<Scalar, Rank>& T1, const TType<Scalar, Rank>& T2)
    {
        typedef TType<Scalar, Rank> TensorType;
        typedef Indextype Index;
        std::array<Index, TensorType::NumIndices> dims;
        for(Index i = 0; i < T1.rank(); i++) { dims[i] = T1.dimensions()[i] * T2.dimensions()[i]; }
        TensorType res(dims);
        res.setZero();
        std::array<Index, TensorType::NumIndices> extents = T2.dimensions();

        std::vector<std::size_t> vec_dims;
        for(const auto& d : T1.dimensions()) { vec_dims.push_back(d); }
        NestedLoopIterator Nelly(T1.rank(), vec_dims);

        for(std::size_t i = Nelly.begin(); i != Nelly.end(); i++) {
            std::array<Index, TensorType::NumIndices> indices;
            for(Index j = 0; j < T1.rank(); j++) { indices[j] = Nelly(j); }
            std::array<Index, TensorType::NumIndices> offsets;
            for(Index i = 0; i < T1.rank(); i++) { offsets[i] = indices[i] * T2.dimensions()[i]; }

            res.slice(offsets, extents) = T1(indices) * T2;
            ++Nelly;
        }
        return res;
    }

    template <typename Scalar, std::size_t Rank, typename Expr1, typename Expr2>
    static void addScale(const Expr1& src, Expr2& dst, const Scalar& scale)
    {
        dst += scale * src;
    }

    // methods rvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
    static TType<Scalar, Rank1 + Rank2 - sizeof...(Is)> contract(const TType<Scalar, Rank1>& T1, const TType<Scalar, Rank2>& T2)
    {
        static_assert(sizeof...(Is) % 2 == 0);
        constexpr Indextype Ncon = sizeof...(Is) / 2;
        static_assert(Rank1 + Rank2 >= Ncon);
        static_assert(Ncon <= 4, "Contractions of more than 4 legs is currently not implemented for EigenTensorLib");

        using seqC = seq::iseq<Indextype, Is...>;
        if constexpr(Ncon == 1) {
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
        }
    }

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(const TType<Scalar, Rank>& T)
    {
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> dims = {p...};
        return TType<Scalar, Rank>(T.shuffle(dims));
    }

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(const TType<Scalar, Rank>& T, seq::iseq<Indextype, p...> s)
    {
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> dims = {p...};
        return TType<Scalar, Rank>(T.shuffle(dims));
    }

    template <typename Expr, Indextype... p>
    static auto shuffle_view(const Expr& T)
    {
        constexpr Indextype Rank = Eigen::internal::traits<Expr>::NumDimensions;
        static_assert(Rank == sizeof...(p));
        std::array<Indextype, Rank> dims = {p...};
        return T.shuffle(dims);
    }

    // template<typename Scalar, std::size_t Rank, Indextype... p>
    // static auto shuffle_view(const cMapTType<Scalar,Rank>& T)
    // {
    //         static_assert(Rank == sizeof...(p));
    //         std::array<Indextype, Rank> dims = {p...};
    //         return T.shuffle(dims);
    // }

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static TType<Scalar, Rank2> reshape(const TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims)
    {
        return T.reshape(dims);
    }

    // methods lvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static void setSubTensor(TType<Scalar, Rank1>& T,
                             const std::array<Indextype, Rank2>& offsets,
                             const std::array<Indextype, Rank2>& extents,
                             const TType<Scalar, Rank1>& S)
    {
        T.slice(offsets, extents) = S;
    }

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static auto slice(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents)
    {
        return T.slice(offsets, extents);
    }

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static auto reshape(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims)
    {
        return T.reshape(dims);
    }
};

#endif
