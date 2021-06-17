#ifndef PLAIN_INTERFACE_EIGEN_ARRAY_IMPL_H_
#define PLAIN_INTERFACE_EIGEN_ARRAY_IMPL_H_

template <>
struct PlainInterface<EigenMatrixLib, ArrayTensorLib>
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
    using Ttype = nda::dense_array<Scalar, Rank>;
    template <typename Scalar, std::size_t Rank>
    using cTtype = const nda::dense_array<Scalar, Rank>;

    template <typename Scalar, std::size_t Rank>
    using MapTtype = nda::dense_array_ref<Scalar, Rank>;
    template <typename Scalar, std::size_t Rank>
    using cMapTtype = nda::const_dense_array_ref<Scalar, Rank>;

    using Indextype = nda::index_t;

    template <typename Scalar>
    using MType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename Scalar>
    using cMType = const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    template <typename Scalar>
    using MapMtype = Eigen::Map<Mtype<Scalar>>;
    template <typename Scalar>
    using cMapMtype = Eigen::Map<cMtype<Scalar>>;
};
#endif
