#ifndef TENSOR_INTERFACE_EIGEN_IMPL_H_
#define TENSOR_INTERFACE_EIGEN_IMPL_H_

#include "yas/serialize.hpp"

#include <Eigen/Dense>
#include <seq/seq.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/Permutations.hpp"

namespace Xped {

struct TensorInterface
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
    static TType<Scalar, Rank> construct(const std::array<Indextype, Rank>& dims, const mpi::XpedWorld& = mpi::getUniverse());

    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> construct(const MapTType<Scalar, Rank>& map);

    template <typename Scalar, int Rank>
    static cTType<Scalar, Rank> construct(const cMapTType<Scalar, Rank>& map);

    template <typename Scalar, std::size_t Rank>
    static TType<Scalar, Rank>
    construct_permutation(const std::array<Indextype, Rank / 2>& dims, const util::Permutation& p, const mpi::XpedWorld& = mpi::getUniverse());

    // map constructors
    template <typename Scalar, std::size_t Rank>
    static cMapTType<Scalar, Rank> cMap(const Scalar* data, const std::array<Indextype, Rank>& dims);

    template <typename Scalar, std::size_t Rank>
    static MapTType<Scalar, Rank> Map(Scalar* data, const std::array<Indextype, Rank>& dims);

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
    static const Scalar* get_raw_data(const TType<Scalar, Rank>& T);

    template <typename Scalar, int Rank>
    static Scalar* get_raw_data(TType<Scalar, Rank>& T);

    // shape info
    template <typename Scalar, int Rank>
    static std::array<Indextype, Rank> dimensions(const TType<Scalar, Rank>& T);

    // tensorProd
    template <typename Scalar, int Rank>
    static TType<Scalar, Rank> tensorProd(const TType<Scalar, Rank>& T1, const TType<Scalar, Rank>& T2);

    template <typename Scalar, std::size_t Rank, typename Expr1, typename Expr2>
    static void addScale(const Expr1& src, Expr2& dst, const Scalar& scale);

    // methods rvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
    static TType<Scalar, Rank1 + Rank2 - sizeof...(Is)> contract(const TType<Scalar, Rank1>& T1, const TType<Scalar, Rank2>& T2);

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(const TType<Scalar, Rank>& T);

    template <typename Scalar, std::size_t Rank, Indextype... p>
    static TType<Scalar, Rank> shuffle(const TType<Scalar, Rank>& T, seq::iseq<Indextype, p...> s);

    template <typename Expr, Indextype... p>
    static const Eigen::TensorShufflingOp<const std::array<Indextype, Eigen::internal::traits<Expr>::NumDimensions>, const Expr>
    shuffle_view(const Expr& T);

    // template<typename Scalar, std::size_t Rank, Indextype... p>
    // static auto shuffle_view(const cMapTType<Scalar,Rank>& T)
    // {
    //         static_assert(Rank == sizeof...(p));
    //         std::array<Indextype, Rank> dims = {p...};
    //         return T.shuffle(dims);
    // }

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static TType<Scalar, Rank2> reshape(const TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims);

    // methods lvalue
    template <typename Scalar, std::size_t Rank1, std::size_t Rank2>
    static void setSubTensor(TType<Scalar, Rank1>& T,
                             const std::array<Indextype, Rank2>& offsets,
                             const std::array<Indextype, Rank2>& extents,
                             const TType<Scalar, Rank1>& S);

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static const Eigen::TensorSlicingOp<const std::array<Indextype, Rank2>, const std::array<Indextype, Rank2>, const TType<Scalar, Rank1>>
    slice(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents);

    template <typename Scalar, int Rank1, std::size_t Rank2>
    static const Eigen::TensorReshapingOp<const std::array<Indextype, Rank2>, const TType<Scalar, Rank1>>
    reshape(TType<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims);

    template <typename Scalar, int Rank>
    static std::string print(const TType<Scalar, Rank>& T);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/TensorInterface_Eigen_impl.cpp"
#endif

#endif
