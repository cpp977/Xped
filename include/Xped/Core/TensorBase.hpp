#ifndef XPED_BASE_H_
#define XPED_BASE_H_

#include "Xped/Interfaces/PlainInterface.hpp"

namespace Xped {

template <typename Derived>
struct TensorTraits
{};

// forward declarations
template <typename XprType>
class AdjointOp;

template <typename XprType>
class ScaledOp;

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib>
class Tensor;

template <typename Derived>
class TensorBase
{
public:
    typedef typename TensorTraits<Derived>::Scalar Scalar;
    typedef typename TensorTraits<Derived>::Symmetry Symmetry;
    typedef typename TensorTraits<Derived>::PlainLib PlainLib;
    typedef typename TensorTraits<Derived>::MatrixType MatrixType;
    typedef typename TensorTraits<Derived>::TensorType TensorType;
    typedef typename TensorTraits<Derived>::VectorType VectorType;

    static constexpr std::size_t Rank = TensorTraits<Derived>::Rank;
    static constexpr std::size_t CoRank = TensorTraits<Derived>::CoRank;
    typedef typename PlainLib::template MapTType<Scalar, Rank + CoRank> TensorMapType;
    typedef typename PlainLib::template cMapTType<Scalar, Rank + CoRank> TensorcMapType;
    typedef typename PlainLib::Indextype IndexType;

    XPED_CONST ScaledOp<Derived> operator*(const Scalar scale) const;

    XPED_CONST AdjointOp<Derived> adjoint() XPED_CONST;

    template <typename OtherDerived>
    Tensor<typename TensorTraits<Derived>::Scalar,
           TensorTraits<Derived>::Rank,
           TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
           typename TensorTraits<Derived>::Symmetry,
           typename TensorTraits<Derived>::PlainLib>
    operator*(OtherDerived&& other) XPED_CONST;

    Scalar trace() XPED_CONST;

    Scalar squaredNorm() XPED_CONST;

    inline Scalar norm() XPED_CONST { return std::sqrt(squaredNorm()); }

    inline Tensor<Scalar, Rank, CoRank, Symmetry, PlainLib> eval() const { return Tensor<Scalar, Rank, CoRank, Symmetry, PlainLib>(derived()); };

protected:
    template <typename Scalar, std::size_t Rank__, std::size_t CoRank__, typename Symmetry__, typename PlainLib__>
    friend class Tensor;
    template <typename OtherDerived>
    friend class TensorBase;

    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/TensorBase.cpp"
#endif

#endif
