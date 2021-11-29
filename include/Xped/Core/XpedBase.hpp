#ifndef XPED_BASE_H_
#define XPED_BASE_H_

#include "Xped/Interfaces/PlainInterface.hpp"

template <typename Derived>
struct XpedTraits
{};

// forward declarations
template <typename XprType>
class AdjointOp;

template <typename XprType>
class ScaledOp;

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename PlainLib>
class Xped;

template <typename Derived>
class XpedBase
{
public:
    typedef typename XpedTraits<Derived>::Scalar Scalar;
    typedef typename XpedTraits<Derived>::Symmetry Symmetry;
    typedef typename XpedTraits<Derived>::PlainLib PlainLib;
    typedef typename XpedTraits<Derived>::MatrixType MatrixType;
    typedef typename XpedTraits<Derived>::TensorType TensorType;
    typedef typename XpedTraits<Derived>::VectorType VectorType;

    static constexpr std::size_t Rank = XpedTraits<Derived>::Rank;
    static constexpr std::size_t CoRank = XpedTraits<Derived>::CoRank;
    typedef typename PlainLib::template MapTType<Scalar, Rank + CoRank> TensorMapType;
    typedef typename PlainLib::template cMapTType<Scalar, Rank + CoRank> TensorcMapType;
    typedef typename PlainLib::Indextype IndexType;

    inline XPED_CONST ScaledOp<Derived> operator*(const Scalar scale) const { return ScaledOp<Derived>(derived(), scale); }

    inline XPED_CONST AdjointOp<Derived> adjoint() XPED_CONST { return AdjointOp<Derived>(derived()); }

    template <typename OtherDerived>
    Xped<typename XpedTraits<Derived>::Scalar,
         XpedTraits<Derived>::Rank,
         XpedTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
         typename XpedTraits<Derived>::Symmetry,
         typename XpedTraits<Derived>::PlainLib>
    operator*(OtherDerived&& other) XPED_CONST;

    Scalar trace() XPED_CONST;

    inline Scalar squaredNorm() XPED_CONST { return (*this * this->adjoint()).trace(); }

    inline Scalar norm() XPED_CONST { return std::sqrt(squaredNorm()); }

    inline Xped<Scalar, Rank, CoRank, Symmetry, PlainLib> eval() const { return Xped<Scalar, Rank, CoRank, Symmetry, PlainLib>(derived()); };

protected:
    template <typename Scalar, std::size_t Rank__, std::size_t CoRank__, typename Symmetry__, typename PlainLib__>
    friend class Xped;
    template <typename OtherDerived>
    friend class XpedBase;

    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }
};

#ifndef XPED_COMPILED_LIB
#    include "Core/XpedBase.cpp"
#endif

#endif
