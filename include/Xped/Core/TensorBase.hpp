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
class CoeffUnaryOp;

template <typename XprType>
class BlockUnaryOp;

template <typename XprType>
class DiagCoeffUnaryOp;

template <typename XprTypeLeft, typename XprTypeRight>
class CoeffBinaryOp;

template <typename XprTypeLeft, typename XprTypeRight>
class DiagCoeffBinaryOp;

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, bool ENABLE_AD, typename AllocationPolicy>
class Tensor;

template <typename Derived>
class TensorBase
{
public:
    typedef typename TensorTraits<Derived>::Scalar Scalar;
    typedef typename TensorTraits<Derived>::Symmetry Symmetry;
    using AllocationPolicy = typename TensorTraits<Derived>::AllocationPolicy;

    static constexpr std::size_t Rank = TensorTraits<Derived>::Rank;
    static constexpr std::size_t CoRank = TensorTraits<Derived>::CoRank;

    using MatrixType = PlainInterface::MType<Scalar>;

    XPED_CONST AdjointOp<Derived> adjoint() XPED_CONST;

    // Unary operations
    XPED_CONST CoeffUnaryOp<Derived> unaryExpr(const std::function<Scalar(Scalar)>& coeff_func) XPED_CONST;

    XPED_CONST BlockUnaryOp<Derived> unaryExpr(const std::function<MatrixType(const MatrixType&)>& coeff_func) XPED_CONST;

    XPED_CONST CoeffUnaryOp<Derived> sqrt() XPED_CONST;
    XPED_CONST CoeffUnaryOp<Derived> inv() XPED_CONST;
    XPED_CONST CoeffUnaryOp<Derived> square() XPED_CONST;
    XPED_CONST CoeffUnaryOp<Derived> abs() XPED_CONST;

    XPED_CONST BlockUnaryOp<Derived> msqrt() XPED_CONST;
    XPED_CONST BlockUnaryOp<Derived> mexp(Scalar factor) XPED_CONST;

    Derived& operator+=(const Scalar offset);
    Derived& operator-=(const Scalar offset);
    Derived& operator*=(const Scalar factor);
    Derived& operator/=(const Scalar divisor);

    XPED_CONST DiagCoeffUnaryOp<Derived> diagUnaryExpr(const std::function<Scalar(Scalar)>& coeff_func) XPED_CONST;

    XPED_CONST DiagCoeffUnaryOp<Derived> diag_inv() XPED_CONST;
    XPED_CONST DiagCoeffUnaryOp<Derived> diag_sqrt() XPED_CONST;

    // Binary operations
    template <typename OtherDerived>
    XPED_CONST DiagCoeffBinaryOp<Derived, OtherDerived> diagBinaryExpr(XPED_CONST TensorBase<OtherDerived>& other,
                                                                       const std::function<Scalar(Scalar, Scalar)>& coeff_func) XPED_CONST;

    template <typename OtherDerived>
    XPED_CONST CoeffBinaryOp<Derived, OtherDerived> binaryExpr(XPED_CONST TensorBase<OtherDerived>& other,
                                                               const std::function<Scalar(Scalar, Scalar)>& coeff_func) XPED_CONST;

    template <typename OtherDerived>
    Derived& operator+=(XPED_CONST TensorBase<OtherDerived>& other);
    template <typename OtherDerived>
    Derived& operator-=(XPED_CONST TensorBase<OtherDerived>& other);

    template <bool = false, typename OtherDerived>
    Tensor<Scalar,
           TensorTraits<Derived>::Rank,
           TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
           Symmetry,
           false,
           AllocationPolicy>
    operator*(XPED_CONST TensorBase<OtherDerived>& other) XPED_CONST;

    template <bool TRACK = false, typename OtherDerived>
    Tensor<Scalar,
           Rank,
           TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
           Symmetry,
           false,
           AllocationPolicy>
    operator*(TensorBase<OtherDerived>&& other) XPED_CONST
    {
        TensorBase<OtherDerived>& tmp = other;
        return this->operator*<TRACK>(tmp);
    }

    template <bool = false>
    Scalar trace() XPED_CONST;

    ScalarTraits<Scalar>::Real maxNorm() XPED_CONST;

    ScalarTraits<Scalar>::Real squaredNorm() XPED_CONST;

    inline ScalarTraits<Scalar>::Real norm() XPED_CONST { return std::sqrt(squaredNorm()); }

    ScalarTraits<Scalar>::Real maxCoeff(std::size_t& max_block, PlainInterface::MIndextype& max_row, PlainInterface::MIndextype& max_col) XPED_CONST;

    inline Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy> eval() const
    {
        return Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>(derived());
    };

    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }

protected:
    template <typename, std::size_t, std::size_t, typename, bool, typename>
    friend class Tensor;
    template <typename OtherDerived>
    friend class TensorBase;

    // inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    // inline Derived& derived() { return *static_cast<Derived*>(this); }
};

template <typename DerivedLeft, typename DerivedRight>
XPED_CONST CoeffBinaryOp<DerivedLeft, DerivedRight> operator+(XPED_CONST TensorBase<DerivedLeft>& left, XPED_CONST TensorBase<DerivedRight>& right)
{
    return left.binaryExpr(right, [](const typename DerivedLeft::Scalar s1, const typename DerivedRight::Scalar s2) { return s1 + s2; });
}

template <typename DerivedLeft, typename DerivedRight>
XPED_CONST CoeffBinaryOp<DerivedLeft, DerivedRight> operator+(TensorBase<DerivedLeft>&& left, TensorBase<DerivedRight>&& right)
{
    TensorBase<DerivedLeft>& tmp_left = left;
    TensorBase<DerivedRight>& tmp_right = right;
    return tmp_left.binaryExpr(tmp_right, [](const typename DerivedLeft::Scalar s1, const typename DerivedRight::Scalar s2) { return s1 + s2; });
}

template <typename DerivedLeft, typename DerivedRight>
XPED_CONST CoeffBinaryOp<DerivedLeft, DerivedRight> operator-(XPED_CONST TensorBase<DerivedLeft>& left, XPED_CONST TensorBase<DerivedRight>& right)
{
    return left.binaryExpr(right, [](const typename DerivedLeft::Scalar s1, const typename DerivedRight::Scalar s2) { return s1 - s2; });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived> operator+(XPED_CONST TensorBase<Derived>& left, const typename Derived::Scalar offset)
{
    return left.unaryExpr([offset](const typename Derived::Scalar s) { return offset + s; });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived> operator+(const typename Derived::Scalar offset, XPED_CONST TensorBase<Derived>& right)
{
    return right.unaryExpr([offset](const typename Derived::Scalar s) { return offset + s; });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived> operator-(XPED_CONST TensorBase<Derived>& left, const typename Derived::Scalar offset)
{
    return left.unaryExpr([offset](const typename Derived::Scalar s) { return s - offset; });
}

template <bool = false, typename Derived>
XPED_CONST CoeffUnaryOp<Derived> operator*(XPED_CONST TensorBase<Derived>& left, const typename Derived::Scalar factor)
{
    return left.unaryExpr([factor](const typename Derived::Scalar s) { return s * factor; });
}

template <bool = false, typename Derived>
XPED_CONST CoeffUnaryOp<Derived> operator*(const typename Derived::Scalar factor, XPED_CONST TensorBase<Derived>& right)
{
    return right.unaryExpr([factor](const typename Derived::Scalar s) { return s * factor; });
}

template <bool = false, typename Derived>
XPED_CONST CoeffUnaryOp<Derived> operator*(const typename Derived::Scalar factor, TensorBase<Derived>&& right)
{
    TensorBase<Derived>& tmp_right = right;
    return tmp_right.unaryExpr([factor](const typename Derived::Scalar s) { return s * factor; });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived> operator/(XPED_CONST TensorBase<Derived>& left, const typename Derived::Scalar divisor)
{
    return left.unaryExpr([divisor](const typename Derived::Scalar s) { return s / divisor; });
}

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/TensorBase.cpp"
#endif

#endif
