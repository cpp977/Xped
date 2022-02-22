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
class DiagCoeffUnaryOp;

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
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

    XPED_CONST AdjointOp<Derived> adjoint() XPED_CONST;

    XPED_CONST CoeffUnaryOp<Derived> unaryExpr(const std::function<Scalar(Scalar)>& coeff_func) XPED_CONST;

    XPED_CONST CoeffUnaryOp<Derived> sqrt() XPED_CONST;
    XPED_CONST CoeffUnaryOp<Derived> operator*(const Scalar scale) XPED_CONST;

    XPED_CONST DiagCoeffUnaryOp<Derived> diagUnaryExpr(const std::function<Scalar(Scalar)>& coeff_func) XPED_CONST;

    XPED_CONST DiagCoeffUnaryOp<Derived> diag_inv() XPED_CONST;

    template <typename OtherDerived>
    Tensor<Scalar,
           TensorTraits<Derived>::Rank,
           TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
           Symmetry,
           AllocationPolicy>
    operator*(XPED_CONST TensorBase<OtherDerived>& other) XPED_CONST;

    template <typename OtherDerived>
    Tensor<Scalar, Rank, TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank, Symmetry, AllocationPolicy>
    operator*(TensorBase<OtherDerived>&& other) XPED_CONST
    {
        TensorBase<OtherDerived>& tmp = other;
        return this->operator*(tmp);
    }

    Scalar trace() XPED_CONST;

    Scalar squaredNorm() XPED_CONST;

    inline Scalar norm() XPED_CONST { return std::sqrt(squaredNorm()); }

    inline Tensor<Scalar, Rank, CoRank, Symmetry, AllocationPolicy> eval() const
    {
        return Tensor<Scalar, Rank, CoRank, Symmetry, AllocationPolicy>(derived());
    };

    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }

protected:
    template <typename Scalar, std::size_t Rank__, std::size_t CoRank__, typename Symmetry__, typename AllocationPolicy__>
    friend class Tensor;
    template <typename OtherDerived>
    friend class TensorBase;

    // inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    // inline Derived& derived() { return *static_cast<Derived*>(this); }
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/TensorBase.cpp"
#endif

#endif
