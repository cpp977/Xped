#include <unordered_set>

#include "Xped/Util/Macros.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/CoeffBinaryOp.hpp"
#include "Xped/Core/CoeffUnaryOp.hpp"
#include "Xped/Core/DiagCoeffUnaryOp.hpp"
#include "Xped/Core/Tensor.hpp"

#include "Xped/Core/TensorBase.hpp"

namespace Xped {

template <typename Derived>
typename TensorTraits<Derived>::Scalar TensorBase<Derived>::trace() XPED_CONST
{
    assert(derived().coupledDomain() == derived().coupledCodomain());
    Scalar out = 0.;
    for(size_t i = 0; i < derived().sector().size(); i++) {
        out += PlainInterface::trace(derived().block(i)) * Symmetry::degeneracy(derived().sector(i));
        // out += derived().block(i).trace() * Symmetry::degeneracy(derived().sector(i));
    }
    return out;
}

template <typename Derived>
typename TensorTraits<Derived>::Scalar TensorBase<Derived>::squaredNorm() XPED_CONST
{
    return (*this * this->adjoint()).trace();
}

template <typename Derived>
XPED_CONST AdjointOp<Derived> TensorBase<Derived>::adjoint() XPED_CONST
{
    return AdjointOp<Derived>(derived());
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived> TensorBase<Derived>::unaryExpr(const std::function<Scalar(Scalar)>& coeff_func) XPED_CONST
{
    return CoeffUnaryOp<Derived>(derived(), coeff_func);
}

template <typename Derived>
Derived& TensorBase<Derived>::operator+=(const Scalar offset)
{
    derived() = unaryExpr([offset](const Scalar s) { return offset + s; });
    return derived();
}

template <typename Derived>
Derived& TensorBase<Derived>::operator-=(const Scalar offset)
{
    derived() = unaryExpr([offset](const Scalar s) { return s - offset; });
    return derived();
}

template <typename Derived>
Derived& TensorBase<Derived>::operator*=(const Scalar scale)
{
    derived() = unaryExpr([scale](const Scalar s) { return scale * s; });
    return derived();
}

template <typename Derived>
Derived& TensorBase<Derived>::operator/=(const Scalar divisor)
{
    derived() = unaryExpr([divisor](const Scalar s) { return s / divisor; });
    return derived();
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived> TensorBase<Derived>::sqrt() XPED_CONST
{
    return unaryExpr([](const Scalar s) { return std::sqrt(s); });
}

template <typename Derived>
XPED_CONST DiagCoeffUnaryOp<Derived> TensorBase<Derived>::diagUnaryExpr(const std::function<Scalar(Scalar)>& coeff_func) XPED_CONST
{
    return DiagCoeffUnaryOp<Derived>(derived(), coeff_func);
}

template <typename Derived>
XPED_CONST DiagCoeffUnaryOp<Derived> TensorBase<Derived>::diag_inv() XPED_CONST
{
    return diagUnaryExpr([](const Scalar s) { return 1. / s; });
}

template <typename Derived>
template <typename OtherDerived>
XPED_CONST CoeffBinaryOp<Derived, OtherDerived> TensorBase<Derived>::binaryExpr(XPED_CONST TensorBase<OtherDerived>& other,
                                                                                const std::function<Scalar(Scalar, Scalar)>& coeff_func) XPED_CONST
{
    return CoeffBinaryOp<Derived, OtherDerived>(derived(), other.derived(), coeff_func);
}

template <typename Derived>
template <typename OtherDerived>
Derived& TensorBase<Derived>::operator+=(XPED_CONST TensorBase<OtherDerived>& other)
{
    derived() = binaryExpr(other, [](const Scalar s1, const Scalar s2) { return s1 + s2; });
    return derived();
}

template <typename Derived>
template <typename OtherDerived>
Derived& TensorBase<Derived>::operator-=(XPED_CONST TensorBase<OtherDerived>& other)
{
    derived() = binaryExpr(other, [](const Scalar s1, const Scalar s2) { return s1 - s2; });
    return derived();
}

template <typename Derived>
template <typename OtherDerived>
Tensor<typename TensorTraits<Derived>::Scalar,
       TensorTraits<Derived>::Rank,
       TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
       typename TensorTraits<Derived>::Symmetry,
       false,
       typename TensorTraits<Derived>::AllocationPolicy>
TensorBase<Derived>::operator*(XPED_CONST TensorBase<OtherDerived>& other) XPED_CONST
{
    typedef typename std::remove_const<std::remove_reference_t<OtherDerived>>::type OtherDerived_;
    static_assert(CoRank == TensorTraits<OtherDerived_>::Rank);
    auto derived_ref = derived();
    auto other_derived_ref = other.derived();
    assert(derived_ref.world() == other_derived_ref.world());
    assert(derived_ref.coupledCodomain() == other_derived_ref.coupledDomain());

    Tensor<Scalar, Rank, TensorTraits<OtherDerived_>::CoRank, Symmetry, false, AllocationPolicy> Tout(
        derived_ref.uncoupledDomain(), other_derived_ref.uncoupledCodomain(), derived_ref.world());
    // std::unordered_set<typename Symmetry::qType> uniqueController;
    auto other_dict = other_derived_ref.dict();
    auto this_dict = derived_ref.dict();
    for(size_t i = 0; i < derived_ref.sector().size(); i++) {
        // uniqueController.insert(derived_ref.sector(i));
        auto it = other_dict.find(derived_ref.sector(i));
        if(it == other_dict.end()) { continue; }
        // auto it_out = Tout.dict().find(derived_ref.sector(i));
        // if(it_out == Tout.dict().end()) {
        Tout.push_back(derived_ref.sector(i), PlainInterface::prod(derived_ref.block(i), other_derived_ref.block(it->second)));
        // SPDLOG_CRITICAL("({},{})x({},{})",
        //                 derived_ref.block(i).rows(),
        //                 derived_ref.block(i).cols(),
        //                 other_derived_ref.block(it->second).rows(),
        //                 other_derived_ref.block(it->second).cols());
        // } else {
        //     Tout.block(it_out->second) += PlainInterface::prod<Scalar>(derived_ref.block(i), other_derived_ref.block(it->second));
        // }
        // Tout.push_back(derived_ref.sector(i), PlainInterface::prod<Scalar>(derived_ref.block(i), other_derived_ref.block(it->second)));
        // Tout.block_[i] = T1.block_[i] * T2.block_[it->second];
    }
    // for(size_t i = 0; i < other_derived_ref.sector().size(); i++) {
    //     if(auto it = uniqueController.find(other_derived_ref.sector(i)); it != uniqueController.end()) { continue; }
    //     auto it = this_dict.find(other_derived_ref.sector(i));
    //     if(it == this_dict.end()) { continue; }
    //     // Tout.push_back(other_derived_ref.sector(i), Plain::template prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
    //     Tout.push_back(other_derived_ref.sector(i), PlainInterface::prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
    // }
    return Tout;
}

} // namespace Xped
