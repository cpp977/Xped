#include <unordered_set>

#include "Xped/Util/Macros.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/ScaledOp.hpp"
#include "Xped/Core/Tensor.hpp"

#include "Xped/Core/TensorBase.hpp"

template <typename Derived>
typename TensorTraits<Derived>::Scalar TensorBase<Derived>::trace() XPED_CONST
{
    assert(derived().coupledDomain() == derived().coupledCodomain());
    Scalar out = 0.;
    for(size_t i = 0; i < derived().sector().size(); i++) {
        out += PlainLib::template trace<Scalar>(derived().block(i)) * Symmetry::degeneracy(derived().sector(i));
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
XPED_CONST ScaledOp<Derived> TensorBase<Derived>::operator*(const Scalar scale) const
{
    return ScaledOp<Derived>(derived(), scale);
}

template <typename Derived>
template <typename OtherDerived>
Tensor<typename TensorTraits<Derived>::Scalar,
     TensorTraits<Derived>::Rank,
     TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
     typename TensorTraits<Derived>::Symmetry,
     typename TensorTraits<Derived>::PlainLib>
TensorBase<Derived>::operator*(OtherDerived&& other) XPED_CONST
{
    typedef typename std::remove_const<std::remove_reference_t<OtherDerived>>::type OtherDerived_;
    static_assert(CoRank == TensorTraits<OtherDerived_>::Rank);
    auto derived_ref = derived();
    auto other_derived_ref = other.derived();
    assert(derived_ref.world() == other_derived_ref.world());
    assert(derived_ref.coupledCodomain() == other_derived_ref.coupledDomain());

    Tensor<Scalar, Rank, TensorTraits<OtherDerived_>::CoRank, Symmetry, PlainLib> Tout;
    Tout.world_ = derived_ref.world();
    Tout.domain = derived_ref.coupledDomain();
    Tout.codomain = other_derived_ref.coupledCodomain();
    Tout.uncoupled_domain = derived_ref.uncoupledDomain();
    Tout.uncoupled_codomain = other_derived_ref.uncoupledCodomain();
    // Tout.sector = T1.sector;
    // Tout.dict = T1.dict;
    // Tout.block_.resize(Tout.sector_.size());
    std::unordered_set<typename Symmetry::qType> uniqueController;
    auto other_dict = other_derived_ref.dict();
    auto this_dict = derived_ref.dict();
    for(size_t i = 0; i < derived_ref.sector().size(); i++) {
        uniqueController.insert(derived_ref.sector(i));
        auto it = other_dict.find(derived_ref.sector(i));
        if(it == other_dict.end()) { continue; }
        // Tout.push_back(derived_ref.sector(i), Plain::template prod<Scalar>(derived_ref.block(i), other_derived_ref.block(it->second)));
        Tout.push_back(derived_ref.sector(i), PlainLib::template prod<Scalar>(derived_ref.block(i), other_derived_ref.block(it->second)));
        // Tout.block_[i] = T1.block_[i] * T2.block_[it->second];
    }
    for(size_t i = 0; i < other_derived_ref.sector().size(); i++) {
        if(auto it = uniqueController.find(other_derived_ref.sector(i)); it != uniqueController.end()) { continue; }
        auto it = this_dict.find(other_derived_ref.sector(i));
        if(it == this_dict.end()) { continue; }
        // Tout.push_back(other_derived_ref.sector(i), Plain::template prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
        Tout.push_back(other_derived_ref.sector(i), PlainLib::template prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
    }
    return Tout;
}
