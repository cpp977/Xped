#ifndef XPED_VARI_VALUE_HPP_
#define XPED_VARI_VALUE_HPP_

#include "stan/math/rev/core/vari.hpp"

class Empty
{};

// template <typename Derived>
// class vari_view_base
// {
//     vari_view_base() = default;
//     friend Derived;

//     inline Derived& derived() { return static_cast<Derived&>(*this); }
//     inline const Derived& derived() const { return static_cast<const Derived&>(*this); }

// public:
// };

namespace stan::math {
template <typename T>
class vari_value<T, require_tensor_v<T>> : public vari_base, public std::conditional_t<is_arena_tensor<T>::value, Empty, chainable_alloc>
{
    using value_type = T; // The underlying type for this class
    using Scalar = typename T::Scalar; // A floating point type
    // using Storage = typename T::Storage;  // A floating point type
    using vari_type = vari_value<T>;

    T val_;

    T adj_;

    template <typename S>
    explicit vari_value(const S& x)
        : val_(x)
        , adj_(x.uncoupledDomain(), x.uncoupledCodomain(), x.world())
    {
        adj_.setZero();
        if constexpr(is_arena_tensor<T>::value) { stan::math::ChainableStack::instance_->var_stack_.push_back(this); }
    }

    template <typename S>
    explicit vari_value(const S& x, bool stacked)
        : val_(x)
        , adj_(x.uncoupledDomain(), x.uncoupledCodomain(), x.world())
    {
        adj_.setZero();
        if(stacked) {
            if constexpr(is_arena_tensor<T>::value) { stan::math::ChainableStack::instance_->var_stack_.push_back(this); }
        } else {
            if constexpr(is_arena_tensor<T>::value) { stan::math::ChainableStack::instance_->var_nochain_stack_.push_back(this); }
        }
    }

    inline const auto& val() const noexcept { return val_; }
    inline auto& val_op() noexcept { return val_; }

    inline auto& adj() noexcept { return adj_; }
    inline auto& adj() const noexcept { return adj_; }
    inline auto& adj_op() noexcept { return adj_; }

    constexpr std::size_t rank() const { return val_.rank(); }
    constexpr std::size_t corank() const { return val_.corank(); }

    virtual void chain() {}

    inline void init_dependent() { adj_.setOnes(); }

    inline void set_zero_adjoint() final { adj_.setZero(); }

    friend std::ostream& operator<<(std::ostream& os, const vari_value<T>* v) { return os << "val: \n" << v->val_ << " \nadj: \n" << v->adj_; }

private:
    template <typename, typename>
    friend class var_value;
};
} // namespace stan::math
#endif
