#ifndef VARI_VALUE_HPP_
#define VARI_VALUE_HPP_

#include "stan/math/rev/core/vari.hpp"

namespace stan::math {
template <typename T>
class vari_value<T, require_tensor_v<T>> : public vari_base
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
        stan::math::ChainableStack::instance_->var_stack_.push_back(this);
    }

    template <typename S>
    explicit vari_value(const S& x, bool stacked)
        : val_(x)
        , adj_(x.uncoupledDomain(), x.uncoupledCodomain(), x.world())
    {
        adj_.setZero();
        if(stacked) {
            ChainableStack::instance_->var_stack_.push_back(this);
        } else {
            ChainableStack::instance_->var_nochain_stack_.push_back(this);
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
