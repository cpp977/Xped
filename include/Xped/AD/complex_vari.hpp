#ifndef XPED_COMPLEX_VARI_HPP_
#define XPED_COMPLEX_VARI_HPP_

#include "stan/math/rev/core/vari.hpp"

namespace stan::math {

template <>
class vari_value<std::complex<double>, void> : public vari_base
{
public:
    using value_type = std::complex<double>;
    /**
     * The value of this variable.
     */
    const value_type val_;
    /**
     * The adjoint of this variable, which is the partial derivative
     * of this variable with respect to the root variable.
     */
    value_type adj_{0.0, 0.0};

    /**
     * Construct a variable implementation from a value.  The
     * adjoint is initialized to zero.
     *
     * All constructed variables are added to the stack.  Variables
     * should be constructed before variables on which they depend
     * to insure proper partial derivative propagation.  During
     * derivative propagation, the chain() method of each variable
     * will be called in the reverse order of construction.
     *
     * @tparam S a floating point type.
     * @param x Value of the constructed variable.
     */
    template <typename S, require_convertible_t<S&, std::complex<double>>* = nullptr>
    vari_value(S x) noexcept
        : val_(x)
    { // NOLINT
        ChainableStack::instance_->var_stack_.push_back(this);
    }

    /**
     * Construct a variable implementation from a value.  The
     *  adjoint is initialized to zero and if `stacked` is `false` this vari
     *  will be not be put on the var_stack. Instead it will only be put on
     *  a stack to keep track of whether the adjoint needs to be set to zero.
     *  Variables should be constructed before variables on which they depend
     *  to insure proper partial derivative propagation.  During
     *  derivative propagation, the chain() method of each variable
     *  will be called in the reverse order of construction.
     *
     * @tparam S n floating point type.
     * @param x Value of the constructed variable.
     * @param stacked If false will put this this vari on the nochain stack so
     * that its `chain()` method is not called.
     */
    template <typename S, require_convertible_t<S&, std::complex<double>>* = nullptr>
    vari_value(S x, bool stacked) noexcept
        : val_(x)
    {
        if(stacked) {
            ChainableStack::instance_->var_stack_.push_back(this);
        } else {
            ChainableStack::instance_->var_nochain_stack_.push_back(this);
        }
    }

    /**
     * Return a constant reference to the value of this vari.
     *
     * @return The value of this vari.
     */
    inline const auto& val() const { return val_; }

    /**
     * Return a reference of the derivative of the root expression with
     * respect to this expression.  This method only works
     * after one of the `grad()` methods has been
     * called.
     *
     * @return Adjoint for this vari.
     */
    inline auto& adj() const { return adj_; }

    /**
     * Return a reference to the derivative of the root expression with
     * respect to this expression.  This method only works
     * after one of the `grad()` methods has been
     * called.
     *
     * @return Adjoint for this vari.
     */
    inline auto& adj() { return adj_; }

    inline void chain() {}

    /**
     * Initialize the adjoint for this (dependent) variable to 1.
     * This operation is applied to the dependent variable before
     * propagating derivatives, setting the derivative of the
     * result with respect to itself to be 1.
     */
    inline void init_dependent() noexcept { adj_ = {1.0, 0.0}; }

    /**
     * Set the adjoint value of this variable to 0.  This is used to
     * reset adjoints before propagating derivatives again (for
     * example in a Jacobian calculation).
     */
    inline void set_zero_adjoint() noexcept final { adj_ = {0.0, 0.0}; }

    /**
     * Insertion operator for vari. Prints the current value and
     * the adjoint value.
     *
     * @param os [in, out] ostream to modify
     * @param v [in] vari object to print.
     *
     * @return The modified ostream.
     */
    friend std::ostream& operator<<(std::ostream& os, const vari_value<value_type>* v) { return os << v->val_ << ":" << v->adj_; }

private:
    template <typename, typename>
    friend class var_value;
};

} // namespace stan::math

#endif
