#ifndef XPED_COMPLEX_VAR_HPP_
#define XPED_COMPLEX_VAR_HPP_

#include "Xped/AD/complex_vari.hpp"
#include "stan/math/rev/core/var.hpp"

namespace stan::math {

template <>
class var_value<std::complex<double>, void>
{
public:
    using value_type = std::complex<double>; // type in vari_value.
    using vari_type = vari_value<value_type>;

    /**
     * Pointer to the implementation of this variable.
     *
     * This value should not be modified, but may be accessed in
     * <code>var</code> operators to construct `vari_value<T>`
     * instances.
     */
    vari_type* vi_;

    /**
     * Return `true` if this variable has been
     * declared, but not been defined.  Any attempt to use an
     * undefined variable's value or adjoint will result in a
     * segmentation fault.
     *
     * @return <code>true</code> if this variable does not yet have
     * a defined variable.
     */
    inline bool is_uninitialized() { return (vi_ == nullptr); }

    /**
     * Construct a variable for later assignment.
     *
     * This is implemented as a no-op, leaving the underlying implementation
     * dangling.  Before an assignment, the behavior is thus undefined just
     * as for a basic double.
     */
    var_value()
        : vi_(nullptr)
    {}

    /**
     * Construct a variable from the specified floating point argument
     * by constructing a new `vari_value<value_type>`. This constructor is only
     * valid when `S` is convertible to this `vari_value`'s `value_type`.
     * @tparam S A type that is convertible to `value_type`.
     * @param x Value of the variable.
     */
    template <typename S, require_convertible_t<S&, value_type>* = nullptr>
    var_value(S x)
        : vi_(new vari_type(x, false))
    {} // NOLINT

    /**
     * Construct a variable from a pointer to a variable implementation.
     * @param vi A vari_value pointer.
     */
    var_value(vari_type* vi)
        : vi_(vi)
    {} // NOLINT

    /**
     * Return a constant reference to the value of this variable.
     *
     * @return The value of this variable.
     */
    inline const auto& val() const noexcept { return vi_->val(); }

    /**
     * Return a reference of the derivative of the root expression with
     * respect to this expression.  This method only works
     * after one of the `grad()` methods has been
     * called.
     *
     * @return Adjoint for this variable.
     */
    inline auto& adj() const noexcept { return vi_->adj(); }

    /**
     * Return a reference to the derivative of the root expression with
     * respect to this expression.  This method only works
     * after one of the `grad()` methods has been
     * called.
     *
     * @return Adjoint for this variable.
     */
    inline auto& adj() noexcept { return vi_->adj_; }

    /**
     * Compute the gradient of this (dependent) variable with respect
     * to all (independent) variables.
     *
     * @tparam CheckContainer Not set by user. The default value of value_type
     *  is used to require that grad is only available for scalar `var_value`
     *  types.
     * The grad() function does <i>not</i> recover memory.
     */
    void grad() { stan::math::grad(vi_); }

    // POINTER OVERRIDES

    /**
     * Return a reference to underlying implementation of this variable.
     *
     * If <code>x</code> is of type <code>var</code>, then applying
     * this operator, <code>*x</code>, has the same behavior as
     * <code>*(x.vi_)</code>.
     *
     * <i>Warning</i>:  The returned reference does not track changes to
     * this variable.
     *
     * @return variable
     */
    inline vari_type& operator*() { return *vi_; }

    /**
     * Return a pointer to the underlying implementation of this variable.
     *
     * If <code>x</code> is of type <code>var</code>, then applying
     * this operator, <code>x-&gt;</code>, behaves the same way as
     * <code>x.vi_-&gt;</code>.
     *
     * <i>Warning</i>: The returned result does not track changes to
     * this variable.
     */
    inline vari_type* operator->() { return vi_; }

    // COMPOUND ASSIGNMENT OPERATORS

    /**
     * The compound add/assignment operator for variables (C++).
     *
     * If this variable is a and the argument is the variable b,
     * then (a += b) behaves exactly the same way as (a = a + b),
     * creating an intermediate variable representing (a + b).
     *
     * @param b The variable to add to this variable.
     * @return The result of adding the specified variable to this variable.
     */
    inline var_value<value_type>& operator+=(const var_value<value_type>& b);

    /**
     * The compound add/assignment operator for scalars (C++).
     *
     * If this variable is a and the argument is the scalar b, then
     * (a += b) behaves exactly the same way as (a = a + b).  Note
     * that the result is an assignable lvalue.
     *
     * @param b The scalar to add to this variable.
     * @return The result of adding the specified variable to this variable.
     */
    inline var_value<value_type>& operator+=(value_type b);

    /**
     * The compound subtract/assignment operator for variables (C++).
     *
     * If this variable is a and the argument is the variable b,
     * then (a -= b) behaves exactly the same way as (a = a - b).
     * Note that the result is an assignable lvalue.
     *
     * @param b The variable to subtract from this variable.
     * @return The result of subtracting the specified variable from
     * this variable.
     */
    inline var_value<value_type>& operator-=(const var_value<value_type>& b);

    /**
     * The compound subtract/assignment operator for scalars (C++).
     *
     * If this variable is a and the argument is the scalar b, then
     * (a -= b) behaves exactly the same way as (a = a - b).  Note
     * that the result is an assignable lvalue.
     *
     * @param b The scalar to subtract from this variable.
     * @return The result of subtracting the specified variable from this
     * variable.
     */
    inline var_value<value_type>& operator-=(value_type b);

    /**
     * The compound multiply/assignment operator for variables (C++).
     *
     * If this variable is a and the argument is the variable b,
     * then (a *= b) behaves exactly the same way as (a = a * b).
     * Note that the result is an assignable lvalue.
     *
     * @param b The variable to multiply this variable by.
     * @return The result of multiplying this variable by the
     * specified variable.
     */
    inline var_value<value_type>& operator*=(const var_value<value_type>& b);

    /**
     * The compound multiply/assignment operator for scalars (C++).
     *
     * If this variable is a and the argument is the scalar b, then
     * (a *= b) behaves exactly the same way as (a = a * b).  Note
     * that the result is an assignable lvalue.
     *
     * @param b The scalar to multiply this variable by.
     * @return The result of multiplying this variable by the specified
     * variable.
     */
    inline var_value<value_type>& operator*=(value_type b);

    /**
     * The compound divide/assignment operator for variables (C++).  If this
     * variable is a and the argument is the variable b, then (a /= b)
     * behaves exactly the same way as (a = a / b).  Note that the
     * result is an assignable lvalue.
     *
     * @param b The variable to divide this variable by.
     * @return The result of dividing this variable by the
     * specified variable.
     */
    inline var_value<value_type>& operator/=(const var_value<value_type>& b);

    /**
     * The compound divide/assignment operator for scalars (C++).
     *
     * If this variable is a and the argument is the scalar b, then
     * (a /= b) behaves exactly the same way as (a = a / b).  Note
     * that the result is an assignable lvalue.
     *
     * @param b The scalar to divide this variable by.
     * @return The result of dividing this variable by the specified
     * variable.
     */
    inline var_value<value_type>& operator/=(value_type b);

    /**
     * Write the value of this autodiff variable and its adjoint to
     * the specified output stream.
     *
     * @param os Output stream to which to write.
     * @param v Variable to write.
     * @return Reference to the specified output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const var_value<value_type>& v)
    {
        if(v.vi_ == nullptr) { return os << "uninitialized"; }
        return os << v.val();
    }
};

inline var_value<std::complex<double>> operator/(const var_value<std::complex<double>>& dividend, const var_value<std::complex<double>>& divisor)
{
    return make_callback_var(dividend.val() / divisor.val(), [dividend, divisor](auto&& vi) {
        dividend.adj() += vi.adj() / divisor.val();
        divisor.adj() -= vi.adj() * dividend.val() / (divisor.val() * divisor.val());
    });
}

} // namespace stan::math

namespace std {

stan::math::var_value<double> real(const stan::math::var_value<std::complex<double>>& z)
{
    return stan::math::make_callback_var(z.val().real(), [z](auto&& vi) { z.adj() += vi.adj(); });
}

} // namespace std

#endif
