#ifndef VAR_VALUE_HPP_
#define VAR_VALUE_HPP_

template <typename T>
struct is_arena_tensor
{
    static const bool value = false;
};

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
struct is_arena_tensor<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>
{
    static const bool value = true;
};

template <typename T>
struct is_arena_tensor_var
{
    static const bool value = false;
};

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
struct is_arena_tensor_var<stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>>
{
    static const bool value = true;
};

template <typename T>
using require_tensor_v = stan::require_t<stan::bool_constant<is_arena_tensor<T>::value>>;

template <typename T>
using require_tensor_t = stan::bool_constant<is_arena_tensor<T>::value>;

template <typename T>
using require_tensor_var_t = stan::bool_constant<is_arena_tensor_var<T>::value>;

template <typename T>
using require_tensor_var_v = stan::require_t<stan::bool_constant<is_arena_tensor_var<T>::value>>;

#include "vari_value.hpp"

namespace stan::math {
template <typename T>
class var_value<T, require_tensor_v<T>>
{
public:
    using value_type = T; // type in vari_value -->ArenaTensor.
    using vari_type = vari_value<value_type>;

    vari_type* vi_;

    inline bool is_uninitialized() noexcept { return (vi_ == nullptr); }

    var_value()
        : vi_(nullptr)
    {}

    template <typename S>
    var_value(const var_value<S>& other)
        : vi_(other.vi_)
    {}

    // var_value(arena_tensor = Tensor<Scalar, rank, arena_allocator<Scalar>>)
    template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
    var_value(const Xped::Tensor<Scalar, Rank, CoRank, Symmetry, AllocationPolicy>& x)
        : vi_(new vari_type(x, false))
    {} // NOLINT

    var_value(vari_type* vi)
        : vi_(vi)
    {}

    inline const auto& val() const noexcept { return vi_->val(); }
    inline auto& val_op() noexcept { return vi_->val_op(); }

    inline auto& adj() noexcept { return vi_->adj(); }
    inline auto& adj() const noexcept { return vi_->adj(); }
    inline auto& adj_op() noexcept { return vi_->adj(); }

    constexpr std::size_t rank() const noexcept { return vi_->rank(); }
    constexpr std::size_t corank() const noexcept { return vi_->corank(); }

    inline vari_type& operator*() { return *vi_; }

    inline vari_type* operator->() { return vi_; }

    template <typename Scalar>
    var_value& operator-=(const Scalar s)
    {
        val_op() = val() - s;
        // stan::math::reverse_pass_callback([this]() mutable { adj() += adj(); });
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const var_value<T>& v)
    {
        if(v.vi_ == nullptr) { return os << "uninitialized"; }
        return os << v.val();
    }
};
} // namespace stan::math
#endif
