#ifndef VAR_VALUE_HPP_
#define VAR_VALUE_HPP_

#include "Xped/Core/Tensor.hpp"

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

    // auto adjoint() const
    // {
    //     using vari_type = decltype(vi_->adjoint());
    //     var_value<typename vari_type::value_type> out(new vari_type(vi_->adjoint()));
    //     return out;
    // }

    template <typename Scalar>
    var_value& operator-=(const Scalar s)
    {
        val_op() = val() - s;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const var_value<T>& v)
    {
        if(v.vi_ == nullptr) { return os << "uninitialized"; }
        return os << v.val();
    }
};

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>
operator-(const stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>& t, Scalar s)
{
    stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>> out(t.val() - s);
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>
operator+(const stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>& t, Scalar s)
{
    stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>> out(t.val() + s);
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>
operator*(const stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>& t, Scalar s)
{
    stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>> out((t.val() * s).eval());
    stan::math::reverse_pass_callback([out, t, s]() mutable { t.adj() += (out.adj() * s).eval(); });
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t MiddleRank, std::size_t CoRank, typename Symmetry>
stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>
operator*(const Xped::ArenaTensor<Scalar, Rank, MiddleRank, Symmetry>& left,
          const stan::math::var_value<Xped::ArenaTensor<Scalar, MiddleRank, CoRank, Symmetry>>& right)
{
    stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>> res(left * right.val());
    stan::math::reverse_pass_callback([res, left, right]() mutable { right.adj() += (left.adjoint() * res.adj()); });
    return res;
}

template <typename Scalar, std::size_t Rank, typename Symmetry>
stan::math::var operator*(const stan::math::var_value<Xped::ArenaTensor<Scalar, 0, Rank, Symmetry>>& left,
                          const stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, 0, Symmetry>>& right)
{
    stan::math::var res = (left.val() * right.val()).block(0)(0, 0);
    stan::math::reverse_pass_callback([res, left, right]() mutable {
        right.adj() += left.val().adjoint() * res.adj();
        left.adj() += res.adj() * right.val().adjoint();
    });
    return res;
}

template <typename Scalar, std::size_t Rank, typename Symmetry>
stan::math::var operator*(const Xped::ArenaTensor<Scalar, 0, Rank, Symmetry>& left,
                          const stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, 0, Symmetry>>& right)
{
    stan::math::var res = (left * right.val()).block(0)(0, 0);
    stan::math::reverse_pass_callback([res, left, right]() mutable { right.adj() += left.adjoint() * res.adj(); });
    return res;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
stan::math::var_value<Xped::ArenaTensor<Scalar, CoRank, Rank, Symmetry>>
adjoint(const stan::math::var_value<Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry>>& t)
{
    stan::math::var_value<Xped::ArenaTensor<Scalar, CoRank, Rank, Symmetry>> res(t.val().adjoint().eval());
    stan::math::reverse_pass_callback([res, t]() mutable { t.adj() += res.adj().adjoint().eval(); });
    return res;
}

} // namespace stan::math
#endif
