#ifndef VAR_VALUE_HPP_
#define VAR_VALUE_HPP_

#include "Xped/Core/Tensor.hpp"

#include "vari_value.hpp"

namespace Xped {

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry_, typename AllocationPolicy_>
class Tensor<Scalar_, Rank, CoRank, Symmetry_, true, AllocationPolicy_>
{
public:
    using Scalar = Scalar_;
    using RealScalar = typename ScalarTraits<Scalar>::Real;

    using Symmetry = Symmetry_;
    using qType = typename Symmetry::qType;

    using AllocationPolicy = AllocationPolicy_;

    using value_type = Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>; // type in vari_value -->ArenaTensor or Tensor.
    using vari_type = stan::math::vari_value<value_type>;

    vari_type* vi_;

    inline bool is_uninitialized() noexcept { return (vi_ == nullptr); }

    Tensor()
        : vi_(nullptr)
    {}

    Tensor(const Xped::Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>& x)
        : vi_(new vari_type(x, false))
    {}

    Tensor(vari_type* vi)
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

    auto adjoint() const
    {
        Xped::Tensor<Scalar, CoRank, Rank, Symmetry, true, AllocationPolicy> out(val().adjoint().eval());
        stan::math::reverse_pass_callback([this_adj = this->adj(), out]() mutable { this_adj += out.adj().adjoint().eval(); });
        return out;
    }

    template <int shift, std::size_t... p>
    auto permute() const
    {
        Xped::Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, true, AllocationPolicy> out(val().template permute<shift, p...>());
        stan::math::reverse_pass_callback([this_adj = this->adj(), out]() mutable {
            using inverse = decltype(Xped::util::constFct::inverse_permutation<seq::iseq<std::size_t, p...>>());
            this_adj += out.adj().template permute<-shift>(inverse{});
        });
    }

    Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy>& operator-=(const Scalar s)
    {
        val_op() = val() - s;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy>& v)
    {
        if(v.vi_ == nullptr) { return os << "uninitialized"; }
        return os << v.val();
    }
};

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, true> operator-(const Tensor<Scalar, Rank, CoRank, Symmetry, true>& t, Scalar s)
{
    Tensor<Scalar, Rank, CoRank, Symmetry, true> out(t.val() - s);
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry> operator+(const Tensor<Scalar, Rank, CoRank, Symmetry, true>& t, Scalar s)
{
    Tensor<Scalar, Rank, CoRank, Symmetry> out(t.val() + s);
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, true> operator*(const Tensor<Scalar, Rank, CoRank, Symmetry, true>& t, Scalar s)
{
    Tensor<Scalar, Rank, CoRank, Symmetry, true> out((t.val() * s).eval());
    stan::math::reverse_pass_callback([out, t, s]() mutable { t.adj() += (out.adj() * s).eval(); });
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t MiddleRank, std::size_t CoRank, typename Symmetry>
Xped::Tensor<Scalar, Rank, CoRank, Symmetry, true> operator*(const Tensor<Scalar, Rank, MiddleRank, Symmetry, false>& left,
                                                             const Tensor<Scalar, MiddleRank, CoRank, Symmetry, true>& right)
{
    Tensor<Scalar, Rank, CoRank, Symmetry, true> res(left * right.val());
    stan::math::reverse_pass_callback([res, left, right]() mutable { right.adj() += (left.adjoint() * res.adj()); });
    return res;
}

template <typename Scalar, std::size_t Rank, typename Symmetry>
stan::math::var operator*(const Tensor<Scalar, 0, Rank, Symmetry, true>& left, const Tensor<Scalar, Rank, 0, Symmetry, true>& right)
{
    stan::math::var res = (left.val() * right.val()).block(0)(0, 0);
    stan::math::reverse_pass_callback([res, left, right]() mutable {
        right.adj() += left.val().adjoint() * res.adj();
        left.adj() += res.adj() * right.val().adjoint();
    });
    return res;
}

template <typename Scalar, std::size_t Rank, typename Symmetry>
stan::math::var operator*(const Tensor<Scalar, 0, Rank, Symmetry, false>& left, const Tensor<Scalar, Rank, 0, Symmetry, true>& right)
{
    stan::math::var res = (left * right.val()).block(0)(0, 0);
    stan::math::reverse_pass_callback([res, left, right]() mutable { right.adj() += left.adjoint() * res.adj(); });
    return res;
}

// template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
// stan::math::var_value<Xped::Tensor<Scalar, CoRank, Rank, Symmetry>>
// adjoint(const stan::math::var_value<Xped::Tensor<Scalar, Rank, CoRank, Symmetry>>& t)
// {
//     stan::math::var_value<Xped::Tensor<Scalar, CoRank, Rank, Symmetry>> res(t.val().adjoint().eval());
//     stan::math::reverse_pass_callback([res, t]() mutable { t.adj() += res.adj().adjoint().eval(); });
//     return res;
// }

} // namespace Xped
#endif
