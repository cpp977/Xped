#ifndef XPED_CONSTFCT_H_
#define XPED_CONSTFCT_H_

#include "seq/seq.h"

namespace Xped::util::constFct {

template <std::size_t Rank>
constexpr std::size_t shift(std::size_t x, std::size_t /*unused*/)
{
    return x + Rank;
}

template <std::size_t Rank>
constexpr bool isGreaterOrEqual(std::size_t x, std::size_t /*unused*/)
{
    return x >= Rank;
}

template <std::size_t Rank>
constexpr bool isSmaller(std::size_t x, std::size_t /*unused*/)
{
    return x < Rank;
}

template <typename Perm, typename Perm::value_type... Is>
constexpr auto inverse_perm_impl(seq::iseq<typename Perm::value_type, Is...>)
{
    return seq::iseq<typename Perm::value_type, seq::index_of<Is, Perm>...>{};
}

template <typename Perm, typename Indices = seq::make<typename Perm::value_type, Perm::size()>>
constexpr auto inverse_permutation()
{
    return inverse_perm_impl<Perm>(Indices{});
}

} // namespace Xped::util::constFct
#endif
