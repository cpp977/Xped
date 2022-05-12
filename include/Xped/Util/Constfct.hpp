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

#if XPED_HAS_NTTP
template <auto a1, std::size_t rank1, auto a2, std::size_t rank2, std::size_t rankres>
consteval auto get_permutations()
{
    constexpr auto r1 = a1.size();
    constexpr auto r2 = a2.size();
    using value_type = typename decltype(a1)::value_type;
    std::array<std::size_t, r1> ind1;
    std::iota(ind1.begin(), ind1.end(), 0);
    std::array<std::size_t, r2> ind2;
    std::iota(ind2.begin(), ind2.end(), 0);

    constexpr int max1 = std::max(0, *std::max_element(a1.begin(), a1.end()));
    constexpr int max2 = std::max(0, *std::max_element(a2.begin(), a2.end()));
    static_assert(max1 == max2);

    std::array<std::size_t, r2> p2{};
    std::array<int, r2> found2;
    std::fill(found2.begin(), found2.end(), -1);
    std::transform(ind2.begin(), ind2.begin() + max2, p2.begin(), [&found2](std::size_t ind) {
        auto val = std::distance(a2.begin(), std::find(a2.begin(), a2.end(), ind + 1));
        found2[ind] = val;
        return val;
    });
    std::transform(ind2.begin() + max2, ind2.end(), p2.begin() + max2, [ind2, &found2](std::size_t ind) {
        auto val =
            *std::find_if(ind2.begin(), ind2.end(), [found2](std::size_t i) { return std::find(found2.begin(), found2.end(), i) == found2.end(); });
        found2[ind] = val;
        return val;
    });

    std::array<std::size_t, r1> p1{};
    std::array<int, r1> found1;
    std::fill(found1.begin(), found1.end(), -1);
    std::transform(ind1.begin(), ind1.begin() + max1, p1.begin() + a1.size() - max1, [&found1](std::size_t ind) {
        auto val = std::distance(a1.begin(), std::find(a1.begin(), a1.end(), ind + 1));
        found1[ind] = val;
        return val;
    });
    std::transform(ind1.begin() + max1, ind1.end(), p1.begin(), [ind1, &found1](std::size_t ind) {
        auto val =
            *std::find_if(ind1.begin(), ind1.end(), [found1](std::size_t i) { return std::find(found1.begin(), found1.end(), i) == found1.end(); });
        found1[ind] = val;
        return val;
    });
    std::array<std::size_t, r1 + r2 - 2 * max1> pres{};
    std::iota(pres.begin(), pres.end(), 0);
    std::array<value_type, r1 + r2 - 2 * max1> tmp{};
    auto second_half = std::copy_if(a1.begin(), a1.end(), tmp.begin(), [](int i) { return i < 0; });
    std::copy_if(a2.begin(), a2.end(), second_half, [](int i) { return i < 0; });
    std::sort(pres.begin(), pres.end(), [tmp](std::size_t i, std::size_t j) { return tmp[i] > tmp[j]; });
    constexpr int shift1 = static_cast<value_type>(rank1) - static_cast<value_type>(a1.size()) + static_cast<value_type>(max1);
    constexpr int shift2 = static_cast<value_type>(rank2) - static_cast<value_type>(max2);
    constexpr int shiftres = static_cast<value_type>(a1.size()) - static_cast<value_type>(max1) - static_cast<value_type>(rankres);
    return std::make_tuple(p1, shift1, p2, shift2, pres, shiftres);
}

template <const auto A, typename decltype(A)::value_type... Is>
auto as_sequence_impl(seq::iseq<typename decltype(A)::value_type, Is...>)
{
    return seq::iseq<typename decltype(A)::value_type, A[Is]...>{};
}

template <const auto A, typename Indices = seq::make<typename decltype(A)::value_type, std::tuple_size<decltype(A)>::value>>
auto as_sequence()
{
    return as_sequence_impl<A>(Indices{});
}

#endif
} // namespace Xped::util::constFct
#endif
