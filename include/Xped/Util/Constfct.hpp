#ifndef CONSTFCT_H_
#define CONSTFCT_H_

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

} // namespace Xped::util::constFct
#endif
