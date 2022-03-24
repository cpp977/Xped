// code from github: https://gist.github.com/klemens-morgenstern/b75599292667a4f53007
// klemens-morgenstern

#ifndef XPED_JOINARRAY_HPP_
#define XPED_JOINARRAY_HPP_

namespace Xped {

namespace detail {
template <std::size_t... Size>
struct num_tuple
{};

template <std::size_t Prepend, typename T>
struct appender
{};

template <std::size_t Prepend, std::size_t... Sizes>
struct appender<Prepend, num_tuple<Sizes...>>
{
    using type = num_tuple<Prepend, Sizes...>;
};

template <std::size_t Size, std::size_t Counter = 0>
struct counter_tuple
{
    using type = typename appender<Counter, typename counter_tuple<Size, Counter + 1>::type>::type;
};

template <std::size_t Size>
struct counter_tuple<Size, Size>
{
    using type = num_tuple<>;
};

} // namespace detail

namespace util {

template<typename T, std::size_t LL, std::size_t RL, std::size_t ... LLs, std::size_t ... RLs>
constexpr std::array<T, LL+RL> join(const std::array<T, LL> rhs, const std::array<T, RL> lhs, detail::num_tuple<LLs...>, detail::num_tuple<RLs...>)
{
	return {rhs[LLs]..., lhs[RLs]... };
};


template<typename T, std::size_t LL, std::size_t RL>
constexpr std::array<T, LL+RL> join(std::array<T, LL> rhs, std::array<T, RL> lhs)
{
	//using l_t = typename detail::counter_tuple<LL>::type;
	return join(rhs, lhs, typename detail::counter_tuple<LL>::type(), typename detail::counter_tuple<RL>::type());
}

} // namespace util

} // namespace Xped
#endif
