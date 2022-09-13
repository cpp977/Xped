#ifndef XPED_COMB_SYM_H_
#define XPED_COMB_SYM_H_

#include <boost/mp11.hpp>

#include "Xped/Symmetry/S1xS2.hpp"

namespace Xped::Sym {

template <typename... Syms>
struct Combined : public Combined<boost::mp11::mp_front<Combined<Syms...>>, boost::mp11::mp_pop_front<Combined<Syms...>>>
{};

template <typename S1, typename S2>
struct Combined<S1, S2> : public S1xS2<S1, S2>
{};

template <typename S1>
struct Combined<S1> : public S1
{};

} // namespace Xped::Sym
#endif
