#ifndef XPED_STAN_ARENA_POLICY_HPP_
#define XPED_STAN_ARENA_POLICY_HPP_

#include "stan/math/rev/core/arena_allocator.hpp"

namespace Xped {

struct StanArenaPolicy
{
    template <typename T>
    using Allocator = stan::math::arena_allocator<T>;
};

} // namespace Xped
#endif
