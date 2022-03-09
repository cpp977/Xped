#ifndef XPED_PMR_POLICY_HPP_
#define XPED_PMR_POLICY_HPP_

#include <memory_resource>

namespace Xped {

struct PmrPolicy
{
    template <typename T>
    using Allocator = std::pmr::polymorphic_allocator<T>;
};

} // namespace Xped
#endif
