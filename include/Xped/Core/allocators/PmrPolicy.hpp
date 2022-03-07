#ifndef XPED_PMR_POLICY_HPP_
#define XPED_PMR_POLICY_HPP_

namespace Xped {

struct PmrPolicy
{
    template <typename T>
    using Allocator = std::pmr::polymorphic_allocator<T>;
};

} // namespace Xped
#endif
