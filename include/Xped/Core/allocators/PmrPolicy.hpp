#ifndef XPED_PMR_POLICY_HPP_
#define XPED_PMR_POLICY_HPP_

#ifdef _LIBCPP_VERSION /*libc++'s memory_resource header is still in experimental.*/
#    include <experimental/memory_resource>
#else
#    include <memory_resource>
#endif

namespace Xped {

struct PmrPolicy
{
    template <typename T>
    using Allocator = std::pmr::polymorphic_allocator<T>;
};

} // namespace Xped
#endif
