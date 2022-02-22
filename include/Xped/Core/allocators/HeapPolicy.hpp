#ifndef XPED_HEAP_POLICY_HPP_
#define XPED_HEAP_POLICY_HPP_

namespace Xped {

struct HeapPolicy
{
    template <typename T>
    using Allocator = std::allocator<T>;
};

} // namespace Xped
#endif
