#ifndef XPED_BONDS_HPP_
#define XPED_BONDS_HPP_

#include <boost/describe.hpp>

#include "Xped/Util/EnumStream.hpp"

namespace Xped::Opts {
enum class Bond : uint32_t
{
    H = (1 << 0),
    V = (1 << 1),
    D1 = (1 << 2),
    D2 = (1 << 3)
};
BOOST_DESCRIBE_ENUM(Bond, H, V, D1, D2)

constexpr enum Bond operator|(const enum Bond selfValue, const enum Bond inValue) { return (enum Bond)(uint32_t(selfValue) | uint32_t(inValue)); }

constexpr enum Bond operator&(const enum Bond selfValue, const enum Bond inValue) { return (enum Bond)(uint32_t(selfValue) & uint32_t(inValue)); }

} // namespace Xped::Opts

#endif
