#ifndef XPED_SPININDEX_HPP_
#define XPED_SPININDEX_HPP_

#include <boost/describe.hpp>

namespace Xped {
BOOST_DEFINE_ENUM_CLASS(SPIN_INDEX, UP, DN, NOSPIN, UPDN)

inline std::ostream& operator<<(std::ostream& s, SPIN_INDEX sigma)
{
    if(sigma == SPIN_INDEX::UP) {
        s << "↑";
    } else if(sigma == SPIN_INDEX::DN) {
        s << "↓";
    } else if(sigma == SPIN_INDEX::NOSPIN) {
        s << "↯";
    } else if(sigma == SPIN_INDEX::UPDN) {
        s << "⇅";
    }
    return s;
}
} // namespace Xped
#endif
