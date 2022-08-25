#ifndef XPED_SUB_LATTICE_HPP_
#define XPED_SUB_LATTICE_HPP_

#include <boost/describe.hpp>

namespace Xped {

enum SUB_LATTICE
{
    A = 1,
    B = -1
};

inline std::ostream& operator<<(std::ostream& s, SUB_LATTICE sublat)
{
    if(sublat == A) {
        s << "A";
    } else if(sublat == B) {
        s << "B";
    }
    return s;
};

} // namespace Xped
#endif
