#ifndef XPED_IPEPS_OPTS_H_
#define XPED_IPEPS_OPTS_H_

#include <boost/describe.hpp>

namespace Xped::Opts {

BOOST_DEFINE_ENUM_CLASS(LEG, LEFT, UP, RIGHT, DOWN, PHYS)

BOOST_DEFINE_ENUM_CLASS(LoadFormat, MATLAB, NATIVE)

} // namespace Xped::Opts
#endif
