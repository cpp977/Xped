#ifndef XPED_IPEPS_OPTS_H_
#define XPED_IPEPS_OPTS_H_

#include <boost/describe.hpp>

namespace Xped::Opts {

BOOST_DEFINE_ENUM_CLASS(Leg, Left, Top, Right, Bottom, Phys)

BOOST_DEFINE_ENUM_CLASS(LoadFormat, MATLAB, Native, JSON, JSON_SU2)

BOOST_DEFINE_ENUM_CLASS(Orientation, H, V)

BOOST_DEFINE_ENUM_CLASS(DiscreteSym, None, C4v)

} // namespace Xped::Opts
#endif
