#include <boost/describe.hpp>

namespace Xped::Opts {

template <typename T, typename = typename std::enable_if_t<boost::describe::has_describe_enumerators<T>::value>>
std::ostream& operator<<(std::ostream& os, const T& t)
{
    os << boost::describe::enum_to_string(t, "Unknown");
    return os;
}

} // namespace Xped::Opts
