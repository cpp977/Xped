#ifndef XPED_ENUM_STREAM_HPP_
#define XPED_ENUM_STREAM_HPP_

#include <boost/describe.hpp>

namespace Xped {

template <typename T, typename = typename std::enable_if_t<boost::describe::has_describe_enumerators<T>::value>>
std::ostream& operator<<(std::ostream& os, const T& t)
{
    os << boost::describe::enum_to_string(t, "Unknown");
    return os;
}

template <typename T, typename = typename std::enable_if_t<boost::describe::has_describe_enumerators<T>::value>>
std::istream& operator>>(std::istream& is, T& t)
{
    std::string tmp;
    is >> tmp;
    bool success = boost::describe::enum_from_string(tmp.c_str(), t);
    if(not success) { throw std::runtime_error(std::string("Invalid enumerator name '") + tmp + "'"); }
    return is;
}

namespace Opts {

template <typename T, typename = typename std::enable_if_t<boost::describe::has_describe_enumerators<T>::value>>
std::ostream& operator<<(std::ostream& os, const T& t)
{
    os << boost::describe::enum_to_string(t, "Unknown");
    return os;
}

template <typename T, typename = typename std::enable_if_t<boost::describe::has_describe_enumerators<T>::value>>
std::istream& operator>>(std::istream& is, T& t)
{
    std::string tmp;
    is >> tmp;
    bool success = boost::describe::enum_from_string(tmp.c_str(), t);
    if(not success) { throw std::runtime_error(std::string("Invalid enumerator name '") + tmp + "'"); }
    return is;
}

} // namespace Opts
} // namespace Xped
#endif
