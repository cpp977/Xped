#ifndef XPED_LOGGING_HPP_
#define XPED_LOGGING_HPP_

#include "fmt/core.h"

#include "spdlog/spdlog.h"

#include <boost/describe.hpp>

namespace Xped {

BOOST_DEFINE_ENUM_CLASS(Verbosity, SILENT, ON_EXIT, ON_ENTRY, PER_ITERATION, DEBUG)

namespace Log {

// template <typename... Args>
// auto log(Verbosity verb, Verbosity policy, fmt::format_string<Args...> s, Args&&... args)
// {
//     if(verb <= policy) {
//         fmt::print(s, std::forward<Args>(args)...);
//         fmt::print("\n");
//     }
// }

template <typename... Args>
auto log(Verbosity verb, Verbosity policy, std::string_view fmt, Args&&... args)
{
    if(verb <= policy) {
        spdlog::info("{}", fmt::vformat(fmt, fmt::make_format_args(std::forward<Args>(args)...)));
        // fmt::vprint(fmt, fmt::make_format_args(std::forward<Args>(args)...));
        // fmt::print("\n");
    }
}

template <typename... Args>
constexpr void on_exit(Verbosity policy, Args&&... args)
{
    return log(Verbosity::ON_EXIT, policy, args...);
}

template <typename... Args>
constexpr void on_entry(Verbosity policy, Args&&... args)
{
    return log(Verbosity::ON_ENTRY, policy, args...);
}

template <typename... Args>
constexpr void per_iteration(Verbosity policy, Args&&... args)
{
    return log(Verbosity::PER_ITERATION, policy, args...);
}

template <typename... Args>
constexpr void debug(Verbosity policy, Args&&... args)
{
    return log(Verbosity::DEBUG, policy, args...);
}

} // namespace Log
} // namespace Xped

#endif
