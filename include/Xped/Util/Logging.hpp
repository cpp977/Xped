#ifndef XPED_LOGGING_HPP_
#define XPED_LOGGING_HPP_

#include "fmt/core.h"

#include "spdlog/spdlog.h"

#include <boost/algorithm/string.hpp>
#include <boost/describe.hpp>

namespace Xped {

BOOST_DEFINE_ENUM_CLASS(Verbosity, SILENT, CRITICAL, WARNING, ON_EXIT, ON_ENTRY, PER_ITERATION, DEBUG)

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
        std::string message = fmt::vformat(fmt, fmt::make_format_args(std::forward<Args>(args)...));
        std::vector<std::string> lines;
        boost::split(lines, message, boost::is_any_of("\n"));
        switch(verb) {
        case Verbosity::DEBUG: {
            for(const auto& line : lines) { spdlog::trace("{}", line); }
            break;
        }
        case Verbosity::PER_ITERATION: {
            for(const auto& line : lines) { spdlog::debug("{}", line); }
            break;
        }
        case Verbosity::ON_ENTRY: {

            for(const auto& line : lines) { spdlog::info("{}", line); }
            break;
        }
        case Verbosity::ON_EXIT: {
            for(const auto& line : lines) { spdlog::warn("{}", line); }
            break;
        }
        case Verbosity::WARNING: {
            for(const auto& line : lines) { spdlog::error("{}", line); }
            break;
        }
        case Verbosity::CRITICAL: {
            for(const auto& line : lines) { spdlog::critical("{}", line); }
            break;
        }
        case Verbosity::SILENT: {
            break;
        }
        }
    }
}

template <typename... Args>
constexpr void critical(Verbosity policy, Args&&... args)
{
    return log(Verbosity::CRITICAL, policy, args...);
}

template <typename... Args>
constexpr void warning(Verbosity policy, Args&&... args)
{
    return log(Verbosity::WARNING, policy, args...);
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
