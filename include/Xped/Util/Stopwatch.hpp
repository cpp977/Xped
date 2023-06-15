#ifndef XPED_STOPWATCH_HPP_
#define XPED_STOPWATCH_HPP_

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

#include "fmt/chrono.h"
#include "fmt/core.h"

namespace Xped::util {

enum class TimeUnit
{
    MILLISECONDS,
    SECONDS,
    MINUTES,
    HOURS,
    DAYS,
    NATURAL
};

std::string format_secs(std::chrono::duration<double, std::ratio<1, 1>> dts)
{
    if(dts.count() < 60.) {
        return fmt::format("{:.2}", dts);
    } else if(dts.count() >= 60. && dts.count() < 3600.) {
        std::chrono::duration<double, std::ratio<60, 1>> dtm = dts;
        return fmt::format("{:.2}", dtm);
    } else if(dts.count() >= 3600. && dts.count() < 86400.) {
        std::chrono::duration<double, std::ratio<3600, 1>> dth = dts;
        return fmt::format("{:.2}", dth);
    }
    std::chrono::duration<double, std::ratio<86400, 1>> dtd = dts;
    return fmt::format("{:.2}d", dtd.count());
}

template <typename ClockClass = std::chrono::high_resolution_clock>
class Stopwatch
{
public:
    Stopwatch();
    Stopwatch(std::string filename_input);

    std::string time_string(TimeUnit u = TimeUnit::NATURAL);
    std::chrono::seconds time();
    double seconds();

    void start();

    template <typename ThemeType>
    std::string info(ThemeType theme, bool RESTART = true);
    inline std::string info();

    template <typename ThemeType>
    void check(ThemeType theme);
    void check();

    template <typename ThemeType, class F, class... ArgTypes>
    std::invoke_result_t<F && (ArgTypes && ...)> runTime(ThemeType theme, F&& f, ArgTypes&&... args);

private:
    std::chrono::time_point<ClockClass> t_start, t_end;
    std::string filename;
    bool SAVING_TO_FILE;
};

template <typename ClockClass>
Stopwatch<ClockClass>::Stopwatch()
{
    SAVING_TO_FILE = false;
    start();
}

template <typename ClockClass>
Stopwatch<ClockClass>::Stopwatch(std::string filename_input)
{
    filename = filename_input;
    std::ofstream file(filename, std::ios::trunc); // delete file contents
    std::fstream outfile;
    outfile.open(filename, std::fstream::app | std::fstream::out);
    outfile.close();
    SAVING_TO_FILE = true;
    start();
}

template <typename ClockClass>
std::string Stopwatch<ClockClass>::time_string(TimeUnit u)
{
    t_end = ClockClass::now();
    std::chrono::duration<double, std::ratio<1, 1>> dt = t_end - t_start;
    if(u == TimeUnit::MILLISECONDS) {
        std::chrono::duration<double, std::ratio<1, 1000>> dtms = dt;
        return fmt::format("{}", dtms);
    } else if(u == TimeUnit::SECONDS) {
        return fmt::format("{}", dt);
    } else if(u == TimeUnit::MINUTES) {
        std::chrono::duration<double, std::ratio<60, 1>> dtm = dt;
        return fmt::format("{}", dtm);
    } else if(u == TimeUnit::HOURS) {
        std::chrono::duration<double, std::ratio<3600, 1>> dth = dt;
        return fmt::format("{}", dth);
    } else if(u == TimeUnit::DAYS) {
        std::chrono::duration<double, std::ratio<86400, 1>> dtd = dt;
        return fmt::format("{}d", dtd.count());
    }
    return format_secs(dt);
}

template <typename ClockClass>
std::chrono::seconds Stopwatch<ClockClass>::time()
{
    t_end = ClockClass::now();
    return std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start);
}

template <typename ClockClass>
double Stopwatch<ClockClass>::seconds()
{
    t_end = ClockClass::now();
    return std::chrono::duration<double, std::ratio<1, 1>>(t_end - t_start).count();
}

template <typename ClockClass>
inline void Stopwatch<ClockClass>::start()
{
    t_start = ClockClass::now();
}

template <typename ClockClass>
template <typename ThemeType>
void Stopwatch<ClockClass>::check(ThemeType theme)
{
    std::cout << info(theme) << std::endl;
}

template <typename ClockClass>
template <typename ThemeType>
std::string Stopwatch<ClockClass>::info(ThemeType theme, bool RESTART)
{
    t_end = ClockClass::now();

    std::chrono::duration<double, std::ratio<1, 1>> dtTest = t_end - t_start;

    std::stringstream ss;
    if(dtTest.count() < 60.) {
        std::chrono::duration<double, std::ratio<1, 1>> dt = t_end - t_start;
        ss << theme << ": " << dt.count() << " #s";
    } else if(dtTest.count() >= 60. && dtTest.count() < 3600.) {
        std::chrono::duration<double, std::ratio<60, 1>> dt = t_end - t_start;
        ss << theme << ": " << dt.count() << " #min";
    } else if(dtTest.count() >= 3600. && dtTest.count() < 86400.) {
        std::chrono::duration<double, std::ratio<3600, 1>> dt = t_end - t_start;
        ss << theme << ": " << dt.count() << " #h";
    } else {
        std::chrono::duration<double, std::ratio<86400, 1>> dt = t_end - t_start;
        ss << theme << ": " << dt.count() << " #d";
    }

    if(SAVING_TO_FILE == true) {
        std::fstream outfile;
        outfile.open(filename, std::fstream::app | std::fstream::out);
        outfile << ss.str() << std::endl;
        outfile.close();
    }

    if(RESTART) { start(); }
    return ss.str();
}

template <typename ClockClass>
template <typename ThemeType, class F, class... ArgTypes>
std::invoke_result_t<F && (ArgTypes && ...)> Stopwatch<ClockClass>::runTime(ThemeType theme, F&& f, ArgTypes&&... args)
{
    start();
    if constexpr(std::is_void<std::invoke_result_t<F && (ArgTypes && ...)>>::value) {
        return std::invoke(std::forward<F>(f), std::forward<ArgTypes>(args)...);
        // if (SAVING_TO_FILE == false) { std::cout << info(theme) << std::endl; }
        // else { info(theme); }
    } else {
        std::invoke_result_t<F && (ArgTypes && ...)> result = std::invoke(std::forward<F>(f), std::forward<ArgTypes>(args)...);
        if(SAVING_TO_FILE == false) {
            std::cout << info(theme) << std::endl;
        } else {
            info(theme);
        }
        return result;
    }
}

template <typename ClockClass>
inline void Stopwatch<ClockClass>::check()
{
    check("");
}

template <typename ClockClass>
inline std::string Stopwatch<ClockClass>::info()
{
    return info("");
}

} // namespace Xped::util
#endif
