#ifndef XPED_OPTIM_OPTS_HPP
#define XPED_OPTIM_OPTS_HPP

#include <boost/describe.hpp>

#include "fmt/color.h"

#include "Xped/Util/EnumStream.hpp"
#include "Xped/Util/TomlHelpers.hpp"

namespace Xped::Opts {

BOOST_DEFINE_ENUM_CLASS(Algorithm, L_BFGS, CONJUGATE_GRADIENT, NELDER_MEAD)

BOOST_DEFINE_ENUM_CLASS(Linesearch, WOLFE, ARMIJO)

struct Optim
{
    Algorithm alg = Algorithm::L_BFGS;

    Linesearch ls = Linesearch::WOLFE;

    double bfgs_xxx;
    bool bfgs_scaling = true;

    double grad_tol = 1.e-6;
    double step_tol = 1.e-12;
    double cost_tol = 1.e-12;

    std::size_t max_steps = 200;
    std::size_t min_steps = 10;

    bool resume = false;
    std::string load = "";

    std::size_t save_period = 0;

    std::string log_format = ".log";

    std::filesystem::path working_directory = std::filesystem::current_path();
    std::filesystem::path logging_directory = "logs";

    template <typename Ar>
    void serialize(Ar& ar) const
    {
        ar& YAS_OBJECT_NVP("OptimOpts",
                           ("alg", alg),
                           ("LineSearch", ls),
                           ("bfgs_scaling", bfgs_scaling),
                           ("grad_tol", grad_tol),
                           ("step_tol", step_tol),
                           ("cost_tol", cost_tol),
                           ("max_steps", max_steps),
                           ("min_steps", min_steps),
                           // ("resume", resume),
                           ("load", load),
                           ("save_period", save_period),
                           ("log_format", log_format),
                           ("working_directory", working_directory.string()),
                           ("logging_directory", logging_directory.string()));
    }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        std::string working_dir, logging_dir;
        ar& YAS_OBJECT_NVP("OptimOpts",
                           ("alg", alg),
                           ("LineSearch", ls),
                           ("bfgs_scaling", bfgs_scaling),
                           ("grad_tol", grad_tol),
                           ("step_tol", step_tol),
                           ("cost_tol", cost_tol),
                           ("max_steps", max_steps),
                           ("min_steps", min_steps),
                           // ("resume", resume),
                           ("load", load),
                           ("save_period", save_period),
                           ("log_format", log_format),
                           ("working_directory", working_dir),
                           ("logging_directory", logging_dir));
        working_directory = working_dir;
        logging_directory = logging_dir;
    }

    inline void info()
    {
        fmt::print("{}:\n", fmt::styled("Optimization options", fmt::emphasis::bold));
        fmt::print("  {:<30} {}\n", "• Algorithm:", fmt::streamed(alg));
        fmt::print("  {:<30} {}\n", "• Linesearch:", fmt::streamed(ls));
        fmt::print("  {:<30} {}\n", "• maximum steps:", max_steps);
        fmt::print("  {:<30} {}\n", "• minimum steps:", min_steps);
        fmt::print("  {:<30} {}\n", "• gradient tolerance:", grad_tol);
        fmt::print("  {:<30} {}\n", "• cost tolerance:", cost_tol);
        fmt::print("  {:<30} {}\n", "• step tolerance:", step_tol);
        fmt::print("  {:<30} {}\n", "• bfgs scaling:", bfgs_scaling);
        fmt::print("  {:<30} {}\n", "• resume:", resume);
        fmt::print("  {:<30} {}\n", "• log format:", log_format);
        fmt::print("  {:<30} {}\n", "• working directory:", working_directory.string());
        fmt::print("  {:<30} {}\n", "• logging directory:", logging_directory.string());
        if(load.size() > 0) { fmt::print("  {:<30} {}\n", "• load from:", load); }
        fmt::print("  {:<30} {}\n", "• save period:", save_period);
    }
};

inline Optim optim_from_toml(const toml::value& t)
{
    Optim res{};
    if(t.contains("algorithm")) { res.alg = util::enum_from_toml<Algorithm>(t.at("algorithm")); }
    if(t.contains("linesearch")) { res.ls = util::enum_from_toml<Linesearch>(t.at("linesearch")); }
    res.grad_tol = t.contains("grad_tol") ? t.at("grad_tol").as_floating() : res.grad_tol;
    res.step_tol = t.contains("step_tol") ? t.at("step_tol").as_floating() : res.step_tol;
    res.cost_tol = t.contains("cost_tol") ? t.at("cost_tol").as_floating() : res.cost_tol;
    res.max_steps = t.contains("max_steps") ? t.at("max_steps").as_integer() : res.max_steps;
    res.min_steps = t.contains("min_steps") ? t.at("min_steps").as_integer() : res.min_steps;
    res.bfgs_scaling = t.contains("bfgs_scaling") ? t.at("bfgs_scaling").as_boolean() : res.bfgs_scaling;
    res.resume = t.contains("resume") ? t.at("resume").as_boolean() : res.resume;
    res.load = t.contains("load") ? static_cast<std::string>(t.at("load").as_string()) : res.load;
    res.save_period = t.contains("save_period") ? t.at("save_period").as_integer() : res.save_period;
    res.log_format = t.contains("log_format") ? static_cast<std::string>(t.at("log_format").as_string()) : res.log_format;
    if(t.contains("working_directory")) {
        res.working_directory = std::filesystem::path(static_cast<std::string>(t.at("working_directory").as_string()));
    }
    if(t.contains("logging_directory")) {
        res.logging_directory = std::filesystem::path(static_cast<std::string>(t.at("logging_directory").as_string()));
    }
    return res;
}

} // namespace Xped::Opts
#endif
