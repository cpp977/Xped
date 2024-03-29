#ifndef XPED_OPTIM_OPTS_HPP
#define XPED_OPTIM_OPTS_HPP

#include <boost/describe.hpp>

#include "fmt/color.h"

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/Util/EnumStream.hpp"
#include "Xped/Util/Logging.hpp"
#include "Xped/Util/TomlHelpers.hpp"
#include "Xped/Util/YasHelpers.hpp"

#include "Xped/PEPS/iPEPSOpts.hpp"

namespace Xped::Opts {

BOOST_DEFINE_ENUM_CLASS(Algorithm, L_BFGS, CONJUGATE_GRADIENT, NELDER_MEAD)

BOOST_DEFINE_ENUM_CLASS(Linesearch, WOLFE, ARMIJO)

struct Optim
{
    Algorithm alg = Algorithm::L_BFGS;

    Linesearch ls = Linesearch::WOLFE;

    double bfgs_xxx;
    bool bfgs_scaling = true;
    std::size_t max_lbfgs_rank = 20ul;

    double grad_tol = 1.e-6;
    double step_tol = 1.e-12;
    double cost_tol = 1.e-12;

    std::size_t max_steps = 200;
    std::size_t min_steps = 10;

    bool resume = false;

    LoadFormat load_format = LoadFormat::Native;
    std::string load = "";
    int qn_scale = 1;

    std::size_t seed = 0;

    std::size_t id = 1ul;

    std::size_t save_period = 0;

    std::string log_format = ".log";

    std::filesystem::path working_directory = std::filesystem::current_path();
    std::filesystem::path logging_directory = "logs";

    std::filesystem::path obs_directory = "obs";
    bool display_obs = true;

    Verbosity verbosity = Verbosity::PER_ITERATION;

    std::size_t restarts = 1ul;

    std::vector<std::size_t> warmup_chis = {};

    template <typename Ar>
    void serialize(Ar& ar)
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
                           ("load", load),
                           ("load format", load_format),
                           ("qn_scale", qn_scale),
                           ("seed", seed),
                           ("id", id),
                           ("save_period", save_period),
                           ("log_format", log_format),
                           ("working_directory", working_directory),
                           ("logging_directory", logging_directory),
                           ("obs_directory", obs_directory),
                           ("verbosity", verbosity),
                           ("display_obs", display_obs),
                           ("restarts", restarts),
                           ("max_lbfgs_rank", max_lbfgs_rank),
                           ("warmup_chis", warmup_chis));
    }

    inline auto info()
    {
        std::string res;
        fmt::format_to(std::back_inserter(res), "{}:\n", fmt::styled("Optimization options", fmt::emphasis::bold));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• Algorithm:", fmt::streamed(alg));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• Linesearch:", fmt::streamed(ls));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• maximum steps:", max_steps);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• minimum steps:", min_steps);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• gradient tolerance:", grad_tol);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• cost tolerance:", cost_tol);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• step tolerance:", step_tol);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• max L-BFGS rank:", max_lbfgs_rank);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• bfgs scaling:", bfgs_scaling);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• resume:", resume);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• log format:", log_format);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• working directory:", working_directory.string());
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• logging directory:", logging_directory.string());
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• obs directory:", obs_directory.string());
        if(load.size() > 0) { fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• load from:", load); }
        if(load.size() > 0) { fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• scale loaded qn by:", qn_scale); }
        if(load.size() > 0) { fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• load format:", fmt::streamed(load_format)); }
        if(load.size() == 0) { fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• seed:", seed); }
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• id:", id);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• save period:", save_period);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• verbosity:", fmt::streamed(verbosity));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• display obs to terminal:", display_obs);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• restarts:", restarts);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}", "• warmup_chis:", restarts);
        return res;
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
    res.max_lbfgs_rank = t.contains("max_lbfgs_rank") ? t.at("max_lbfgs_rank").as_integer() : res.max_lbfgs_rank;
    res.resume = t.contains("resume") ? t.at("resume").as_boolean() : res.resume;
    res.load = t.contains("load") ? static_cast<std::string>(t.at("load").as_string()) : res.load;
    res.qn_scale = t.contains("qn_scale") ? (t.at("qn_scale").as_integer()) : res.qn_scale;
    if(t.contains("load_format")) { res.load_format = util::enum_from_toml<LoadFormat>(t.at("load_format")); }
    res.save_period = t.contains("save_period") ? t.at("save_period").as_integer() : res.save_period;
    res.log_format = t.contains("log_format") ? static_cast<std::string>(t.at("log_format").as_string()) : res.log_format;
    if(t.contains("working_directory")) {
        std::filesystem::path tmp_wd(static_cast<std::string>(t.at("working_directory").as_string()));
        if(tmp_wd.is_relative()) {
            res.working_directory = std::filesystem::current_path() / tmp_wd;
        } else {
            res.working_directory = tmp_wd;
        }
    }
    if(t.contains("logging_directory")) {
        res.logging_directory = std::filesystem::path(static_cast<std::string>(t.at("logging_directory").as_string()));
    }
    if(t.contains("obs_directory")) { res.obs_directory = std::filesystem::path(static_cast<std::string>(t.at("obs_directory").as_string())); }
    if(t.contains("verbosity")) { res.verbosity = util::enum_from_toml<Verbosity>(t.at("verbosity")); }
    res.display_obs = t.contains("display_obs") ? t.at("display_obs").as_boolean() : res.display_obs;
    res.seed = t.contains("seed") ? (t.at("seed").as_integer()) : res.seed;
    res.id = t.contains("id") ? (t.at("id").as_integer()) : res.id;
    res.restarts = t.contains("restarts") ? (t.at("restarts").as_integer()) : res.restarts;
    res.warmup_chis = t.contains("warmup_chis") ? toml::find<std::vector<std::size_t>>(t, "warmup_chis") : res.warmup_chis;
    return res;
}

} // namespace Xped::Opts
#endif
