#ifndef XPED_IMAG_OPTS_H_
#define XPED_IMAG_OPTS_H_

#include "toml.hpp"

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "fmt/color.h"
#include "fmt/core.h"

#include <boost/describe.hpp>

#include "Xped/Util/Logging.hpp"
#include "Xped/Util/TomlHelpers.hpp"

namespace Xped {

struct TimeProtocol
{
    std::vector<std::size_t> t_steps;
    std::vector<double> dts;
};

namespace Opts {

BOOST_DEFINE_ENUM_CLASS(Update, SIMPLE, FULL, CLUSTER)

struct Imag
{
    std::vector<std::size_t> Ds = {2ul, 3ul, 4ul};
    std::vector<std::vector<std::size_t>> chis = {{20ul}, {30ul}, {40ul}};

    std::vector<std::size_t> t_steps = {100ul, 100ul, 100ul, 100ul, 100ul, 100ul, 100ul, 100ul, 100ul};
    std::vector<double> dts = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001};

    Update update = Update::SIMPLE;

    std::size_t max_steps = 50;
    std::size_t min_steps = 0;

    double tol = 1.e-10;

    bool resume = false;

    std::string load = "";
    LoadFormat load_format = LoadFormat::NATIVE;
    int qn_scale = 1;

    std::size_t seed = 0ul;

    std::size_t id = 1ul;

    Verbosity verbosity = Verbosity::ON_ENTRY;

    std::filesystem::path working_directory = std::filesystem::current_path();
    std::filesystem::path logging_directory = "logs";
    std::string log_format = ".log";

    std::filesystem::path obs_directory = "obs";
    bool display_obs = true;

    bool multi_init = false;
    std::vector<std::size_t> init_seeds = {0, 1, 2};
    std::vector<std::size_t> init_Ds = {2ul, 3ul, 2ul};
    std::vector<std::size_t> init_t_steps = {40ul, 40ul, 40ul};
    std::vector<double> init_dts = {0.1, 0.01, 0.001};
    std::size_t init_chi = 28;

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("ImagOpts",
                           ("Ds", Ds),
                           ("chis", chis),
                           ("t_steps", t_steps),
                           ("dts", dts),
                           ("max_steps", max_steps),
                           ("min_steps", min_steps),
                           ("update", update),
                           ("tol", tol),
                           ("verbosity", verbosity),
                           ("working_directory", working_directory),
                           ("logging_directory", logging_directory),
                           ("log_format", log_format),
                           ("obs_directory", obs_directory),
                           ("load", load),
                           ("load format", load_format),
                           ("seed", seed),
                           ("id", id),
                           ("display_obs", display_obs),
                           ("qn_scale", qn_scale),
                           ("multi_init", multi_init),
                           ("init_t_steps", init_t_steps),
                           ("init_dts", init_dts),
                           ("init_seeds", init_seeds),
                           ("init_Ds", init_Ds),
                           ("init_chi", init_chi));
        assert(Ds.size() == chis.size());
        assert(dts.size() == t_steps.size());
    }

    inline auto info()
    {
        std::string res;
        fmt::format_to(std::back_inserter(res), "{}:\n", fmt::styled("Imaginary time-evolution options", fmt::emphasis::bold));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• Ds:", Ds);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• chis:", chis);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• t_steps:", t_steps);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• dts:", dts);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• update:", fmt::streamed(update));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• maximum time steps:", max_steps);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• minimum time steps:", min_steps);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• tolerance:", tol);
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
        fmt::format_to(std::back_inserter(res), "  {:<30} {}", "• verbosity:", fmt::streamed(verbosity));
        return res;
    }
};

inline Imag imag_from_toml(const toml::value& t)
{
    Imag res{};
    res.Ds = t.contains("Ds") ? toml::find<std::vector<std::size_t>>(t, "Ds") : res.Ds;
    res.chis = t.contains("chis") ? toml::find<std::vector<std::vector<std::size_t>>>(t, "chis") : res.chis;
    if(t.contains("update")) { res.update = util::enum_from_toml<Update>(t.at("update")); }
    res.max_steps = t.contains("max_steps") ? t.at("max_steps").as_integer() : res.max_steps;
    res.min_steps = t.contains("min_steps") ? t.at("min_steps").as_integer() : res.min_steps;
    res.dts = t.contains("dts") ? toml::find<std::vector<double>>(t, "dts") : res.dts;
    res.t_steps =
        t.contains("t_steps") ? toml::find<std::vector<std::size_t>>(t, "t_steps") : std::vector<std::size_t>(res.dts.size(), res.max_steps);

    res.tol = t.contains("tol") ? t.at("tol").as_floating() : res.tol;
    res.resume = t.contains("resume") ? t.at("resume").as_boolean() : res.resume;
    if(t.contains("verbosity")) { res.verbosity = util::enum_from_toml<Verbosity>(t.at("verbosity")); }
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
    res.display_obs = t.contains("display_obs") ? t.at("display_obs").as_boolean() : res.display_obs;
    res.load = t.contains("load") ? static_cast<std::string>(t.at("load").as_string()) : res.load;
    if(t.contains("load_format")) { res.load_format = util::enum_from_toml<LoadFormat>(t.at("load_format")); }
    res.qn_scale = t.contains("qn_scale") ? (t.at("qn_scale").as_integer()) : res.qn_scale;
    res.seed = t.contains("seed") ? (t.at("seed").as_integer()) : res.seed;
    res.id = t.contains("id") ? (t.at("id").as_integer()) : res.id;
    res.multi_init = t.contains("multi_init") ? (t.at("multi_init").as_boolean()) : res.multi_init;
    if(res.multi_init) { res.init_dts = t.contains("init_dts") ? toml::find<std::vector<double>>(t, "init_dts") : res.init_dts; }
    if(res.multi_init) { res.init_t_steps = t.contains("init_t_steps") ? toml::find<std::vector<std::size_t>>(t, "init_t_steps") : res.init_t_steps; }
    if(res.multi_init) { res.init_seeds = t.contains("init_seeds") ? toml::find<std::vector<std::size_t>>(t, "init_seeds") : res.init_seeds; }
    if(res.multi_init) { res.init_chi = t.contains("init_chi") ? t.at("init_chi").as_integer() : res.init_chi; }
    if(res.multi_init) { res.init_Ds = t.contains("init_Ds") ? toml::find<std::vector<std::size_t>>(t, "init_Ds") : res.init_Ds; }
    return res;
}

} // namespace Opts

} // namespace Xped

#endif
