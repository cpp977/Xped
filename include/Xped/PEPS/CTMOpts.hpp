#ifndef XPED_CTM_OPTS_HPP
#define XPED_CTM_OPTS_HPP

#include "toml.hpp"

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/ostream.h"

#include <boost/describe.hpp>

#include "Xped/Util/Logging.hpp"
#include "Xped/Util/TomlHelpers.hpp"

namespace Xped::Opts {

BOOST_DEFINE_ENUM_CLASS(DIRECTION, LEFT, RIGHT, TOP, BOTTOM)

BOOST_DEFINE_ENUM_CLASS(CORNER, UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT)

BOOST_DEFINE_ENUM_CLASS(PROJECTION, CORNER, HALF, FULL)

BOOST_DEFINE_ENUM_CLASS(CTM_INIT, FROM_TRIVIAL, FROM_A)

struct CTM
{
    std::size_t chi;

    std::size_t max_presteps = 100;
    std::size_t track_steps = 4;

    double tol_E = 1.e-10;
    double tol_N = 1.e-10;

    double reinit_env_tol = 1.e-1;

    CTM_INIT init = CTM_INIT::FROM_A;

    Verbosity verbosity = Verbosity::ON_ENTRY;

    std::string load = "";
    int qn_scale = 1;

    bool COMPARE_TO_FD = false;

    bool EXPORT_CTM_SPECTRA = false;

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("CTMOpts",
                           ("chi", chi),
                           ("max_presteps", max_presteps),
                           ("track_steps", track_steps),
                           ("tol_E", tol_E),
                           ("tol_N", tol_N),
                           ("reinit_env_tol", reinit_env_tol),
                           ("init", init),
                           ("verbosity", verbosity),
                           ("load", load),
                           ("qn_scale", qn_scale),
                           ("COMPARE_TO_FD", COMPARE_TO_FD),
                           ("EXPORT_CTM_SPECTRA", EXPORT_CTM_SPECTRA));
    }

    inline auto info()
    {
        std::string res;
        fmt::format_to(std::back_inserter(res), "{}:\n", fmt::styled("CTM options", fmt::emphasis::bold));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• chi:", chi);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• init:", fmt::streamed(init));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• maximum pre-steps:", max_presteps);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• tracked steps:", track_steps);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• energy tolerance:", tol_E);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• norm tolerance:", tol_N);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• compare ad to fd:", COMPARE_TO_FD);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• export spectra if corner matrix:", EXPORT_CTM_SPECTRA);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• reinit_env_tol:", reinit_env_tol);
        if(load.size() > 0) { fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• load from:", load); }
        if(load.size() > 0) { fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• scale loaded qn by:", qn_scale); }
        fmt::format_to(std::back_inserter(res), "  {:<30} {}", "• verbosity:", fmt::streamed(verbosity));
        return res;
    }
};

inline CTM ctm_from_toml(const toml::value& t)
{
    CTM res{};
    res.chi = t.contains("chi") ? t.at("chi").as_integer() : res.chi;
    if(t.contains("init")) { res.init = util::enum_from_toml<CTM_INIT>(t.at("init")); }
    res.max_presteps = t.contains("max_presteps") ? t.at("max_presteps").as_integer() : res.max_presteps;
    res.track_steps = t.contains("track_steps") ? t.at("track_steps").as_integer() : res.track_steps;
    res.tol_E = t.contains("tol_E") ? t.at("tol_E").as_floating() : res.tol_E;
    res.reinit_env_tol = t.contains("reinit_env_tol") ? t.at("reinit_env_tol").as_floating() : res.reinit_env_tol;
    if(t.contains("verbosity")) { res.verbosity = util::enum_from_toml<Verbosity>(t.at("verbosity")); }
    res.load = t.contains("load") ? static_cast<std::string>(t.at("load").as_string()) : res.load;
    res.qn_scale = t.contains("qn_scale") ? (t.at("qn_scale").as_integer()) : res.qn_scale;
    res.COMPARE_TO_FD = t.contains("COMPARE_TO_FD") ? t.at("COMPARE_TO_FD").as_boolean() : res.COMPARE_TO_FD;
    res.EXPORT_CTM_SPECTRA = t.contains("EXPORT_CTM_SPECTRA") ? t.at("EXPORT_CTM_SPECTRA").as_boolean() : res.EXPORT_CTM_SPECTRA;
    return res;
}

struct CTMCheckpoint
{
    bool GROW_ALL = false;
    bool MOVE = false;
    bool CORNER = false;
    bool PROJECTORS = false;
    bool RENORMALIZE = false;
    bool RDM = false;

    auto info() const
    {
        std::string res;
        fmt::format_to(std::back_inserter(res), "{}:\n", fmt::styled("Checkpointing settings", fmt::emphasis::bold));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• grow_all:", GROW_ALL);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• move:", MOVE);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• corner contraction:", CORNER);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• projector computation:", PROJECTORS);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• Renormalization step:", RENORMALIZE);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}", "• RDM contraction:", RDM);
        return res;
    }
};

} // namespace Xped::Opts
#endif
