#ifndef XPED_CTM_OPTS_HPP
#define XPED_CTM_OPTS_HPP

#include "toml.hpp"

#include "fmt/color.h"
#include "fmt/core.h"

#include <boost/describe.hpp>

#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/Util/Logging.hpp"

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
                           ("verbosity", verbosity));
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
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• reinit_env_tol:", reinit_env_tol);
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
    // res.cell = t.contains("cell") ? UnitCell(toml::get<std::vector<std::vector<std::string>>>(toml::find(t, "cell"))) : UnitCell();
    return res;
}

struct CTMCheckpoint
{
    bool GROW_ALL = false;
    bool MOVE = false;
    bool CORNER = false;
    bool PROJECTORS = false;
    bool RENORMALIZE = false;

    auto info() const
    {
        std::string res;
        fmt::format_to(std::back_inserter(res), "{}:\n", fmt::styled("Checkpointing settings", fmt::emphasis::bold));
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• grow_all:", GROW_ALL);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• move:", MOVE);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• corner contraction:", CORNER);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}\n", "• projector computation:", PROJECTORS);
        fmt::format_to(std::back_inserter(res), "  {:<30} {}", "• Renormalization step:", RENORMALIZE);
        return res;
    }
};

} // namespace Xped::Opts
#endif
