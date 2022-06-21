#ifndef XPED_CTM_OPTS_HPP
#define XPED_CTM_OPTS_HPP

#include "toml.hpp"

#include "Xped/PEPS/UnitCell.hpp"

namespace Xped::Opts {

struct CTM
{
    std::size_t chi;

    std::size_t max_presteps = 100;
    std::size_t track_steps = 4;

    double tol_E = 1.e-10;
    double tol_N = 1.e-10;

    double reinit_env_tol = 1.e-1;

    inline void info()
    {
        fmt::print("CTM options:\n");
        fmt::print("  {:<20} {}\n", "• chi:", chi);
        fmt::print("  {:<20} {}\n", "• maximum pre-steps:", max_presteps);
        fmt::print("  {:<20} {}\n", "• tracked steps:", track_steps);
        fmt::print("  {:<20} {}\n", "• energy tolerance:", tol_E);
        fmt::print("  {:<20} {}\n", "• norm tolerance:", tol_N);
        fmt::print("  {:<20} {}\n", "• reinit_env_tol:", reinit_env_tol);
    }
};

inline CTM ctm_from_toml(const toml::value& t)
{
    CTM res{};
    res.chi = t.contains("chi") ? t.at("chi").as_integer() : res.chi;
    res.max_presteps = t.contains("max_presteps") ? t.at("max_presteps").as_integer() : res.max_presteps;
    res.track_steps = t.contains("track_steps") ? t.at("track_steps").as_integer() : res.track_steps;
    res.tol_E = t.contains("tol_E") ? t.at("tol_E").as_floating() : res.tol_E;
    res.reinit_env_tol = t.contains("reinit_env_tol") ? t.at("reinit_env_tol").as_floating() : res.reinit_env_tol;
    // res.cell = t.contains("cell") ? UnitCell(toml::get<std::vector<std::vector<std::string>>>(toml::find(t, "cell"))) : UnitCell();
    return res;
}

} // namespace Xped::Opts
#endif
