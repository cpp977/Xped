#ifndef XPED_OPTIM_OPTS_HPP
#define XPED_OPTIM_OPTS_HPP

#include <boost/describe.hpp>

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

    inline void info()
    {
        fmt::print("Optimization options:\n");
        fmt::print("  {:<20} {}\n", "• Algorithm:", alg);
        fmt::print("  {:<20} {}\n", "• Linesearch:", ls);
        fmt::print("  {:<20} {}\n", "• maximum steps:", max_steps);
        fmt::print("  {:<20} {}\n", "• minimum steps:", min_steps);
        fmt::print("  {:<20} {}\n", "• gradient tolerance:", grad_tol);
        fmt::print("  {:<20} {}\n", "• cost tolerance:", cost_tol);
        fmt::print("  {:<20} {}\n", "• step tolerance:", step_tol);
        fmt::print("  {:<20} {}\n", "• bfgs scaling:", bfgs_scaling);
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
    return res;
}

} // namespace Xped::Opts
#endif
