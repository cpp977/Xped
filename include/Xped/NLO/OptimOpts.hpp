#ifndef XPED_OPTIM_OPTS_HPP
#define XPED_OPTIM_OPTS_HPP

namespace Xped::Opts {

enum class Algorithm
{
    L_BFGS,
    CONJUGATE_GRADIENT,
    NELDER_MEAD
};

enum class Linesearch
{
    WOLFE,
    ARMIJO
};

struct Optim
{
    Algorithm alg = Algorithm::L_BFGS;

    Linesearch ls = Linesearch::WOLFE;

    double bfgs_xxx;

    double grad_tol;
    double step_tol;
    std::size_t max_steps;
    std::size_t min_steps;
};

} // namespace Xped::Opts
#endif
