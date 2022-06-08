#ifndef XPED_OPTIM_OPTS_HPP
#define XPED_OPTIM_OPTS_HPP

namespace Xped::Opts {

struct Optim
{
    // Algorithm alg;

    // LineSearch ls;

    double bfgs_xxx;

    double grad_tol;
    double step_tol;
    std::size_t max_steps;
    std::size_t min_steps;
};

} // namespace Xped::Opts
#endif
