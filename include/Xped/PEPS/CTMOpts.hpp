#ifndef XPED_CTM_OPTS_HPP
#define XPED_CTM_OPTS_HPP

namespace Xped::Opts {

struct CTM
{
    std::size_t chi;

    std::size_t max_presteps = 100;
    std::size_t track_steps = 4;

    double tol_E = 1.e-10;
    double tol_N;

    double reinit_env_tol = 1.e-1;
};

} // namespace Xped::Opts
#endif
