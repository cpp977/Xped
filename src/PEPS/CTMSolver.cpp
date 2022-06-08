#include <limits>

#include "Xped/PEPS/CTMSolver.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
template <typename HamScalar>
typename ScalarTraits<Scalar>::Real CTMSolver<Scalar, Symmetry>::solve(const std::shared_ptr<iPEPS<Scalar, Symmetry>>& Psi,
                                                                       Scalar* gradient,
                                                                       const Hamiltonian<HamScalar, Symmetry>& H,
                                                                       bool CALC_GRAD)
{
    Jack.set_A(Psi);
    Jack.info();
    if(REINIT_ENV) {
        fmt::print("\t{: >3} Reinit env.\n", "•");
        Jack = CTM<Scalar, Symmetry, false>(Psi, opts.chi);
        Jack.init();
    }
    double E = std::numeric_limits<Scalar>::quiet_NaN();
    double Eprev = std::numeric_limits<Scalar>::quiet_NaN();
    fmt::print("\t{: >3} steps without gradient tracking.\n", "•");
    for(std::size_t step = 0; step < opts.max_presteps; ++step) {
        Jack.left_move();
        Jack.right_move();
        Jack.top_move();
        Jack.bottom_move();
        Jack.computeRDM();
        auto [E_h, E_v] = avg(Jack, H);
        double E = (E_h.sum() + E_v.sum()) / Jack.cell().uniqueSize();
        step == 0 ? fmt::print("\t{: >3} {:2d}: E={:2.8f}\n", "•", step, E)
                  : fmt::print("\t{: >3} {:2d}: E={:2.8f}, conv={:2.10g}\n", "•", step, E, std::abs(E - Eprev));
        if(std::abs(E - Eprev) < opts.tol_E) { break; }
        Eprev = E;
    }

    if(not CALC_GRAD) {
        REINIT_ENV = true;
        return E;
    }

    stan::math::nested_rev_autodiff nested;
    Xped::CTM<double, Symmetry, true> Jim(Jack);
    fmt::print("\t{: >3} forward pass:\n", "•");
    Jim.template solve<false>(opts.track_steps);
    auto [E_h, E_v] = avg(Jim, H);
    auto res = (E_h.sum() + E_v.sum()) / Jim.cell().uniqueSize();
    E = res.val();
    fmt::print("\t{: >3} backward pass:\n", "•");
    stan::math::grad(res.vi_);
    std::size_t count = 0;
    for(auto it = Jim.Psi()->gradbegin(); it != Jim.Psi()->gradend(); ++it) { gradient[count++] = *it; }
    grad_norm = *std::max_element(gradient, gradient + Psi->plainSize(), [](Scalar a, Scalar b) { return std::abs(a) < std::abs(b); });
    REINIT_ENV = grad_norm < opts.reinit_env_tol ? false : true;
    return E;
}

} // namespace Xped