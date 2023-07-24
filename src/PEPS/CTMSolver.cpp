#include <limits>

#include "Xped/PEPS/CTMSolver.hpp"
#include "Xped/Util/Logging.hpp"
#include "Xped/Util/Stopwatch.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, typename HamScalar, bool ALL_OUT_LEGS, Opts::CTMCheckpoint CPOpts, std::size_t TRank>
template <bool AD>
typename ScalarTraits<Scalar>::Real
CTMSolver<Scalar, Symmetry, HamScalar, ALL_OUT_LEGS, CPOpts, TRank>::solve(std::shared_ptr<iPEPS<Scalar, Symmetry, ALL_OUT_LEGS>> Psi,
                                                                           Scalar* gradient,
                                                                           Hamiltonian<HamScalar, Symmetry>& H)
{
    util::Stopwatch<> total_t;
    Jack.set_A(Psi);
    if constexpr(AD) {
        Log::on_entry(opts.verbosity,
                      "  CTMSolver(χ={}, {}): UnitCell=({}x{}), init={}, reinit env={}, max steps(untracked)={}, steps(tracked)={}",
                      opts.chi,
                      Symmetry::name(),
                      Jack.cell().Lx,
                      Jack.cell().Ly,
                      fmt::streamed(Jack.init_mode()),
                      REINIT_ENV,
                      opts.max_presteps,
                      opts.track_steps);
    } else {
        Log::on_entry(opts.verbosity,
                      "  CTMSolver(χ={}, {}): UnitCell=({}x{}), init={}, reinit env={}, max steps={}",
                      opts.chi,
                      Symmetry::name(),
                      Jack.cell().Lx,
                      Jack.cell().Ly,
                      fmt::streamed(Jack.init_mode()),
                      REINIT_ENV,
                      opts.max_presteps);
    }
    Log::on_entry(opts.verbosity, "   Environment for {}", Psi->info());
    if(REINIT_ENV) {
        Jack = CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, false>(Psi, opts.chi, opts.init);
        Jack.init();
    }

    double E = std::numeric_limits<typename ScalarTraits<Scalar>::Real>::quiet_NaN();
    double Eprev = 1000.;
    double Eprevprev = 1000.;
    util::Stopwatch<> pre_t;
    std::size_t used_steps = 0ul;
    for(std::size_t step = 0; step < opts.max_presteps; ++step) {
        util::Stopwatch<> move_t;
        Jack.grow_all();
        Jack.computeRDM();
        auto Hobs = H.asObservable();
        auto [E_h, E_v, E_d1, E_d2] = avg(Jack, Hobs);
        E = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jack.cell().uniqueSize();
        step == 0 ? Log::per_iteration(opts.verbosity, "  {: >3} {:2d}: E={:2.8f}, t={}", "▷", step, E, move_t.time_string())
                  : Log::per_iteration(
                        opts.verbosity, "  {: >3} {:2d}: E={:2.8f}, conv={:2.10g}, t={}", "▷", step, E, std::abs(E - Eprev), move_t.time_string());
        if(std::abs(E - Eprev) < opts.tol_E) {
            used_steps = step;
            break;
        }
        if(std::abs(E - Eprevprev) < opts.tol_E) {
            Log::per_iteration(opts.verbosity, "  {: >3} Oscillation -> break", "•");
            used_steps = step;
            break;
        }
        Eprevprev = Eprev;
        Eprev = E;
    }
    Jack.checkHermiticity();
    if(used_steps == 0) { used_steps = opts.max_presteps; }
    auto pre_time = pre_t.time_string();
    Log::per_iteration(opts.verbosity, "  {: >3} pre steps: {}", "•", pre_time);
    if constexpr(not AD) {
        REINIT_ENV = true;
        Log::on_exit(
            opts.verbosity, "  CTMSolver(χ={}({}), runtime={}[{} steps]): E={:.8f}", opts.chi, Jack.fullChi(), total_t.time_string(), used_steps, E);
        return E;
    } else {
        stan::math::nested_rev_autodiff nested;
        Xped::CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, true, CPOpts> Jim(Jack);
        util::Stopwatch<> forward_t;
        Jim.solve(opts.track_steps);
        auto forward_time = forward_t.time_string();
        Log::per_iteration(opts.verbosity, "  {: >3} forward pass: {}", "•", forward_time);
        auto Hobs = H.asObservable();
        auto [E_h, E_v, E_d1, E_d2] = avg(Jim, Hobs);
        auto res = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jim.cell().uniqueSize();
        E = res.val();
        util::Stopwatch<> backward_t;
        stan::math::grad(res.vi_);
        auto backward_time = backward_t.time_string();
        Log::per_iteration(opts.verbosity, "  {: >3} backward pass: {}", "•", backward_time);
        auto grad = Jim.Psi()->graddata();
        for(std::size_t i = 0; i < grad.size(); ++i) { gradient[i] = grad[i]; }
        // std::size_t count = 0;
        // for(auto it = Jim.Psi()->gradbegin(); it != Jim.Psi()->gradend(); ++it) { gradient[count++] = *it; }
        // grad_norm = std::abs(*std::max_element(gradient, gradient + Psi->plainSize(), [](Scalar a, Scalar b) { return std::abs(a) < std::abs(b);
        // }));
        if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
            grad_norm = std::sqrt(
                std::inner_product(gradient,
                                   gradient + Psi->plainSize(),
                                   gradient,
                                   0.,
                                   std::plus<>(),
                                   [](const std::complex<double>& c1, const std::complex<double>& c2) { return std::real(c1 * std::conj(c2)); }));
        } else {
            grad_norm = std::sqrt(std::inner_product(gradient, gradient + Psi->plainSize(), gradient, 0.));
        }
        REINIT_ENV = grad_norm < opts.reinit_env_tol ? false : true;
        Log::on_exit(opts.verbosity,
                     "  CTMSolver(χ={}({}), runtime={} [{} steps, pre={}, forward={}, backward={}]): E={:.8f}, |∇|={:.1e}",
                     opts.chi,
                     Jack.fullChi(),
                     total_t.time_string(),
                     used_steps,
                     pre_time,
                     forward_time,
                     backward_time,
                     E,
                     grad_norm);
        return E;
    }
}

} // namespace Xped
