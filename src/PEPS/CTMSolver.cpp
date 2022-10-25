#include <limits>

#include "Xped/PEPS/CTMSolver.hpp"
#include "Xped/Util/Logging.hpp"
#include "Xped/Util/Stopwatch.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, Opts::CTMCheckpoint CPOpts, std::size_t TRank>
template <typename HamScalar>
typename ScalarTraits<Scalar>::Real CTMSolver<Scalar, Symmetry, CPOpts, TRank>::solve(std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi,
                                                                                      Scalar* gradient,
                                                                                      Hamiltonian<Symmetry>& H,
                                                                                      bool CALC_GRAD)
{
    util::Stopwatch<> total_t;
    Jack.set_A(Psi);
    // Jack.info();
    Log::on_entry(opts.verbosity,
                  "  CTMSolver(χ={}, {}): UnitCell=({}x{}), init={}, reinit env={}, max steps(untracked)={}, steps(tracked)={}",
                  Jack.chi(),
                  Symmetry::name(),
                  Jack.cell().Lx,
                  Jack.cell().Ly,
                  fmt::streamed(Jack.init_mode()),
                  REINIT_ENV,
                  opts.max_presteps,
                  opts.track_steps);
    if(REINIT_ENV) {
        Jack = CTM<Scalar, Symmetry, TRank, false>(Psi, opts.chi);
        Jack.init();
    }

    double E = std::numeric_limits<Scalar>::quiet_NaN();
    double Eprev = std::numeric_limits<Scalar>::quiet_NaN();
    double Eprevprev = std::numeric_limits<Scalar>::quiet_NaN();
    util::Stopwatch<> pre_t;
    for(std::size_t step = 0; step < opts.max_presteps; ++step) {
        util::Stopwatch<> move_t;
        Jack.grow_all();
        Jack.computeRDM();
        auto [E_h, E_v, E_d1, E_d2] = avg(Jack, H);
        double E = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jack.cell().uniqueSize();
        step == 0
            ? Log::per_iteration(opts.verbosity, "  {: >3} {:2d}: E={:2.8f}, t={}", "▷", step, E, move_t.time())
            : Log::per_iteration(opts.verbosity, "  {: >3} {:2d}: E={:2.8f}, conv={:2.10g}, t={}", "▷", step, E, std::abs(E - Eprev), move_t.time());
        if(std::abs(E - Eprev) < opts.tol_E) { break; }
        if(std::abs(E - Eprevprev) < opts.tol_E) {
            Log::per_iteration(opts.verbosity, "  {: >3} Oscillation -> break", "•");
            break;
        }
        Eprevprev = Eprev;
        Eprev = E;
    }
    auto pre_time = pre_t.time();
    Log::per_iteration(opts.verbosity, "  {: >3} pre steps: {}", "•", pre_time);
    if(not CALC_GRAD) {
        REINIT_ENV = true;
        Log::on_exit(opts.verbosity, "  CTMSolver(runtime={}): E={.8f}", total_t.time(), E);
        return E;
    }

    stan::math::nested_rev_autodiff nested;
    Xped::CTM<double, Symmetry, TRank, true, CPOpts> Jim(Jack);
    util::Stopwatch<> forward_t;
    Jim.solve(opts.track_steps);
    auto forward_time = forward_t.time();
    Log::per_iteration(opts.verbosity, "  {: >3} forward pass: {}", "•", forward_time);
    auto [E_h, E_v, E_d1, E_d2] = avg(Jim, H);
    auto res = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jim.cell().uniqueSize();
    E = res.val();
    util::Stopwatch<> backward_t;
    stan::math::grad(res.vi_);
    auto backward_time = backward_t.time();
    Log::per_iteration(opts.verbosity, "  {: >3} backward pass: {}", "•", backward_time);
    std::size_t count = 0;
    for(auto it = Jim.Psi()->gradbegin(); it != Jim.Psi()->gradend(); ++it) { gradient[count++] = *it; }
    grad_norm = std::abs(*std::max_element(gradient, gradient + Psi->plainSize(), [](Scalar a, Scalar b) { return std::abs(a) < std::abs(b); }));
    REINIT_ENV = grad_norm < opts.reinit_env_tol ? false : true;
    Log::on_exit(opts.verbosity,
                 "  CTMSolver(runtime={} [pre={}, forward={}, backward={}]): E={:.8f}, |∇|={:.1e}",
                 total_t.time(),
                 pre_time,
                 forward_time,
                 backward_time,
                 E,
                 grad_norm);
    return E;
}

} // namespace Xped
