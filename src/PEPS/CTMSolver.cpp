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
    if(CALC_GRAD) {
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
        Jack = CTM<Scalar, Symmetry, TRank, false>(Psi, opts.chi, opts.init);
        Jack.init();
    }

    double E = std::numeric_limits<Scalar>::quiet_NaN();
    double Eprev = std::numeric_limits<Scalar>::quiet_NaN();
    double Eprevprev = std::numeric_limits<Scalar>::quiet_NaN();
    util::Stopwatch<> pre_t;
    std::size_t used_steps = 0ul;
    for(std::size_t step = 0; step < opts.max_presteps; ++step) {
        util::Stopwatch<> move_t;
        Jack.grow_all();
        Jack.computeRDM();
        auto [E_h, E_v, E_d1, E_d2] = avg(Jack, H);
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
    if(used_steps == 0) { used_steps = opts.max_presteps; }
    auto pre_time = pre_t.time_string();
    Log::per_iteration(opts.verbosity, "  {: >3} pre steps: {}", "•", pre_time);
    if(not CALC_GRAD) {
        REINIT_ENV = true;
        Log::on_exit(
            opts.verbosity, "  CTMSolver(χ={}({}), runtime={}[{} steps]): E={:.8f}", opts.chi, Jack.fullChi(), total_t.time_string(), used_steps, E);
        return E;
    }

    stan::math::nested_rev_autodiff nested;
    Xped::CTM<double, Symmetry, TRank, true, CPOpts> Jim(Jack);
    util::Stopwatch<> forward_t;
    Jim.solve(opts.track_steps);
    auto forward_time = forward_t.time_string();
    Log::per_iteration(opts.verbosity, "  {: >3} forward pass: {}", "•", forward_time);
    auto [E_h, E_v, E_d1, E_d2] = avg(Jim, H);
    auto res = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jim.cell().uniqueSize();
    E = res.val();
    util::Stopwatch<> backward_t;
    stan::math::grad(res.vi_);
    auto backward_time = backward_t.time_string();
    Log::per_iteration(opts.verbosity, "  {: >3} backward pass: {}", "•", backward_time);
    std::size_t count = 0;
    for(auto it = Jim.Psi()->gradbegin(); it != Jim.Psi()->gradend(); ++it) { gradient[count++] = *it; }
    grad_norm = std::abs(*std::max_element(gradient, gradient + Psi->plainSize(), [](Scalar a, Scalar b) { return std::abs(a) < std::abs(b); }));
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

} // namespace Xped
