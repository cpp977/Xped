#ifndef XPED_SELFNLO_CORE_H_
#define XPED_SELFNLO_CORE_H_

// =============================================================================
// Xped self-contained nonlinear optimizer.
//
// Clean-room, header-only replacement for the subset of the ceres-solver
// GradientProblemSolver API that Xped uses (NLO/CeresSolve.hpp).  It keeps the
// `ceres` namespace and include paths (`ceres/gradient_problem_solver.h`, ...)
// so the rest of Xped compiles unchanged, but it has **zero external
// dependencies** (no ceres, no glog, no gflags, no SuiteSparse).
//
// The optimization algorithm is a port of TeneT.c's LBFGS engine
// (src/optimize/lbfgs.cpp), which itself is a documented, pinned port of
// OptimKit.jl v0.4.2:
//   * L-BFGS two-loop recursion with NORMALIZED curvature pairs
//     (s/|s|, y/|s|, rho = <s,s>/<s,y>) and initial Hessian scaling
//     gamma = <s,y>/<y,y> from the newest pair,
//   * Hager-Zhang line search (Algorithm 851: CG_DESCENT, Hager & Zhang,
//     ACM TOMS 32 (2006)) with approximate-Wolfe acceptance -- this serves
//     the WOLFE line-search option; ARMIJO uses simple backtracking,
//   * curvature-pair acceptance <s,y>/<s,s> > |g_new|/10000,
//   * soft degradation: a zero-step line search clears the history and
//     retries with steepest descent before giving up.
// NONLINEAR_CONJUGATE_GRADIENT (FR/PR+/HS) and STEEPEST_DESCENT directions
// share the same line searches.
//
// Semantics matched to ceres where Xped depends on them:
//   * Solve() always evaluates cost AND gradient together (Xped's Energy
//     functor requires a valid gradient pointer on every call),
//   * callbacks are invoked with a filled IterationSummary after the initial
//     evaluation (iteration 0) and after every accepted iteration,
//   * termination on max_num_iterations / function_tolerance /
//     gradient_tolerance (max-norm) / parameter_tolerance,
//   * Summary::iterations holds the per-iteration records and FullReport()
//     yields a human-readable summary.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ceres {

// ---- enums (value subset used by Xped plus common defaults) -----------------

enum LineSearchDirectionType { STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, LBFGS, BFGS };
enum LineSearchType { ARMIJO, WOLFE };
enum LineSearchInterpolationType { BISECTION, QUADRATIC, CUBIC };
enum NonlinearConjugateGradientType { FLETCHER_REEVES, POLAK_RIBIERE, HESTENES_STIEFEL };
enum LoggingType { SILENT, PER_MINIMIZER_ITERATION };
enum CallbackReturnType { SOLVER_ABORT, SOLVER_TERMINATE_SUCCESSFULLY, SOLVER_CONTINUE };
enum TerminationType { CONVERGENCE, NO_CONVERGENCE, FAILURE, USER_SUCCESS, USER_FAILURE };

// ---- problem definition -----------------------------------------------------

class FirstOrderFunction
{
public:
    virtual ~FirstOrderFunction() = default;
    // cost is a scalar output; gradient has NumParameters() entries and is
    // ALWAYS non-null in this implementation.
    virtual bool Evaluate(const double* parameters, double* cost, double* gradient) const = 0;
    virtual int NumParameters() const = 0;
};

class GradientProblem
{
public:
    explicit GradientProblem(FirstOrderFunction* function)
        : function_(function)
    {}

    int NumParameters() const { return function_->NumParameters(); }
    bool Evaluate(const double* parameters, double* cost, double* gradient) const { return function_->Evaluate(parameters, cost, gradient); }
    const FirstOrderFunction* function() const { return function_.get(); }

private:
    std::unique_ptr<FirstOrderFunction> function_;
};

// ---- iteration reporting ----------------------------------------------------

struct IterationSummary
{
    int iteration = 0;
    bool step_is_valid = true;
    bool step_is_successful = true;
    double cost = 0.0;
    double cost_change = 0.0;
    double gradient_norm = 0.0;     // 2-norm
    double gradient_max_norm = 0.0; // inf-norm
    double step_norm = 0.0;
    double step_size = 0.0; // line search alpha
    int line_search_function_evaluations = 0;
    double iteration_time_in_seconds = 0.0;
    double cumulative_time_in_seconds = 0.0;
};

class IterationCallback
{
public:
    virtual ~IterationCallback() = default;
    virtual CallbackReturnType operator()(const IterationSummary& summary) = 0;
};

// ---- solver -----------------------------------------------------------------

class GradientProblemSolver
{
public:
    struct Options
    {
        LineSearchDirectionType line_search_direction_type = LBFGS;
        LineSearchType line_search_type = WOLFE;
        NonlinearConjugateGradientType nonlinear_conjugate_gradient_type = FLETCHER_REEVES;
        int max_lbfgs_rank = 20;
        bool use_approximate_eigenvalue_bfgs_scaling = false;
        // interpolation used by the (default) bracket-and-zoom Wolfe line
        // search and by the Armijo backtracking: CUBIC (ceres default),
        // QUADRATIC, or BISECTION
        LineSearchInterpolationType line_search_interpolation_type = CUBIC;

        // extension beyond the ceres API: which engine backs the WOLFE line
        // search.  INTERPOLATING_WOLFE is the ceres-faithful default
        // (Nocedal-Wright bracket+zoom, strong Wolfe, safeguarded polynomial
        // interpolation).  HAGER_ZHANG is the OptimKit/TeneT.c line search
        // (approximate Wolfe; the production choice of TeneT.jl).
        enum LineSearchEngine { INTERPOLATING_WOLFE, HAGER_ZHANG };
        LineSearchEngine line_search_engine = INTERPOLATING_WOLFE;
        double min_line_search_step_size = 1e-9;
        double line_search_sufficient_function_decrease = 1e-4; // Armijo c1
        double max_line_search_step_contraction = 1e-3;
        double min_line_search_step_contraction = 0.6;
        int max_num_line_search_step_size_iterations = 20;
        int max_num_line_search_direction_restarts = 5;
        double line_search_sufficient_curvature_decrease = 0.9; // Wolfe c2
        double max_line_search_step_expansion = 10.0;
        int max_num_iterations = 50;
        double max_solver_time_in_seconds = 1e9;
        double function_tolerance = 1e-6;
        double gradient_tolerance = 1e-10;
        double parameter_tolerance = 1e-8;
        LoggingType logging_type = PER_MINIMIZER_ITERATION;
        bool minimizer_progress_to_stdout = false;
        bool update_state_every_iteration = false;
        std::vector<IterationCallback*> callbacks;

        // Hager-Zhang parameters (OptimKit v0.4.2 defaults via TeneT.c)
        double hz_c1 = 0.1;
        double hz_c2 = 0.9;
        double hz_epsilon = 1e-6;
        double hz_theta = 0.5;
        double hz_gamma = 2.0 / 3.0;
        double hz_rho = 5.0;
        int hz_maxiter = 10;
        int hz_maxfg = 20;
    };

    struct Summary
    {
        TerminationType termination_type = FAILURE;
        std::string message = "";
        double initial_cost = 0.0;
        double final_cost = 0.0;
        int num_cost_evaluations = 0;
        double total_time_in_seconds = 0.0;
        std::vector<IterationSummary> iterations;

        bool IsSolutionUsable() const { return termination_type == CONVERGENCE || termination_type == NO_CONVERGENCE || termination_type == USER_SUCCESS; }

        std::string BriefReport() const
        {
            char buf[256];
            std::snprintf(buf,
                          sizeof(buf),
                          "Xped self-contained solver. Iterations: %d, Initial cost: %.6e, Final cost: %.6e, Termination: %s",
                          static_cast<int>(iterations.size()),
                          initial_cost,
                          final_cost,
                          TerminationString());
            return std::string(buf);
        }

        std::string FullReport() const
        {
            std::string out = "\nSolver Summary (Xped self-contained L-BFGS engine, TeneT.c/OptimKit port)\n\n";
            char buf[512];
            std::snprintf(buf,
                          sizeof(buf),
                          "Iterations                     % 8d\nCost evaluations               % 8d\nInitial cost               % .6e\nFinal cost                 % .6e\nTotal time (s)             % .6e\nTermination:               %s\n%s\n",
                          static_cast<int>(iterations.empty() ? 0 : iterations.back().iteration),
                          num_cost_evaluations,
                          initial_cost,
                          final_cost,
                          total_time_in_seconds,
                          TerminationString(),
                          message.c_str());
            out += buf;
            return out;
        }

    private:
        const char* TerminationString() const
        {
            switch(termination_type) {
            case CONVERGENCE: return "CONVERGENCE";
            case NO_CONVERGENCE: return "NO_CONVERGENCE";
            case USER_SUCCESS: return "USER_SUCCESS";
            case USER_FAILURE: return "USER_FAILURE";
            default: return "FAILURE";
            }
        }
    };
};

// ---- implementation ---------------------------------------------------------

namespace internal {

using Vec = std::vector<double>;

inline double dot(const Vec& a, const Vec& b)
{
    double r = 0;
    for(std::size_t i = 0; i < a.size(); ++i) { r += a[i] * b[i]; }
    return r;
}
inline double norm2(const Vec& a) { return std::sqrt(dot(a, a)); }
inline double norm_inf(const Vec& a)
{
    double r = 0;
    for(double v : a) { r = std::max(r, std::fabs(v)); }
    return r;
}
inline void axpy(double a, const Vec& x, Vec& y)
{
    for(std::size_t i = 0; i < x.size(); ++i) { y[i] += a * x[i]; }
}
inline Vec scale(const Vec& x, double a)
{
    Vec r(x.size());
    for(std::size_t i = 0; i < x.size(); ++i) { r[i] = a * x[i]; }
    return r;
}
inline Vec add_scaled(const Vec& x, double a, const Vec& eta)
{
    Vec r(x.size());
    for(std::size_t i = 0; i < x.size(); ++i) { r[i] = x[i] + a * eta[i]; }
    return r;
}
inline Vec sub(const Vec& a, const Vec& b)
{
    Vec r(a.size());
    for(std::size_t i = 0; i < a.size(); ++i) { r[i] = a[i] - b[i]; }
    return r;
}

// evaluation counter wrapper around the GradientProblem
struct FgEval
{
    const GradientProblem& problem;
    int nfg = 0;

    // returns false if the user functor reported failure
    bool operator()(const Vec& x, double& f, Vec& g)
    {
        ++nfg;
        return problem.Evaluate(x.data(), &f, g.data());
    }
};

// -----------------------------------------------------------------------------
// Hager-Zhang line search: port of TeneT.c src/optimize/lbfgs.cpp (itself a
// pinned port of OptimKit v0.4.2 linesearches.jl).  Euclidean: the path is
// x0 + alpha*eta, path tangent == eta.
// -----------------------------------------------------------------------------

constexpr double kEps1 = 2.220446049250313e-16;
const double kEps34 = 1.8189894035458565e-12; // kEps1^(3/4)

struct LsPoint
{
    double alpha = 0.0;
    double phi = 0.0;
    double dphi = 0.0;
    Vec x, g;
};

struct LsOutcome
{
    LsPoint p;
    int nfg = 0;
    bool wolfe = false;
};

inline double secant(double a, double b, double fa, double fb) { return (a * fb - b * fa) / (fb - fa); }

class HagerZhang
{
public:
    HagerZhang(FgEval& fg, const GradientProblemSolver::Options& opt, LsPoint p0, const Vec& eta)
        : fg_(fg)
        , o_(opt)
        , p0_(std::move(p0))
        , eta_(eta)
        , fmax_(p0_.phi + opt.hz_epsilon)
    {}

    LsOutcome run(double alpha0)
    {
        if(p0_.dphi >= 0.0) { return {p0_, nfg_, false}; }
        LsPoint c = takestep(alpha0);
        if(wolfe_ok(c)) { return {c, nfg_, true}; } // acceptfirst = true

        auto [a, b] = bracket(std::move(c));
        if(a.alpha == b.alpha) { return {a, nfg_, true}; }
        if(b.alpha - a.alpha < kEps1) { return {a, nfg_, false}; }

        for(int k = 1;; ++k) {
            if(k >= o_.hz_maxiter || nfg_ >= o_.hz_maxfg) {
                return {a, nfg_, false}; // a-point: phi <= phi0+eps, dphi < 0
            }
            const double dalpha = b.alpha - a.alpha;

            const double ac = secant(a.alpha, b.alpha, a.dphi, b.dphi);
            auto [A, B] = update(a, b, ac);
            if(A.alpha == B.alpha) { return {A, nfg_, true}; }
            if(ac == B.alpha) {
                const double ac2 = secant(b.alpha, B.alpha, b.dphi, B.dphi);
                std::tie(a, b) = update(A, B, ac2);
            } else if(ac == A.alpha) {
                const double ac2 = secant(a.alpha, A.alpha, a.dphi, A.dphi);
                std::tie(a, b) = update(A, B, ac2);
            } else {
                a = std::move(A);
                b = std::move(B);
            }
            if(a.alpha == b.alpha) { return {a, nfg_, true}; }

            if(b.alpha - a.alpha > o_.hz_gamma * dalpha) { std::tie(a, b) = update(a, b, (a.alpha + b.alpha) / 2.0); }
            if(a.alpha == b.alpha) { return {a, nfg_, true}; }
            if(b.alpha - a.alpha < kEps1) { return {a, nfg_, false}; }
        }
    }

private:
    LsPoint takestep(double alpha)
    {
        LsPoint p;
        p.alpha = alpha;
        p.x = add_scaled(p0_.x, alpha, eta_);
        p.g.resize(p.x.size());
        fg_(p.x, p.phi, p.g);
        p.dphi = dot(p.g, eta_);
        ++nfg_;
        return p;
    }

    bool wolfe_ok(const LsPoint& c) const
    {
        const double c1 = o_.hz_c1, c2 = o_.hz_c2;
        const bool exact = (c.phi <= p0_.phi + c1 * c.alpha * p0_.dphi) && (c.dphi > c2 * p0_.dphi);
        const bool approx = (c.phi <= fmax_) && ((2.0 * c1 - 1.0) * p0_.dphi >= c.dphi) && (c.dphi >= c2 * p0_.dphi);
        return exact || approx;
    }

    std::pair<LsPoint, LsPoint> update(const LsPoint& a, const LsPoint& b, double ac)
    {
        if(!(a.alpha < ac && ac < b.alpha)) { return {a, b}; } // U0 (filters NaN)
        LsPoint c = takestep(ac);
        if(!std::isfinite(c.phi) || !std::isfinite(c.dphi)) {
            return bisect(a, std::move(c)); // soft degradation on non-finite
        }
        if(wolfe_ok(c)) { return {c, c}; }
        if(c.dphi >= 0.0) { return {a, c}; } // U1
        if(c.phi <= fmax_) { return {c, b}; } // U2
        return bisect(a, std::move(c)); // U3
    }

    std::pair<LsPoint, LsPoint> bisect(LsPoint a, LsPoint b)
    {
        int local_nfg = 0;
        while(true) {
            if(b.alpha - a.alpha <= kEps34 || local_nfg >= o_.hz_maxfg) { return {std::move(a), std::move(b)}; }
            const double ad = (1.0 - o_.hz_theta) * a.alpha + o_.hz_theta * b.alpha;
            LsPoint d = takestep(ad);
            ++local_nfg;
            if(wolfe_ok(d)) { return {d, d}; }
            if(d.dphi >= 0.0) { return {std::move(a), std::move(d)}; }
            if(d.phi <= fmax_) {
                a = std::move(d);
            } else {
                b = std::move(d);
            }
        }
    }

    std::pair<LsPoint, LsPoint> bracket(LsPoint c)
    {
        LsPoint a = p0_;
        double alpha = c.alpha;
        while(true) {
            while(!(std::isfinite(c.phi) && std::isfinite(c.dphi)) && nfg_ < o_.hz_maxfg) {
                alpha = (a.alpha + alpha) / 2.0;
                c = takestep(alpha);
            }
            if(!std::isfinite(c.phi) || !std::isfinite(c.dphi)) { return {a, a}; }
            if(c.dphi >= 0.0) { return {std::move(a), std::move(c)}; } // B1
            if(c.phi > fmax_) { return bisect(p0_, std::move(c)); } // B2
            if(nfg_ >= o_.hz_maxfg) { return {c, c}; } // bounded B3 expansion
            a = std::move(c);
            alpha *= o_.hz_rho;
            c = takestep(alpha);
        }
    }

    FgEval& fg_;
    const GradientProblemSolver::Options& o_;
    LsPoint p0_;
    const Vec& eta_;
    double fmax_;
    int nfg_ = 0;
};

// -----------------------------------------------------------------------------
// Polynomial interpolation helpers (Nocedal & Wright, Numerical Optimization,
// 2nd ed., eqs. 3.58/3.59).  Both return NaN when the formula degenerates so
// callers can fall back to bisection.
// -----------------------------------------------------------------------------

// minimizer of the cubic Hermite interpolant through (a, fa, dfa), (b, fb, dfb)
inline double cubic_minimizer(double a, double fa, double dfa, double b, double fb, double dfb)
{
    const double d1 = dfa + dfb - 3.0 * (fa - fb) / (a - b);
    const double disc = d1 * d1 - dfa * dfb;
    if(!(disc >= 0.0)) { return std::numeric_limits<double>::quiet_NaN(); }
    const double d2 = std::copysign(std::sqrt(disc), b - a);
    const double denom = dfb - dfa + 2.0 * d2;
    if(denom == 0.0) { return std::numeric_limits<double>::quiet_NaN(); }
    return b - (b - a) * (dfb + d2 - d1) / denom;
}

// minimizer of the quadratic interpolant through (a, fa, dfa) and (b, fb)
inline double quadratic_minimizer(double a, double fa, double dfa, double b, double fb)
{
    const double db = b - a;
    const double denom = 2.0 * (fb - fa - dfa * db);
    if(denom == 0.0) { return std::numeric_limits<double>::quiet_NaN(); }
    return a - dfa * db * db / denom;
}

// shared trial-point evaluation on the line x0 + alpha*eta
inline LsPoint ls_takestep(FgEval& fg, const LsPoint& p0, const Vec& eta, double alpha)
{
    LsPoint p;
    p.alpha = alpha;
    p.x = add_scaled(p0.x, alpha, eta);
    p.g.resize(p.x.size());
    fg(p.x, p.phi, p.g);
    p.dphi = dot(p.g, eta);
    return p;
}

// -----------------------------------------------------------------------------
// Strong-Wolfe line search with safeguarded polynomial interpolation: the
// classic bracket-and-zoom scheme (Nocedal & Wright, Algorithms 3.5/3.6),
// which is also the construction behind ceres' WOLFE line search.  The zoom
// trial point comes from cubic Hermite interpolation by default
// (line_search_interpolation_type == CUBIC, the ceres default), from
// one-sided quadratic interpolation (QUADRATIC), or from plain bisection
// (BISECTION); an interpolated point that is non-finite or too close to the
// bracket boundary falls back to bisection.
// -----------------------------------------------------------------------------

class WolfeInterpolating
{
public:
    WolfeInterpolating(FgEval& fg, const GradientProblemSolver::Options& opt, LsPoint p0, const Vec& eta)
        : fg_(fg)
        , o_(opt)
        , p0_(std::move(p0))
        , eta_(eta)
        , c1_(opt.line_search_sufficient_function_decrease)
        , c2_(opt.line_search_sufficient_curvature_decrease)
    {}

    LsOutcome run(double alpha0)
    {
        if(p0_.dphi >= 0.0) { return {p0_, nfg_, false}; } // not a descent direction

        // ---- bracketing phase (N&W Algorithm 3.5) ----
        LsPoint prev = p0_;
        double alpha = alpha0;
        for(int i = 0; i < o_.max_num_line_search_step_size_iterations; ++i) {
            LsPoint cur = takestep(alpha);
            if(!std::isfinite(cur.phi) || !std::isfinite(cur.dphi)) {
                // shrink toward the last good point until the trial is finite
                alpha = 0.5 * (prev.alpha + alpha);
                if(alpha <= o_.min_line_search_step_size) { break; }
                continue;
            }
            if(cur.phi > p0_.phi + c1_ * cur.alpha * p0_.dphi || (prev.alpha > 0.0 && cur.phi >= prev.phi)) { return zoom(std::move(prev), std::move(cur)); }
            if(std::fabs(cur.dphi) <= -c2_ * p0_.dphi) { return {std::move(cur), nfg_, true}; } // strong Wolfe holds
            if(cur.dphi >= 0.0) { return zoom(std::move(cur), std::move(prev)); }
            // still descending with sufficient decrease: expand.  The trial
            // comes from cubic extrapolation over the last two points,
            // safeguarded into (1.5, max_line_search_step_expansion] * alpha.
            double guess = cubic_minimizer(prev.alpha, prev.phi, prev.dphi, cur.alpha, cur.phi, cur.dphi);
            const double lo = 1.5 * alpha, hi = o_.max_line_search_step_expansion * alpha;
            if(!(guess > lo && guess < hi)) { guess = hi; }
            prev = std::move(cur);
            alpha = guess;
        }
        // budget exhausted while still descending: prev is the best point with
        // sufficient decrease seen so far (alpha = 0 if we never moved)
        return {std::move(prev), nfg_, false};
    }

private:
    LsPoint takestep(double alpha)
    {
        LsPoint p = ls_takestep(fg_, p0_, eta_, alpha);
        ++nfg_;
        return p;
    }

    double interpolate(const LsPoint& lo, const LsPoint& hi) const
    {
        switch(o_.line_search_interpolation_type) {
        case CUBIC: return cubic_minimizer(lo.alpha, lo.phi, lo.dphi, hi.alpha, hi.phi, hi.dphi);
        case QUADRATIC: return quadratic_minimizer(lo.alpha, lo.phi, lo.dphi, hi.alpha, hi.phi);
        case BISECTION:
        default: return 0.5 * (lo.alpha + hi.alpha);
        }
    }

    // N&W Algorithm 3.6.  Invariants: lo satisfies sufficient decrease, the
    // interval between lo and hi brackets a strong-Wolfe point, and
    // dphi_lo * (hi.alpha - lo.alpha) < 0.  lo/hi are NOT ordered by alpha.
    LsOutcome zoom(LsPoint lo, LsPoint hi)
    {
        for(int j = 0; j < o_.max_num_line_search_step_size_iterations; ++j) {
            const double a_min = std::min(lo.alpha, hi.alpha);
            const double a_max = std::max(lo.alpha, hi.alpha);
            const double range = a_max - a_min;
            if(range <= o_.min_line_search_step_size) { break; }

            double aj = interpolate(lo, hi);
            // safeguard: require an interior point away from the boundary
            if(!std::isfinite(aj) || aj <= a_min + 0.01 * range || aj >= a_max - 0.01 * range) { aj = a_min + 0.5 * range; }

            LsPoint cur = takestep(aj);
            if(!std::isfinite(cur.phi) || !std::isfinite(cur.dphi) || cur.phi > p0_.phi + c1_ * aj * p0_.dphi || cur.phi >= lo.phi) {
                hi = std::move(cur); // too far (a non-finite trial counts as too far)
            } else {
                if(std::fabs(cur.dphi) <= -c2_ * p0_.dphi) { return {std::move(cur), nfg_, true}; }
                if(cur.dphi * (hi.alpha - lo.alpha) >= 0.0) { hi = std::move(lo); }
                lo = std::move(cur);
            }
        }
        // no certified strong-Wolfe point: return lo, which satisfies
        // sufficient decrease whenever it moved (alpha = 0 otherwise)
        return {std::move(lo), nfg_, false};
    }

    FgEval& fg_;
    const GradientProblemSolver::Options& o_;
    LsPoint p0_;
    const Vec& eta_;
    double c1_, c2_;
    int nfg_ = 0;
};

// backtracking Armijo with interpolated, safeguarded contraction (the ceres
// ARMIJO construction: the next trial is the polynomial-interpolation
// minimizer clamped into [max_step_contraction, min_step_contraction]*alpha)
inline LsOutcome armijo_search(FgEval& fg, const GradientProblemSolver::Options& o, const LsPoint& p0, const Vec& eta, double alpha0)
{
    LsOutcome out;
    if(p0.dphi >= 0.0) {
        out.p = p0;
        return out;
    }
    double alpha = alpha0;
    const double c1 = o.line_search_sufficient_function_decrease;
    for(int it = 0; it < o.max_num_line_search_step_size_iterations; ++it) {
        LsPoint p = ls_takestep(fg, p0, eta, alpha);
        ++out.nfg;
        if(std::isfinite(p.phi) && p.phi <= p0.phi + c1 * alpha * p0.dphi) {
            out.p = std::move(p);
            out.wolfe = true;
            return out;
        }
        double next = std::numeric_limits<double>::quiet_NaN();
        if(std::isfinite(p.phi)) {
            switch(o.line_search_interpolation_type) {
            case CUBIC:
                if(std::isfinite(p.dphi)) { next = cubic_minimizer(0.0, p0.phi, p0.dphi, alpha, p.phi, p.dphi); }
                if(!std::isfinite(next)) { next = quadratic_minimizer(0.0, p0.phi, p0.dphi, alpha, p.phi); }
                break;
            case QUADRATIC: next = quadratic_minimizer(0.0, p0.phi, p0.dphi, alpha, p.phi); break;
            case BISECTION:
            default: next = 0.5 * alpha; break;
            }
        }
        if(!std::isfinite(next)) { next = 0.5 * alpha; }
        next = std::min(o.min_line_search_step_contraction * alpha, std::max(o.max_line_search_step_contraction * alpha, next));
        alpha = next;
        if(alpha < o.min_line_search_step_size) { break; }
    }
    out.p = p0; // zero step
    return out;
}

// -----------------------------------------------------------------------------
// L-BFGS two-loop machinery (TeneT.c / OptimKit v0.4.2 lbfgs.jl).
// -----------------------------------------------------------------------------

struct CurvaturePair
{
    Vec s, y; // stored NORMALIZED: s/|s|, y/|s|
    double rho = 0.0; // <s,s>/<s,y> of the unnormalized pair
};

class History
{
public:
    explicit History(int m)
        : buf_(static_cast<std::size_t>(m))
        , cap_(m)
    {}
    int size() const { return len_; }
    void clear()
    {
        first_ = 0;
        len_ = 0;
    }
    const CurvaturePair& at(int i) const { return buf_[static_cast<std::size_t>((first_ + i) % cap_)]; }
    void push(CurvaturePair p)
    {
        if(len_ < cap_) {
            buf_[static_cast<std::size_t>((first_ + len_) % cap_)] = std::move(p);
            ++len_;
        } else {
            buf_[static_cast<std::size_t>(first_)] = std::move(p);
            first_ = (first_ + 1) % cap_;
        }
    }

private:
    std::vector<CurvaturePair> buf_;
    int cap_ = 0, first_ = 0, len_ = 0;
};

// two-loop recursion; gamma scaling follows use_approximate_eigenvalue_bfgs_scaling
inline Vec two_loop(const History& H, const Vec& g, std::vector<double>& alpha_ws, bool eigenvalue_scaling)
{
    const int L = H.size();
    Vec q = g;
    for(int k = L - 1; k >= 0; --k) {
        const CurvaturePair& p = H.at(k);
        alpha_ws[static_cast<std::size_t>(k)] = p.rho * dot(p.s, q);
        axpy(-alpha_ws[static_cast<std::size_t>(k)], p.y, q);
    }
    const CurvaturePair& newest = H.at(L - 1);
    // OptimKit/TeneT always scale with gamma = <s,y>/<y,y> of the newest pair
    // (the Oren-Luenberger approximate-eigenvalue scaling); without the flag
    // we still apply it -- it is essential for well-scaled tensor problems --
    // but from the FIRST stored pair only, mirroring ceres' "initial scaling
    // only" behavior as closely as the normalized storage allows.
    double gamma;
    if(eigenvalue_scaling) {
        gamma = dot(newest.s, newest.y) / dot(newest.y, newest.y);
    } else {
        const CurvaturePair& oldest = H.at(0);
        gamma = dot(oldest.s, oldest.y) / dot(oldest.y, oldest.y);
    }
    Vec z = scale(q, gamma);
    for(int k = 0; k < L; ++k) {
        const CurvaturePair& p = H.at(k);
        const double beta = p.rho * dot(p.y, z);
        axpy(alpha_ws[static_cast<std::size_t>(k)] - beta, p.s, z);
    }
    return z;
}

} // namespace internal

// ---- Solve ------------------------------------------------------------------

inline void
Solve(const GradientProblemSolver::Options& options, const GradientProblem& problem, double* parameters_ptr, GradientProblemSolver::Summary* summary)
{
    using namespace internal;
    const auto t_start = std::chrono::steady_clock::now();
    auto seconds_since_start = [&t_start]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };

    const int n = problem.NumParameters();
    Vec x(parameters_ptr, parameters_ptr + n);
    Vec g(static_cast<std::size_t>(n), 0.0);
    double f = 0.0;

    FgEval fg{problem};
    *summary = GradientProblemSolver::Summary{};

    auto write_back = [&]() { std::copy(x.begin(), x.end(), parameters_ptr); };

    if(!fg(x, f, g)) {
        summary->termination_type = FAILURE;
        summary->message = "Initial cost evaluation failed.";
        return;
    }
    summary->initial_cost = f;
    double gradnorm = norm2(g);

    History H(std::max(1, options.max_lbfgs_rank));
    std::vector<double> alpha_ws(static_cast<std::size_t>(std::max(1, options.max_lbfgs_rank)), 0.0);
    Vec prev_g, prev_eta; // for nonlinear CG
    double prev_alpha = 0.0;
    double prev_dphi0 = 0.0;
    int ncg_since_restart = 0;

    // NCG needs a much more exact line search than (L-)BFGS: tighten the
    // curvature constant to the CG-recommended value (for both engines).
    GradientProblemSolver::Options ls_options = options;
    if(options.line_search_direction_type == NONLINEAR_CONJUGATE_GRADIENT) {
        ls_options.hz_c1 = 1e-4;
        ls_options.hz_c2 = 0.1;
        ls_options.line_search_sufficient_function_decrease = 1e-4;
        ls_options.line_search_sufficient_curvature_decrease = 0.1;
    }

    int iteration = 0;
    auto record_iteration = [&](double cost_change, double step_norm, double step_size, int ls_evals, double iter_time) {
        IterationSummary is;
        is.iteration = iteration;
        is.cost = f;
        is.cost_change = cost_change;
        is.gradient_norm = gradnorm;
        is.gradient_max_norm = norm_inf(g);
        is.step_norm = step_norm;
        is.step_size = step_size;
        is.line_search_function_evaluations = ls_evals;
        is.iteration_time_in_seconds = iter_time;
        is.cumulative_time_in_seconds = seconds_since_start();
        summary->iterations.push_back(is);
        if(options.logging_type != SILENT && options.minimizer_progress_to_stdout) {
            std::printf("iter %4d: f = %.12e, |g| = %.4e, step = %.2e\n", iteration, f, gradnorm, step_size);
        }
        return is;
    };
    auto run_callbacks = [&](const IterationSummary& is) -> CallbackReturnType {
        for(IterationCallback* cb : options.callbacks) {
            const CallbackReturnType ret = (*cb)(is);
            if(ret != SOLVER_CONTINUE) { return ret; }
        }
        return SOLVER_CONTINUE;
    };
    auto finish = [&](TerminationType t, std::string msg) {
        summary->termination_type = t;
        summary->message = std::move(msg);
        summary->final_cost = f;
        summary->num_cost_evaluations = fg.nfg;
        summary->total_time_in_seconds = seconds_since_start();
        write_back();
    };

    // iteration 0 report (initial state)
    {
        const IterationSummary is0 = record_iteration(0.0, 0.0, 0.0, 1, seconds_since_start());
        const CallbackReturnType ret = run_callbacks(is0);
        if(ret == SOLVER_ABORT) {
            finish(USER_FAILURE, "Callback requested abort at iteration 0.");
            return;
        }
        if(ret == SOLVER_TERMINATE_SUCCESSFULLY) {
            finish(USER_SUCCESS, "Callback requested successful termination at iteration 0.");
            return;
        }
    }
    if(norm_inf(g) <= options.gradient_tolerance) {
        finish(CONVERGENCE, "Gradient tolerance reached at the initial point.");
        return;
    }

    bool steepest_retry_used = false;
    while(iteration < options.max_num_iterations) {
        const auto t_iter = std::chrono::steady_clock::now();

        // ---- search direction ----
        Vec eta;
        bool used_lbfgs_direction = false;
        switch(options.line_search_direction_type) {
        case LBFGS:
        case BFGS: {
            if(H.size() > 0) {
                eta = scale(two_loop(H, g, alpha_ws, options.use_approximate_eigenvalue_bfgs_scaling), -1.0);
                used_lbfgs_direction = true;
            } else {
                eta = scale(g, -0.01 / gradnorm); // OptimKit first-iteration guess
            }
            break;
        }
        case NONLINEAR_CONJUGATE_GRADIENT: {
            // Powell restarts: on the first iteration, periodically (every n
            // iterations), and when consecutive gradients lose conjugacy
            // (|<g, g_prev>| >= 0.2 <g,g>) -- standard practice for NCG with
            // inexact line searches.
            const bool restart = prev_eta.empty() || ncg_since_restart >= n ||
                                 (!prev_g.empty() && std::fabs(dot(g, prev_g)) >= 0.2 * dot(g, g));
            if(restart) {
                eta = scale(g, -1.0);
                ncg_since_restart = 0;
            } else {
                double beta = 0.0;
                const double gg_prev = dot(prev_g, prev_g);
                switch(options.nonlinear_conjugate_gradient_type) {
                case FLETCHER_REEVES: beta = dot(g, g) / gg_prev; break;
                case POLAK_RIBIERE: beta = std::max(0.0, (dot(g, g) - dot(g, prev_g)) / gg_prev); break;
                case HESTENES_STIEFEL: {
                    Vec ymg = sub(g, prev_g);
                    const double denom = dot(prev_eta, ymg);
                    beta = (denom != 0.0) ? dot(g, ymg) / denom : 0.0;
                    break;
                }
                }
                eta = scale(prev_eta, beta);
                axpy(-1.0, g, eta);
                if(dot(g, eta) >= 0.0) { // restart on non-descent
                    eta = scale(g, -1.0);
                    ncg_since_restart = 0;
                }
            }
            ++ncg_since_restart;
            break;
        }
        case STEEPEST_DESCENT:
        default: {
            eta = (iteration == 0) ? scale(g, -0.01 / gradnorm) : scale(g, -1.0);
            break;
        }
        }

        // ---- line search ----
        LsPoint p0;
        p0.alpha = 0.0;
        p0.phi = f;
        p0.dphi = dot(g, eta);
        p0.x = x;
        p0.g = g;

        double alpha0 = 1.0;
        if(options.line_search_direction_type != LBFGS && options.line_search_direction_type != BFGS) {
            // classical CG/SD initial trial: keep the predicted decrease of
            // the previous accepted step, alpha0 = alpha_prev * dphi0_prev/dphi0
            if(prev_alpha > 0.0 && p0.dphi < 0.0 && std::isfinite(prev_dphi0 / p0.dphi)) {
                alpha0 = std::min(1e12, std::max(1e-12, prev_alpha * (prev_dphi0 / p0.dphi)));
            } else {
                alpha0 = 1.0 / (1.0 + gradnorm);
            }
        }
        prev_dphi0 = p0.dphi;

        LsOutcome out;
        if(options.line_search_type == ARMIJO) {
            out = armijo_search(fg, ls_options, p0, eta, alpha0);
        } else if(ls_options.line_search_engine == GradientProblemSolver::Options::HAGER_ZHANG) {
            HagerZhang ls(fg, ls_options, p0, eta);
            out = ls.run(alpha0);
        } else {
            WolfeInterpolating ls(fg, ls_options, p0, eta);
            out = ls.run(alpha0);
        }
        ++iteration;

        if(out.p.alpha == 0.0) {
            // zero step: TeneT.c cascade -- clear history / retry steepest once
            if(used_lbfgs_direction && H.size() > 0) {
                H.clear();
                --iteration;
                continue;
            }
            if(!steepest_retry_used && options.line_search_direction_type == NONLINEAR_CONJUGATE_GRADIENT && !prev_eta.empty()) {
                prev_eta.clear();
                steepest_retry_used = true;
                --iteration;
                continue;
            }
            finish((out.nfg == 0) ? FAILURE : NO_CONVERGENCE,
                   (out.nfg == 0) ? "Line search was given a non-descent direction; no progress possible."
                                  : "Line search could not find a step; returning best iterate.");
            return;
        }

        // ---- accept the step ----
        const double cost_change = f - out.p.phi;
        const Vec s = sub(out.p.x, x);
        const double step_norm = norm2(s);
        const double x_norm = norm2(x);
        const double gradnorm_new = norm2(out.p.g);

        // curvature pair update for L-BFGS
        if(options.line_search_direction_type == LBFGS || options.line_search_direction_type == BFGS) {
            const Vec y = sub(out.p.g, g);
            const double sy = dot(s, y);
            const double ss = dot(s, s);
            if(ss > 0.0 && sy / ss > gradnorm_new / 10000.0) {
                const double snorm = std::sqrt(ss);
                H.push({scale(s, 1.0 / snorm), scale(y, 1.0 / snorm), ss / sy});
            }
        }
        prev_g = g;
        prev_eta = eta;
        prev_alpha = out.p.alpha;

        x = out.p.x;
        f = out.p.phi;
        g = out.p.g;
        gradnorm = gradnorm_new;
        if(options.update_state_every_iteration) { write_back(); }

        const double iter_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_iter).count();
        const IterationSummary is = record_iteration(cost_change, step_norm, out.p.alpha, out.nfg, iter_time);
        const CallbackReturnType ret = run_callbacks(is);
        if(ret == SOLVER_ABORT) {
            finish(USER_FAILURE, "Callback requested abort.");
            return;
        }
        if(ret == SOLVER_TERMINATE_SUCCESSFULLY) {
            finish(USER_SUCCESS, "Callback requested successful termination.");
            return;
        }

        // ---- convergence tests (ceres semantics) ----
        if(norm_inf(g) <= options.gradient_tolerance) {
            finish(CONVERGENCE, "Gradient tolerance reached.");
            return;
        }
        if(std::fabs(cost_change) <= options.function_tolerance * std::fabs(f)) {
            finish(CONVERGENCE, "Function tolerance reached.");
            return;
        }
        if(step_norm <= options.parameter_tolerance * (x_norm + options.parameter_tolerance)) {
            finish(CONVERGENCE, "Parameter tolerance reached.");
            return;
        }
        if(seconds_since_start() > options.max_solver_time_in_seconds) {
            finish(NO_CONVERGENCE, "Maximum solver time reached.");
            return;
        }
    }

    finish(NO_CONVERGENCE, "Maximum number of iterations reached.");
}

} // namespace ceres

#endif // XPED_SELFNLO_CORE_H_
