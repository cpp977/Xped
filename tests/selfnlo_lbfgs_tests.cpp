// Standalone verification of the self-contained optimizer (include/selfnlo),
// exercised through the same ceres API shapes Xped's NLO/CeresSolve.hpp uses:
//   g++ -std=c++20 -I include/selfnlo tests/selfnlo_lbfgs_tests.cpp -o lbfgs_tests
//
// Covers: FirstOrderFunction/GradientProblem ownership, Options with
// callbacks, LBFGS + WOLFE (Hager-Zhang), NONLINEAR_CONJUGATE_GRADIENT,
// STEEPEST_DESCENT, ARMIJO, convergence tolerances, Summary/FullReport.

#include <cmath>
#include <cstdio>
#include <vector>

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

static int g_failures = 0;
#define CHECK(cond)                                                                                                                                  \
    do {                                                                                                                                             \
        if(!(cond)) {                                                                                                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                                              \
            ++g_failures;                                                                                                                            \
        }                                                                                                                                            \
    } while(0)

// N-dimensional Rosenbrock (chained), the classic L-BFGS acceptance test
class Rosenbrock final : public ceres::FirstOrderFunction
{
public:
    explicit Rosenbrock(int n)
        : n_(n)
    {}
    bool Evaluate(const double* x, double* cost, double* gradient) const override
    {
        double f = 0;
        for(int i = 0; i < n_; ++i) { gradient[i] = 0; }
        for(int i = 0; i + 1 < n_; ++i) {
            const double t1 = 1.0 - x[i];
            const double t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + 100.0 * t2 * t2;
            gradient[i] += -2.0 * t1 - 400.0 * x[i] * t2;
            gradient[i + 1] += 200.0 * t2;
        }
        cost[0] = f;
        return true;
    }
    int NumParameters() const override { return n_; }

private:
    int n_;
};

// ill-conditioned quadratic: f = 1/2 sum_i kappa_i x_i^2, kappa spanning 1..1e4
class Quadratic final : public ceres::FirstOrderFunction
{
public:
    explicit Quadratic(int n)
        : n_(n)
    {}
    bool Evaluate(const double* x, double* cost, double* gradient) const override
    {
        double f = 0;
        for(int i = 0; i < n_; ++i) {
            const double k = std::pow(10.0, 4.0 * i / (n_ - 1));
            f += 0.5 * k * x[i] * x[i];
            gradient[i] = k * x[i];
        }
        cost[0] = f;
        return true;
    }
    int NumParameters() const override { return n_; }

private:
    int n_;
};

struct CountingCallback final : public ceres::IterationCallback
{
    int calls = 0;
    int last_iteration = -1;
    double last_cost = 0;
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override
    {
        ++calls;
        last_iteration = summary.iteration;
        last_cost = summary.cost;
        return ceres::SOLVER_CONTINUE;
    }
};

static ceres::GradientProblemSolver::Summary run(ceres::FirstOrderFunction* f,
                                                 std::vector<double>& x,
                                                 ceres::LineSearchDirectionType dir,
                                                 ceres::LineSearchType ls,
                                                 int max_iter = 2000,
                                                 ceres::GradientProblemSolver::Options::LineSearchEngine engine =
                                                     ceres::GradientProblemSolver::Options::INTERPOLATING_WOLFE,
                                                 ceres::LineSearchInterpolationType interp = ceres::CUBIC)
{
    ceres::GradientProblem problem(f);
    ceres::GradientProblemSolver::Options options;
    options.line_search_direction_type = dir;
    options.line_search_type = ls;
    options.line_search_engine = engine;
    options.line_search_interpolation_type = interp;
    options.logging_type = ceres::SILENT;
    options.max_num_iterations = max_iter;
    options.function_tolerance = 0.0; // drive to gradient tolerance
    options.parameter_tolerance = 0.0;
    options.gradient_tolerance = 1e-10;
    options.use_approximate_eigenvalue_bfgs_scaling = true;
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, x.data(), &summary);
    return summary;
}

int main()
{
    using Engine = ceres::GradientProblemSolver::Options::LineSearchEngine;

    // ---- L-BFGS + Wolfe on Rosenbrock: both line-search engines, all
    //      interpolation types of the cubic-Wolfe engine ----
    struct LsCase
    {
        Engine engine;
        ceres::LineSearchInterpolationType interp;
        const char* name;
    };
    for(auto lc : {LsCase{Engine::INTERPOLATING_WOLFE, ceres::CUBIC, "Wolfe-cubic  "},
                   LsCase{Engine::INTERPOLATING_WOLFE, ceres::QUADRATIC, "Wolfe-quadr  "},
                   LsCase{Engine::INTERPOLATING_WOLFE, ceres::BISECTION, "Wolfe-bisect "},
                   LsCase{Engine::HAGER_ZHANG, ceres::CUBIC, "Hager-Zhang  "}}) {
        for(int n : {2, 24}) {
            std::vector<double> x(n, -1.2);
            for(int i = 1; i < n; i += 2) { x[i] = 1.0; }
            auto summary = run(new Rosenbrock(n), x, ceres::LBFGS, ceres::WOLFE, 2000, lc.engine, lc.interp);
            double err = 0;
            for(int i = 0; i < n; ++i) { err = std::max(err, std::fabs(x[i] - 1.0)); }
            std::printf("LBFGS/%s Rosenbrock N=%2d: f* = %.3e, max|x-1| = %.3e, iters = %d, evals = %d\n",
                        lc.name,
                        n,
                        summary.final_cost,
                        err,
                        static_cast<int>(summary.iterations.back().iteration),
                        summary.num_cost_evaluations);
            CHECK(summary.termination_type == ceres::CONVERGENCE);
            CHECK(summary.final_cost < 1e-15);
            CHECK(err < 1e-6);
        }
    }

    // ---- strong-Wolfe certificate on an accepted step (interpolating engine) ----
    {
        const int n = 16;
        std::vector<double> x0(n, 1.0), x(n, 1.0);
        Quadratic q(n);
        double f0 = 0, f1 = 0;
        std::vector<double> g0(n), g1(n);
        q.Evaluate(x0.data(), &f0, g0.data());

        ceres::GradientProblem problem(new Quadratic(n));
        ceres::GradientProblemSolver::Options options;
        options.logging_type = ceres::SILENT;
        options.max_num_iterations = 1;
        ceres::GradientProblemSolver::Summary summary;
        ceres::Solve(options, problem, x.data(), &summary);

        q.Evaluate(x.data(), &f1, g1.data());
        double s_dot_g0 = 0, s_dot_g1 = 0;
        for(int i = 0; i < n; ++i) {
            s_dot_g0 += (x[i] - x0[i]) * g0[i];
            s_dot_g1 += (x[i] - x0[i]) * g1[i];
        }
        const double c1 = options.line_search_sufficient_function_decrease;
        const double c2 = options.line_search_sufficient_curvature_decrease;
        CHECK(s_dot_g0 < 0.0); // moved along a descent direction
        CHECK(f1 <= f0 + c1 * s_dot_g0); // sufficient decrease
        CHECK(std::fabs(s_dot_g1) <= c2 * std::fabs(s_dot_g0)); // strong curvature
        std::printf("strong-Wolfe certificate: f %.4e -> %.4e, |s.g1|/|s.g0| = %.3f (c2 = %.1f)\n",
                    f0,
                    f1,
                    std::fabs(s_dot_g1) / std::fabs(s_dot_g0),
                    c2);
    }

    // ---- ill-conditioned quadratic, all direction types ----
    struct Case
    {
        ceres::LineSearchDirectionType dir;
        ceres::LineSearchType ls;
        const char* name;
        double tol;
        int max_iter;
    };
    for(auto c : {Case{ceres::LBFGS, ceres::WOLFE, "LBFGS/Wolfe", 1e-12, 2000},
                  Case{ceres::LBFGS, ceres::ARMIJO, "LBFGS/Armijo", 1e-10, 2000},
                  Case{ceres::NONLINEAR_CONJUGATE_GRADIENT, ceres::WOLFE, "NCG/Wolfe", 1e-8, 20000},
                  Case{ceres::STEEPEST_DESCENT, ceres::WOLFE, "SD/Wolfe", 1e-4, 20000}}) {
        const int n = 16;
        std::vector<double> x(n, 1.0);
        auto summary = run(new Quadratic(n), x, c.dir, c.ls, c.max_iter);
        std::printf("%-13s quadratic κ=1e4: f* = %.3e, iters = %d, evals = %d\n",
                    c.name,
                    summary.final_cost,
                    static_cast<int>(summary.iterations.back().iteration),
                    summary.num_cost_evaluations);
        CHECK(summary.final_cost < c.tol);
    }

    // ---- callback protocol (Xped relies on iteration 0 + every iteration) ----
    {
        const int n = 8;
        std::vector<double> x(n, -1.2);
        CountingCallback counter;
        ceres::GradientProblem problem(new Rosenbrock(n));
        ceres::GradientProblemSolver::Options options;
        options.logging_type = ceres::SILENT;
        options.max_num_iterations = 50;
        options.callbacks.push_back(&counter);
        options.update_state_every_iteration = true;
        ceres::GradientProblemSolver::Summary summary;
        ceres::Solve(options, problem, x.data(), &summary);
        CHECK(counter.calls == static_cast<int>(summary.iterations.size()));
        CHECK(summary.iterations.front().iteration == 0);
        CHECK(counter.last_iteration == summary.iterations.back().iteration);
        CHECK(summary.iterations.back().cost <= summary.iterations.front().cost);
        // problem->function() accessor used by Xped's getCTMSolver()
        CHECK(dynamic_cast<const Rosenbrock*>(problem.function()) != nullptr);
        std::printf("callbacks: %d calls over %d iterations, cost %.3e -> %.3e\n",
                    counter.calls,
                    static_cast<int>(summary.iterations.back().iteration),
                    summary.iterations.front().cost,
                    summary.iterations.back().cost);
        std::printf("%s\n", summary.BriefReport().c_str());
    }

    if(g_failures == 0) {
        std::printf("\nSELF-CONTAINED OPTIMIZER TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d FAILURES\n", g_failures);
    return 1;
}
