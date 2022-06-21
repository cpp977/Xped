#include <memory>

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

#include "Xped/NLO/OptimOpts.hpp"
#include "Xped/PEPS/CTMOpts.hpp"

#include "Xped/PEPS/CTMSolver.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
class Energy final : public ceres::FirstOrderFunction
{
public:
    Energy(std::unique_ptr<CTMSolver<Scalar, Symmetry>>&& solver,
           const Xped::Tensor<Scalar, 2, 2, Symmetry>& op,
           std::shared_ptr<Xped::iPEPS<Scalar, Symmetry, false>> Psi)
        : impl(std::move(solver))
        , op(op)
        , Psi(Psi)
    {}

    ~Energy() override {}

    bool Evaluate(const double* parameters, double* cost, double* gradient) const override
    {
        Psi->set_data(parameters);
        cost[0] = impl->solve(Psi, gradient, op, gradient != nullptr);
        return true;
    }

    std::unique_ptr<CTMSolver<Scalar, Symmetry>> impl;
    Xped::Tensor<Scalar, 2, 2, Symmetry, false> op;
    std::shared_ptr<Xped::iPEPS<Scalar, Symmetry, false>> Psi;
    int NumParameters() const override { return Psi->plainSize(); }
};

template <typename Scalar, typename Symmetry>
struct iPEPSSolverAD
{
    template <typename Sc, typename Sym>
    using Hamiltonian = Tensor<Sc, 2, 2, Sym, false>;

    iPEPSSolverAD(Opts::Optim optim_opts, Opts::CTM ctm_opts)
        : optim_opts(optim_opts)
        , ctm_opts(ctm_opts)
    {}

    template <typename HamScalar>
    void solve(std::shared_ptr<iPEPS<Scalar, Symmetry>>& Psi, const Hamiltonian<HamScalar, Symmetry>& H)
    {
        optim_opts.info();
        ctm_opts.info();
        auto Dwain = std::make_unique<CTMSolver<Scalar, Symmetry>>(ctm_opts);
        ceres::GradientProblem problem(new Energy<Scalar, Symmetry>(std::move(Dwain), H, Psi));

        std::vector<Scalar> parameters = Psi->data();

        ceres::GradientProblemSolver::Options options;
        // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
        // options.line_search_direction_type = ceres::STEEPEST_DESCENT;
        switch(optim_opts.alg) {
        case Opts::Algorithm::L_BFGS: options.line_search_direction_type = ceres::LBFGS; break;
        case Opts::Algorithm::CONJUGATE_GRADIENT: options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT; break;
        case Opts::Algorithm::NELDER_MEAD: throw std::invalid_argument("Nelder-Mead optimization is not in ceres.");
        }

        switch(optim_opts.ls) {
        case Opts::Linesearch::WOLFE: options.line_search_type = ceres::WOLFE; break;
        case Opts::Linesearch::ARMIJO: options.line_search_type = ceres::ARMIJO; break;
        }
        options.minimizer_progress_to_stdout = true;
        options.use_approximate_eigenvalue_bfgs_scaling = optim_opts.bfgs_scaling;
        // options.line_search_interpolation_type = ceres::BISECTION;
        options.max_num_iterations = optim_opts.max_steps;
        options.function_tolerance = optim_opts.cost_tol;
        options.parameter_tolerance = optim_opts.step_tol;
        options.gradient_tolerance = optim_opts.grad_tol;
        options.update_state_every_iteration = true;
        ceres::GradientProblemSolver::Summary summary;
        ceres::Solve(options, problem, parameters.data(), &summary);
        std::cout << summary.FullReport() << "\n";
    }

    Opts::Optim optim_opts;
    Opts::CTM ctm_opts;
};

} // namespace Xped
