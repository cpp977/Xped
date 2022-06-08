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
        auto Dwain = std::make_unique<CTMSolver<Scalar, Symmetry>>(ctm_opts);
        ceres::GradientProblem problem(new Energy<Scalar, Symmetry>(std::move(Dwain), H, Psi));

        std::vector<Scalar> parameters = Psi->data();

        ceres::GradientProblemSolver::Options options;
        // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
        // options.line_search_direction_type = ceres::STEEPEST_DESCENT;
        options.line_search_direction_type = ceres::LBFGS;
        // options.line_search_type = ceres::ARMIJO;
        options.line_search_type = ceres::WOLFE;
        options.minimizer_progress_to_stdout = true;
        options.use_approximate_eigenvalue_bfgs_scaling = true;
        // options.line_search_interpolation_type = ceres::BISECTION;
        options.max_num_iterations = 500;
        options.function_tolerance = 1.e-10;
        options.parameter_tolerance = 0.;
        options.update_state_every_iteration = true;
        ceres::GradientProblemSolver::Summary summary;
        ceres::Solve(options, problem, parameters.data(), &summary);
        std::cout << summary.FullReport() << "\n";
    }

    Opts::Optim optim_opts;
    Opts::CTM ctm_opts;
};

} // namespace Xped
