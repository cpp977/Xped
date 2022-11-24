#include <filesystem>
#include <fstream>
#include <memory>

#include "fmt/color.h"
#include "fmt/core.h"

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

#include "highfive/H5File.hpp"

#include "Xped/NLO/OptimOpts.hpp"
#include "Xped/PEPS/CTMOpts.hpp"

#include "Xped/PEPS/CTMSolver.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, Opts::CTMCheckpoint CPOpts, std::size_t TRank = 2>
class Energy final : public ceres::FirstOrderFunction
{
    template <typename Sym>
    using Hamiltonian = TwoSiteObservable<Sym>;

public:
    Energy(std::unique_ptr<CTMSolver<Scalar, Symmetry, CPOpts, TRank>> solver,
           Hamiltonian<Symmetry>& op,
           std::shared_ptr<iPEPS<Scalar, Symmetry, false>> Psi)
        : impl(std::move(solver))
        , op(op)
        , Psi(Psi)

    {}

    ~Energy() override {}

    bool Evaluate(const double* parameters, double* cost, double* gradient) const override
    {
        Psi->set_data(parameters);
        cost[0] = impl->template solve<double>(Psi, gradient, op, gradient != nullptr);
        return true;
    }
    std::unique_ptr<CTMSolver<Scalar, Symmetry, CPOpts, TRank>> impl;
    Hamiltonian<Symmetry>& op;
    std::shared_ptr<iPEPS<Scalar, Symmetry, false>> Psi;
    int NumParameters() const override { return Psi->plainSize(); }
};

template <typename Scalar, typename Symmetry, Opts::CTMCheckpoint CPOpts = Opts::CTMCheckpoint{}, std::size_t TRank = 2>
struct iPEPSSolverAD
{
    template <typename Sym>
    using Hamiltonian = TwoSiteObservable<Sym>;
    using EnergyFunctor = Energy<Scalar, Symmetry, CPOpts, TRank>;

    iPEPSSolverAD() = delete;

    iPEPSSolverAD(Opts::Optim optim_opts, Opts::CTM ctm_opts, std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi_in, Hamiltonian<Symmetry>& H_in)
        : optim_opts(optim_opts)
        , H(H_in)
        , Psi(Psi_in)
    {
        if(optim_opts.resume) {
            constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
            yas::load<flags>((this->H.file_name() + ".xped").c_str(), *this);
        } else {
            if(optim_opts.load != "") {
                switch(optim_opts.load_format) {
                case Opts::LoadFormat::MATLAB: {
                    Psi->loadFromMatlab(std::filesystem::path(optim_opts.load), "cpp", optim_opts.qn_scale);
                    break;
                }
                case Opts::LoadFormat::NATIVE: {
                    constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                    iPEPS<Scalar, Symmetry> tmp_Psi;
                    yas::load<flags>((this->H.file_name() + "_D" + std::to_string(Psi->D) + ".psi").c_str(), tmp_Psi);
                    Psi = std::make_shared<iPEPS<Scalar, Symmetry>>(std::move(tmp_Psi));
                    break;
                }
                }
                std::cout << Psi->cell().pattern << std::endl << H.data_h.pat << std::endl;
                assert(Psi->cell().pattern == H.data_h.pat);
            }
            problem = std::make_unique<ceres::GradientProblem>(
                new EnergyFunctor(std::move(std::make_unique<CTMSolver<Scalar, Symmetry, CPOpts, TRank>>(ctm_opts)), H, Psi));
            std::filesystem::create_directories(optim_opts.working_directory / optim_opts.logging_directory);
            if(optim_opts.log_format == ".h5") {
                HighFive::File file((optim_opts.working_directory / optim_opts.logging_directory).string() + "/" + this->H.file_name() + ".h5",
                                    optim_opts.resume ? HighFive::File::ReadWrite : HighFive::File::Excl);
                HighFive::DataSpace dataspace = HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED});

                // // Use chunking
                HighFive::DataSetCreateProps props;
                props.add(HighFive::Chunking(std::vector<hsize_t>{10}));

                // Create the dataset
                if(not file.exist("/iteration")) {
                    HighFive::DataSet dataset_i = file.createDataSet("/iteration", dataspace, HighFive::create_datatype<int>(), props);
                }
                if(not file.exist("/cost")) {
                    HighFive::DataSet dataset_c = file.createDataSet("/cost", dataspace, HighFive::create_datatype<double>(), props);
                }
                if(not file.exist("/grad_norm")) {
                    HighFive::DataSet dataset_g = file.createDataSet("/grad_norm", dataspace, HighFive::create_datatype<double>(), props);
                }
            }
            if(not optim_opts.obs_directory.empty()) {
                std::filesystem::create_directories(optim_opts.working_directory / optim_opts.obs_directory);
                HighFive::File file((optim_opts.working_directory / optim_opts.obs_directory).string() + "/" + this->H.file_name() + ".h5",
                                    optim_opts.resume ? HighFive::File::ReadWrite : HighFive::File::Excl);
                H.initObsfile(file);
            }
        }
    }

    template <typename HamScalar>
    void solve()
    {
        Log::on_entry(optim_opts.verbosity,
                      "{}: Model={}(Bonds: V:{}, H:{}, D1: {}, D2: {})",
                      fmt::styled("iPEPSSolverAD", fmt::emphasis::bold),
                      H.format(),
                      (H.bond & Opts::Bond::V) == Opts::Bond::V,
                      (H.bond & Opts::Bond::H) == Opts::Bond::H,
                      (H.bond & Opts::Bond::D1) == Opts::Bond::D1,
                      (H.bond & Opts::Bond::D2) == Opts::Bond::D2);
        Log::on_entry(optim_opts.verbosity, "{}", optim_opts.info());
        Log::on_entry(optim_opts.verbosity, "{}", getCTMSolver()->opts.info());
        Log::on_entry(optim_opts.verbosity, "{}", CPOpts.info());

        // auto Dwain = std::make_unique<CTMSolver<Scalar, Symmetry, CPOpts, TRank>>(ctm_opts);
        // ceres::GradientProblem problem(new Energy<Scalar, Symmetry, CPOpts, TRank>(std::move(Dwain), H, Psi));
        // problem = ceres::GradientProblem(new EnergyFunctor(std::move(Dwain), H, Psi));
        std::vector<Scalar> parameters = Psi->data();

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
        options.logging_type = ceres::SILENT;
        options.minimizer_progress_to_stdout = true;
        options.use_approximate_eigenvalue_bfgs_scaling = optim_opts.bfgs_scaling;
        // options.line_search_interpolation_type = ceres::BISECTION;
        options.max_num_iterations = optim_opts.max_steps;
        options.function_tolerance = optim_opts.cost_tol;
        options.parameter_tolerance = optim_opts.step_tol;
        options.gradient_tolerance = optim_opts.grad_tol;
        options.update_state_every_iteration = true;
        GetStateCallback get_state_c(this->state);
        options.callbacks.push_back(&get_state_c);
        LoggingCallback logging_c(*this);
        options.callbacks.push_back(&logging_c);
        ObsCallback obs_c(*this);
        options.callbacks.push_back(&obs_c);
        CustomCallback custom_c(*this, getCTMSolver()->getCTM());
        options.callbacks.push_back(&custom_c);
        SaveCallback save_c(*this);
        if(optim_opts.save_period > 0) { options.callbacks.push_back(&save_c); }
        ceres::GradientProblemSolver::Summary summary;
        ceres::Solve(options, *problem, parameters.data(), &summary);
        custom_c((summary.iterations.size() > 0) ? summary.iterations.back() : ceres::IterationSummary{});
        obs_c((summary.iterations.size() > 0) ? summary.iterations.back() : ceres::IterationSummary{});
        Log::on_exit(optim_opts.verbosity, "{}", summary.FullReport());
    }

    struct SolverState
    {
        std::size_t current_iter = 0;
        double grad_norm;
        double cost;
        double step_norm;
        template <typename Ar>
        void serialize(Ar& ar)
        {
            ar& YAS_OBJECT_NVP("SolverState", ("current_iter", current_iter), ("grad_norm", grad_norm), ("cost", cost), ("step_norm", step_norm));
        }
    };

    CTMSolver<Scalar, Symmetry, CPOpts, TRank>* getCTMSolver() { return dynamic_cast<const EnergyFunctor*>(problem->function())->impl.get(); }

    Opts::Optim optim_opts;
    std::function<void(XPED_CONST CTM<Scalar, Symmetry, TRank>& ctm, std::size_t)> callback = [](XPED_CONST CTM<Scalar, Symmetry, TRank>&,
                                                                                                 std::size_t) {};
    SolverState state;

    ceres::GradientProblemSolver::Options options;
    Hamiltonian<Symmetry>& H;
    std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi;
    std::unique_ptr<ceres::GradientProblem> problem;

    template <typename Ar>
    void serialize(Ar& ar) const
    {
        ar& YAS_OBJECT_NVP("iPEPSSolverAD",
                           ("CTMSolver", *dynamic_cast<const EnergyFunctor*>(problem->function())->impl),
                           ("Psi", *Psi),
                           ("optim_opts", optim_opts),
                           ("state", state));
    }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        CTMSolver<Scalar, Symmetry, CPOpts, TRank> tmp_CTMSolver;
        iPEPS<Scalar, Symmetry> tmp_Psi;
        ar& YAS_OBJECT_NVP("iPEPSSolverAD", ("CTMSolver", tmp_CTMSolver), ("Psi", tmp_Psi), ("optim_opts", optim_opts), ("state", state));
        // optim_opts.max_steps = optim_opts.max_steps > state.current_iter ? optim_opts.max_steps - state.current_iter : 1;
        Psi = std::make_shared<iPEPS<Scalar, Symmetry>>(std::move(tmp_Psi));
        auto Dwain = std::make_unique<CTMSolver<Scalar, Symmetry, CPOpts, TRank>>(std::move(tmp_CTMSolver));
        problem = std::make_unique<ceres::GradientProblem>(new EnergyFunctor(std::move(Dwain), H, Psi));
    }

    struct CustomCallback : public ceres::IterationCallback
    {
        CustomCallback(const iPEPSSolverAD& s, XPED_CONST CTM<Scalar, Symmetry, TRank>& c)
            : s(s)
            , c(c)
        {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
        {
            s.callback(c, summary.iteration);
            return ceres::SOLVER_CONTINUE;
        }

        const iPEPSSolverAD& s;
        XPED_CONST CTM<Scalar, Symmetry, TRank>& c;
    };

    struct GetStateCallback : public ceres::IterationCallback
    {
        GetStateCallback(SolverState& state_in)
            : state(state_in)
        {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
        {
            state.current_iter = summary.iteration;
            state.grad_norm = summary.gradient_norm;
            state.cost = summary.cost;
            state.step_norm = summary.step_norm;
            return ceres::SOLVER_CONTINUE;
        }
        SolverState& state;
    };

    struct LoggingCallback : public ceres::IterationCallback
    {
        LoggingCallback(const iPEPSSolverAD& solver_in)
            : solver(solver_in)
        {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
        {
            Log::per_iteration(solver.optim_opts.verbosity,
                               "{}{:^4d}: E={:+.8f}, ΔE={:.1e}, |∇|={:.1e}, step time={}, total time={}",
                               fmt::styled("Iteration=", fmt::emphasis::bold),
                               fmt::styled(summary.iteration, fmt::emphasis::bold),
                               summary.cost,
                               summary.cost_change,
                               summary.gradient_norm,
                               util::format_secs(std::chrono::duration<double, std::ratio<1, 1>>(summary.iteration_time_in_seconds)),
                               util::format_secs(std::chrono::duration<double, std::ratio<1, 1>>(summary.cumulative_time_in_seconds)));
            if(solver.optim_opts.log_format == ".h5") {
                HighFive::File file((solver.optim_opts.working_directory / solver.optim_opts.logging_directory).string() + "/" +
                                        solver.H.file_name() + ".h5",
                                    HighFive::File::OpenOrCreate);
                auto insert_elem = [&file](const std::string& name, auto elem) {
                    auto d = file.getDataSet(name);
                    std::size_t curr_size = d.getElementCount();
                    d.resize({curr_size + 1});
                    decltype(elem) elem_a[1] = {elem};
                    d.select({curr_size}, {1}).write(elem_a);
                };
                insert_elem("/iteration", summary.iteration);
                insert_elem("/cost", summary.cost);
                insert_elem("/grad_norm", summary.gradient_norm);
            } else {
                std::ofstream ofs((solver.optim_opts.working_directory / solver.optim_opts.logging_directory).string() + "/" + solver.H.file_name() +
                                      solver.optim_opts.log_format,
                                  std::ios::app);
                ofs << summary.iteration << '\t' << summary.cost << '\t' << summary.gradient_norm << std::endl;
                ofs.close();
            }
            return ceres::SOLVER_CONTINUE;
        }
        const iPEPSSolverAD& solver;
    };

    struct ObsCallback : public ceres::IterationCallback
    {
        ObsCallback(iPEPSSolverAD& solver_in)
            : solver(solver_in)
        {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
        {
            if(solver.optim_opts.display_obs or not solver.optim_opts.obs_directory.empty()) { solver.H.computeObs(solver.getCTMSolver()->getCTM()); }
            if(solver.optim_opts.display_obs) {
                Log::per_iteration(solver.optim_opts.verbosity, "  Observables:\n{}", solver.H.getObsString("    "));
            }
            if(not solver.optim_opts.obs_directory.empty()) {
                HighFive::File file((solver.optim_opts.working_directory / solver.optim_opts.obs_directory).string() + "/" + solver.H.file_name() +
                                        ".h5",
                                    HighFive::File::OpenOrCreate);
                solver.H.obsToFile(file);
            }
            return ceres::SOLVER_CONTINUE;
        }
        iPEPSSolverAD& solver;
    };

    struct SaveCallback : public ceres::IterationCallback
    {
        SaveCallback(const iPEPSSolverAD& solver_in)
            : solver(solver_in)
        {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
        {
            assert(solver.optim_opts.save_period > 0);
            if(summary.iteration == 0) { return ceres::SOLVER_CONTINUE; }
            if(summary.iteration % solver.optim_opts.save_period == 0) {
                yas::file_ostream ofs((solver.H.file_name() + ".xped").c_str(), /*trunc*/ 1);
                constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                yas::save<flags>(ofs, solver);
            }
            return ceres::SOLVER_CONTINUE;
        }
        const iPEPSSolverAD& solver;
    };
};

} // namespace Xped
