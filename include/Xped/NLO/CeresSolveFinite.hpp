#include <filesystem>
#include <fstream>
#include <memory>

#include "fmt/color.h"
#include "fmt/core.h"

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

#include "highfive/H5File.hpp"

#include "Xped/NLO/OptimOpts.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/PEPS/Models/Hamiltonian.hpp"

#include "Xped/PEPS/ExactSolver.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, std::size_t TRank = 2>
class Energy final : public ceres::FirstOrderFunction
{
public:
    Energy(std::unique_ptr<ExactSolver<Scalar, Symmetry, TRank>> solver,
           Hamiltonian<double, Symmetry>& op,
           std::shared_ptr<iPEPS<Scalar, Symmetry, true, false>> Psi)
        : impl(std::move(solver))
        , op(op)
        , Psi(Psi)
    {}

    ~Energy() override {}

    bool Evaluate(const double* parameters, double* cost, double* gradient) const override
    {
        if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
            const std::complex<double>* params_compl = reinterpret_cast<const std::complex<double>*>(parameters);
            Psi->set_data(params_compl);
            std::complex<double>* gradient_compl = reinterpret_cast<std::complex<double>*>(gradient);
            cost[0] = impl->template solve<Scalar, true>(Psi, gradient_compl, op);
        } else {
            Psi->set_data(parameters);
            cost[0] = impl->template solve<Scalar, true>(Psi, gradient, op);
        }
        return true;
    }
    std::unique_ptr<ExactSolver<Scalar, Symmetry, TRank>> impl;
    Hamiltonian<double, Symmetry>& op;
    std::shared_ptr<iPEPS<Scalar, Symmetry, true, false>> Psi;
    int NumParameters() const override { return ScalarTraits<Scalar>::IS_COMPLEX() ? 2 * Psi->plainSize() : Psi->plainSize(); }
};

template <typename Scalar, typename Symmetry, std::size_t TRank = 2>
struct fPEPSSolverAD
{
    using EnergyFunctor = Energy<Scalar, Symmetry, TRank>;

    fPEPSSolverAD() = delete;

    fPEPSSolverAD(Opts::Optim optim_opts,
                  Opts::CTM ctm_opts,
                  std::shared_ptr<iPEPS<Scalar, Symmetry, true>> Psi_in,
                  Hamiltonian<double, Symmetry>& H_in)
        : optim_opts(optim_opts)
        , H(H_in)
        , Psi(Psi_in)
    {
        std::filesystem::create_directories(optim_opts.working_directory);
        if(optim_opts.resume) {
            constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
            try {
                yas::load<flags>((optim_opts.working_directory.string() + "/" + this->H.file_name() +
                                  fmt::format("_D={}_seed={}_id={}.ad", Psi->D, optim_opts.seed, optim_opts.id))
                                     .c_str(),
                                 *this);
            } catch(const std::exception& e) {
                fmt::print("Error while loading file ({}) for resuming simulation.\n",
                           optim_opts.working_directory.string() + "/" + this->H.file_name() +
                               fmt::format("_D={}_seed={}_id={}.ad", Psi->D, optim_opts.seed, optim_opts.id));
                std::cout << std::flush;
                throw;
            }
        } else {
            if(optim_opts.load != "") {
                std::filesystem::path load_p(optim_opts.load);
                if(load_p.is_relative()) { load_p = optim_opts.working_directory / load_p; }
                switch(optim_opts.load_format) {
                case Opts::LoadFormat::MATLAB: {
                    // Psi->loadFromMatlab(load_p, "cpp", optim_opts.qn_scale);
                    break;
                }
                case Opts::LoadFormat::JSON: {
                    // Psi->loadFromJson(load_p);
                    // auto perm1 = Psi->As[0].template permute<0, 1, 2, 3, 0, 4>();
                    // fmt::print("check1={}\n", (Psi->As[0] - perm1).norm());
                    // auto perm2 = Psi->As[0].template permute<0, 0, 3, 2, 1, 4>();
                    // fmt::print("check2={}\n", (Psi->As[0] - perm2).norm());
                    break;
                }

                case Opts::LoadFormat::Native: {
                    constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                    iPEPS<Scalar, Symmetry, true> tmp_Psi;
                    try {
                        yas::load<flags>(load_p.string().c_str(), tmp_Psi);
                    } catch(const yas::serialization_exception& se) {
                        fmt::print(
                            "Error while deserializing file ({}) with initial wavefunction.\nThis might be because of incompatible symmetries between this simulation and the loaded wavefunction.",
                            load_p.string());
                        std::cout << std::flush;
                        throw;
                    } catch(const yas::io_exception& ie) {
                        fmt::print("Error while loading file ({}) with initial wavefunction.\n", load_p.string());
                        std::cout << std::flush;
                        throw;
                    } catch(const std::exception& e) {
                        fmt::print("Unknown error while loading file ({}) with initial wavefunction.\n", load_p.string());
                        std::cout << std::flush;
                        throw;
                    }
                    Psi = std::make_shared<iPEPS<Scalar, Symmetry, true>>(std::move(tmp_Psi));
                    break;
                }
                }
                assert(Psi->cell().pattern == H.data_h.pat);
            } else {
                Psi->setRandom(optim_opts.seed);
            }

            problem = std::make_unique<ceres::GradientProblem>(
                new EnergyFunctor(std::move(std::make_unique<ExactSolver<Scalar, Symmetry, TRank>>(ctm_opts.verbosity)), H, Psi));
            std::filesystem::create_directories(optim_opts.working_directory / optim_opts.logging_directory);
            if(optim_opts.log_format == ".h5") {
                try {
                    HighFive::File file((optim_opts.working_directory / optim_opts.logging_directory).string() + "/" + this->H.file_name() +
                                            fmt::format("_D={}_seed={}_id={}.h5", Psi->D, optim_opts.seed, optim_opts.id),
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
                } catch(const std::exception& e) {
                    fmt::print(fg(fmt::color::red),
                               "There already exists a log file for this simulation:{}.\n",
                               (optim_opts.working_directory / optim_opts.logging_directory).string() + "/" + this->H.file_name() +
                                   fmt::format("_D={}_seed={}_id={}.h5", Psi->D, optim_opts.seed, optim_opts.id));
                    std::cout << std::flush;
                    throw;
                }
            }
            if(not optim_opts.obs_directory.empty()) {
                std::filesystem::create_directories(optim_opts.working_directory / optim_opts.obs_directory);
                try {
                    HighFive::File file((optim_opts.working_directory / optim_opts.obs_directory).string() + "/" + this->H.file_name() +
                                            fmt::format("_D={}_seed={}_id={}.h5", Psi->D, optim_opts.seed, optim_opts.id),
                                        optim_opts.resume ? HighFive::File::ReadWrite : HighFive::File::Excl);
                } catch(const std::exception& e) {
                    fmt::print(fg(fmt::color::red),
                               "Error while creating the observable file for this simulation:{}.\n",
                               (optim_opts.working_directory / optim_opts.obs_directory).string() + "/" + this->H.file_name() +
                                   fmt::format("_D={}_seed={}_id={}.h5", Psi->D, optim_opts.seed, optim_opts.id));
                    std::cout << std::flush;
                    throw;
                }
            }
        }
    }

    template <typename HamScalar>
    void solve()
    {
        Log::on_entry(optim_opts.verbosity,
                      "{}: Model={}(Bonds: V:{}, H:{}, D1: {}, D2: {})",
                      fmt::styled("fPEPSSolverAD", fmt::emphasis::bold),
                      H.format(),
                      (H.bond & Opts::Bond::V) == Opts::Bond::V,
                      (H.bond & Opts::Bond::H) == Opts::Bond::H,
                      (H.bond & Opts::Bond::D1) == Opts::Bond::D1,
                      (H.bond & Opts::Bond::D2) == Opts::Bond::D2);
        Log::on_entry(optim_opts.verbosity, "{}", optim_opts.info());

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
        options.max_lbfgs_rank = optim_opts.max_lbfgs_rank;
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
        // CustomCallback custom_c(*this, getCTMSolver()->getCTM());
        // options.callbacks.push_back(&custom_c);
        SaveCallback save_c(*this);
        if(optim_opts.save_period > 0) { options.callbacks.push_back(&save_c); }
        for(auto rep = 0ul; rep < optim_opts.restarts; ++rep) {
            Log::on_entry(optim_opts.verbosity, "Do {} optimization (repetition={})", fmt::streamed(optim_opts.alg), rep);
            ceres::GradientProblemSolver::Summary summary;
            if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
                typename ScalarTraits<Scalar>::Real* params_real = reinterpret_cast<double*>(parameters.data());
                ceres::Solve(options, *problem, params_real, &summary);
            } else {
                ceres::Solve(options, *problem, parameters.data(), &summary);
            }
            // custom_c((summary.iterations.size() > 0) ? summary.iterations.back() : ceres::IterationSummary{});
            obs_c((summary.iterations.size() > 0) ? summary.iterations.back() : ceres::IterationSummary{});
            Log::on_exit(optim_opts.verbosity, "{}", summary.FullReport());
        }
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

    ExactSolver<Scalar, Symmetry, TRank>* getSolver() { return dynamic_cast<const EnergyFunctor*>(problem->function())->impl.get(); }
    const ExactSolver<Scalar, Symmetry, TRank>* getSolver() const { return dynamic_cast<const EnergyFunctor*>(problem->function())->impl.get(); }

    Opts::Optim optim_opts;
    // std::function<void(XPED_CONST CTM<Scalar, Symmetry, TRank>& ctm, std::size_t)> callback = [](XPED_CONST CTM<Scalar, Symmetry, TRank>&,
    //                                                                                              std::size_t) {};
    SolverState state;

    ceres::GradientProblemSolver::Options options;
    Hamiltonian<double, Symmetry>& H;
    std::shared_ptr<iPEPS<Scalar, Symmetry, true>> Psi;
    std::unique_ptr<ceres::GradientProblem> problem;

    template <typename Ar>
    void serialize(Ar& ar) const
    {
        ar& YAS_OBJECT_NVP("fPEPSSolverAD", ("Psi", *Psi), ("optim_opts", optim_opts), ("state", state));
    }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        iPEPS<Scalar, Symmetry, true> tmp_Psi;
        ar& YAS_OBJECT_NVP("fPEPSSolverAD", ("Psi", tmp_Psi), ("optim_opts", optim_opts), ("state", state));
        Psi = std::make_shared<iPEPS<Scalar, Symmetry, true>>(std::move(tmp_Psi));

        auto Dwain = std::make_unique<ExactSolver<Scalar, Symmetry, TRank>>(optim_opts.verbosity);
        problem = std::make_unique<ceres::GradientProblem>(new EnergyFunctor(std::move(Dwain), H, Psi));
    }

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
        LoggingCallback(const fPEPSSolverAD& solver_in)
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
                                        solver.H.file_name() +
                                        fmt::format("_D={}_seed={}_id={}.h5", solver.Psi->D, solver.optim_opts.seed, solver.optim_opts.id),
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
                std::ofstream ofs(
                    (solver.optim_opts.working_directory / solver.optim_opts.logging_directory).string() + "/" + solver.H.file_name() +
                        fmt::format(
                            "_D={}_seed={}_id={}{}", solver.Psi->D, solver.optim_opts.seed, solver.optim_opts.id, solver.optim_opts.log_format),
                    std::ios::app);
                ofs << summary.iteration << '\t' << std::setprecision(12) << summary.cost << '\t' << summary.gradient_norm << std::endl;
                ofs.close();
            }
            return ceres::SOLVER_CONTINUE;
        }
        const fPEPSSolverAD& solver;
    };

    struct ObsCallback : public ceres::IterationCallback
    {
        ObsCallback(fPEPSSolverAD& solver_in)
            : solver(solver_in)
        {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
        {
            if(solver.optim_opts.display_obs or not solver.optim_opts.obs_directory.empty()) {
                auto rho = solver.getSolver()->rho;
                auto rho1 = solver.getSolver()->rho1;
                auto rho_d = solver.getSolver()->rho_d;
                for(auto& ob : solver.H.obs) {
                    if(auto* one = dynamic_cast<OneSiteObservable<double, Symmetry>*>(ob.get()); one != nullptr) { avg(rho1, *one); }
                    if(auto* one_c = dynamic_cast<OneSiteObservable<std::complex<double>, Symmetry>*>(ob.get()); one_c != nullptr) {
                        avg(rho1, *one_c);
                    }
                    if(auto* two = dynamic_cast<TwoSiteObservable<double, Symmetry>*>(ob.get()); two != nullptr) {
                        avg(rho, *two, Opts::Bond::H);
                        avg(rho_d, *two, Opts::Bond::D1);
                    }
                    if(auto* two_c = dynamic_cast<TwoSiteObservable<std::complex<double>, Symmetry>*>(ob.get()); two_c != nullptr) {
                        avg(rho, *two_c, Opts::Bond::H);
                        avg(rho_d, *two_c, Opts::Bond::D1);
                    }
                }
            }
            if(solver.optim_opts.display_obs) {
                Log::per_iteration(solver.optim_opts.verbosity, "  Observables:\n{}", solver.H.getObsString("    "));
            }
            if(not solver.optim_opts.obs_directory.empty()) {
                HighFive::File file((solver.optim_opts.working_directory / solver.optim_opts.obs_directory).string() + "/" + solver.H.file_name() +
                                        fmt::format("_D={}_seed={}_id={}.h5", solver.Psi->D, solver.optim_opts.seed, solver.optim_opts.id),
                                    HighFive::File::OpenOrCreate);
                std::string e_name = fmt::format("/energy");
                if(not file.exist(e_name)) {
                    HighFive::DataSpace dataspace = HighFive::DataSpace({0, 0}, {HighFive::DataSpace::UNLIMITED, HighFive::DataSpace::UNLIMITED});

                    // Use chunking
                    HighFive::DataSetCreateProps props;
                    props.add(HighFive::Chunking(std::vector<hsize_t>{2, 2}));

                    // Create the dataset
                    HighFive::DataSet dataset = file.createDataSet(e_name, dataspace, HighFive::create_datatype<double>(), props);
                }
                {
                    auto d = file.getDataSet(e_name);
                    std::vector<std::vector<double>> data;
                    data.push_back(std::vector<double>(1, summary.cost));
                    std::size_t curr_size = d.getDimensions()[0];
                    d.resize({curr_size + 1, data[0].size()});
                    d.select({curr_size, 0}, {1, data[0].size()}).write(data);
                }
                std::string g_name = fmt::format("/grad");
                if(not file.exist(g_name)) {
                    HighFive::DataSpace dataspace = HighFive::DataSpace({0, 0}, {HighFive::DataSpace::UNLIMITED, HighFive::DataSpace::UNLIMITED});

                    // Use chunking
                    HighFive::DataSetCreateProps props;
                    props.add(HighFive::Chunking(std::vector<hsize_t>{2, 2}));

                    // Create the dataset
                    HighFive::DataSet dataset = file.createDataSet(g_name, dataspace, HighFive::create_datatype<double>(), props);
                }
                {
                    auto d = file.getDataSet(g_name);
                    std::vector<std::vector<double>> data;
                    data.push_back(std::vector<double>(1, summary.gradient_norm));
                    std::size_t curr_size = d.getDimensions()[0];
                    d.resize({curr_size + 1, data[0].size()});
                    d.select({curr_size, 0}, {1, data[0].size()}).write(data);
                }
                solver.H.obsToFile(file, fmt::format("/"));
            }
            return ceres::SOLVER_CONTINUE;
        }
        fPEPSSolverAD& solver;
    };

    struct SaveCallback : public ceres::IterationCallback
    {
        SaveCallback(const fPEPSSolverAD& solver_in)
            : solver(solver_in)
        {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
        {
            assert(solver.optim_opts.save_period > 0);
            if(summary.iteration == 0) { return ceres::SOLVER_CONTINUE; }
            if(summary.iteration % solver.optim_opts.save_period == 0) {
                constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                yas::file_ostream ofs_ad((solver.optim_opts.working_directory.string() + "/" + solver.H.file_name() +
                                          fmt::format("_D={}_seed={}_id={}.ad", solver.Psi->D, solver.optim_opts.seed, solver.optim_opts.id))
                                             .c_str(),
                                         /*trunc*/ 1);
                yas::save<flags>(ofs_ad, solver);
                yas::file_ostream ofs_psi((solver.optim_opts.working_directory.string() + "/" + solver.H.file_name() +
                                           fmt::format("_D={}_seed={}_id={}.psi", solver.Psi->D, solver.optim_opts.seed, solver.optim_opts.id))
                                              .c_str(),
                                          /*trunc*/ 1);
                yas::save<flags>(ofs_psi, *solver.Psi);
            }
            return ceres::SOLVER_CONTINUE;
        }
        const fPEPSSolverAD& solver;
    };
};

} // namespace Xped
