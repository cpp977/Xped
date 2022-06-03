#include <cstddef>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef _OPENMP
#    include "omp.h"
#endif

#include "stan/math/rev.hpp"

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/MPS/Mps.hpp"
#include "Xped/MPS/MpsAlgebra.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/Models/Heisenberg.hpp"
#include "Xped/PEPS/Models/KondoNecklace.hpp"
#include "Xped/PEPS/iPEPS.hpp"

#include "Xped/AD/ADTensor.hpp"

#include "Xped/Util/Stopwatch.hpp"

#include "TOOLS/ArgParser.h"

template <typename Scalar, typename Symmetry>
class Energy final : public ceres::FirstOrderFunction
{
public:
    Energy(const Xped::Tensor<Scalar, 2, 2, Symmetry>& op, std::shared_ptr<Xped::iPEPS<Scalar, Symmetry, false>> Psi, std::size_t chi)
        : op(op)
        , Psi(Psi)
        , chi(chi)
    {}

    ~Energy() override {}

    bool Evaluate(const double* parameters, double* cost, double* gradient) const override
    {
        Psi->set_data(parameters);
        Xped::CTM<double, Symmetry, false> Jack(Psi, chi);
        Jack.init();
        double Eold = 1000.;
        for(std::size_t step = 0; step < 100; ++step) {
            Jack.left_move();
            Jack.right_move();
            Jack.top_move();
            Jack.bottom_move();
            Jack.computeRDM();
            auto [E_h, E_v] = avg(Jack, op);
            double E = (E_h.sum() + E_v.sum()) / Jack.cell().size();
            SPDLOG_CRITICAL("Step={:2d}, E={:2.5f}, conv={:2.10g}", step, E, std::abs(E - Eold));
            if(std::abs(E - Eold) < 1.e-12) { break; }
            Eold = E;
        }

        stan::math::nested_rev_autodiff nested;
        stan::math::print_stack(std::cout);
        // auto Psi_ad = std::make_shared<Xped::iPEPS<double, Symmetry, true>>(*Psi);

        Xped::CTM<double, Symmetry, true> Jim(Jack);
        Jim.template solve<false>();
        auto [E_h, E_v] = avg(Jim, op);
        auto res = (E_h.sum() + E_v.sum()) / Jim.cell().size();
        cost[0] = res.val();
        if(gradient != nullptr) {
            SPDLOG_CRITICAL("Backwards pass:");
            stan::math::grad(res.vi_);
            std::size_t count = 0;
            for(auto it = Jim.Psi()->gradbegin(); it != Jim.Psi()->gradend(); ++it) { gradient[count++] = *it; }
        }
        return true;
    }

    Xped::Tensor<Scalar, 2, 2, Symmetry, false> op;
    std::shared_ptr<Xped::iPEPS<Scalar, Symmetry, false>> Psi;
    std::size_t chi;

    int NumParameters() const override { return Psi->plainSize(); }
};

int main(int argc, char* argv[])
{
    {
        // gflags::ParseCommandLineFlags(&argc, &argv, false);

#ifdef XPED_USE_MPI
        MPI_Init(&argc, &argv);
        Xped::mpi::XpedWorld world(argc, argv);
        auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
#else
        Xped::mpi::XpedWorld world;
        auto my_logger = spdlog::basic_logger_mt("info", "logs/log.txt");
#endif
        spdlog::set_default_logger(my_logger);
        std::ios::sync_with_stdio(true);

        ArgParser args(argc, argv);
        google::InitGoogleLogging(argv[0]);
        auto Minit = args.get<std::size_t>("Minit", 3);

        my_logger->sinks()[0]->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
        if(world.rank == 0) {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
            my_logger->sinks().push_back(console_sink);
        }
        spdlog::set_default_logger(my_logger);

        SPDLOG_INFO("Number of MPI processes: {}", world.np);
        SPDLOG_INFO("I am process number #={}", world.rank);
        SPDLOG_INFO("Number of MPI processes: {}", world.np);

        typedef double Scalar;
        // typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        // typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        typedef Xped::Sym::U0<double> Symmetry;

        Xped::Qbasis<Symmetry, 1> aux;
        Xped::Tensor<Scalar, 2, 2, Symmetry> ham;
        auto J = args.get<double>("J", 1.);
        auto I = args.get<double>("I", 0.);
        auto Jk = args.get<double>("Jk", 1.);

        if constexpr(std::is_same_v<Symmetry, Xped::Sym::SU2<Xped::Sym::SpinSU2>>) {
            aux.push_back({1}, 3);
            // aux.push_back({2}, 1);
            aux.push_back({3}, 1);
            ham = Xped::KondoNecklace<Symmetry>::twoSiteHamiltonian(Jk, J, I);
            // ham = Xped::Heisenberg<Symmetry>::twoSiteHamiltonian();
        } else if constexpr(std::is_same_v<Symmetry, Xped::Sym::U0<double>>) {
            aux.push_back({}, Minit);
            // ham = Xped::KondoNecklace<Symmetry>::twoSiteHamiltonian(Jk, Jk, J, J, I, I);
            ham = Xped::Heisenberg<Symmetry>::twoSiteHamiltonian();
        }

        // Xped::Pattern p({{'a', 'b'}, {'c', 'd'}});
        Xped::Pattern p({std::vector<char>{'a'}});
        SPDLOG_CRITICAL("Pattern:\n {}", p);
        auto Lx = args.get<std::size_t>("Lx", 1);
        auto Ly = args.get<std::size_t>("Ly", 1);
        auto Psi = std::make_shared<Xped::iPEPS<double, Symmetry, false>>(Xped::UnitCell(Lx, Ly), aux, ham.uncoupledDomain()[0]);
        Psi->setRandom();
        std::size_t chi = args.get<std::size_t>("chi", 20);

        ceres::GradientProblem problem(new Energy<Scalar, Symmetry>(ham, Psi, chi));

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

#ifdef XPED_CACHE_PERMUTE_OUTPUT
        std::cout << "total hits=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
        std::cout << "total misses=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
        std::cout << "hit rate=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

#ifdef XPED_USE_MPI
        MPI_Finalize();
#endif
    }
}
