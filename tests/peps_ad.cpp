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

#include "toml.hpp"

#ifdef _OPENMP
#    include "omp.h"
#endif

#include "stan/math/rev.hpp"

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

#include "Xped/Util/Permutations.hpp"

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/TomlHelpers.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"
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

#include "Xped/PEPS/iPEPSSolverAD.hpp"

// template <typename Scalar, typename Symmetry>
// class Energy final : public ceres::FirstOrderFunction
// {
// public:
//     Energy(const Xped::Tensor<Scalar, 2, 2, Symmetry>& op, std::shared_ptr<Xped::iPEPS<Scalar, Symmetry, false>> Psi, std::size_t chi)
//         : op(op)
//         , Psi(Psi)
//         , chi(chi)
//     {}

//     ~Energy() override {}

//     bool Evaluate(const double* parameters, double* cost, double* gradient) const override
//     {
//         Psi->set_data(parameters);
//         Xped::CTM<double, Symmetry, false> Jack(Psi, chi);
//         Jack.init();
//         double Eold = 1000.;
//         for(std::size_t step = 0; step < 100; ++step) {
//             Jack.left_move();
//             Jack.right_move();
//             Jack.top_move();
//             Jack.bottom_move();
//             Jack.computeRDM();
//             auto [E_h, E_v] = avg(Jack, op);
//             double E = (E_h.sum() + E_v.sum()) / Jack.cell().uniqueSize();
//             SPDLOG_CRITICAL("Step={:2d}, E={:2.5f}, conv={:2.10g}", step, E, std::abs(E - Eold));
//             if(std::abs(E - Eold) < 1.e-12) { break; }
//             Eold = E;
//         }

//         stan::math::nested_rev_autodiff nested;
//         stan::math::print_stack(std::cout);
//         // auto Psi_ad = std::make_shared<Xped::iPEPS<double, Symmetry, true>>(*Psi);

//         Xped::CTM<double, Symmetry, true> Jim(Jack);
//         Jim.template solve<false>();
//         auto [E_h, E_v] = avg(Jim, op);
//         auto res = (E_h.sum() + E_v.sum()) / Jim.cell().uniqueSize();
//         cost[0] = res.val();
//         if(gradient != nullptr) {
//             SPDLOG_CRITICAL("Backwards pass:");
//             stan::math::grad(res.vi_);
//             std::size_t count = 0;
//             for(auto it = Jim.Psi()->gradbegin(); it != Jim.Psi()->gradend(); ++it) { gradient[count++] = *it; }
//         }
//         return true;
//     }

//     Xped::Tensor<Scalar, 2, 2, Symmetry, false> op;
//     std::shared_ptr<Xped::iPEPS<Scalar, Symmetry, false>> Psi;
//     std::size_t chi;

//     int NumParameters() const override { return Psi->plainSize(); }
// };

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
        // typedef Xped::Sym::ZN<Xped::Sym::SpinU1, 36, double> Symmetry;
        typedef Xped::Sym::U0<double> Symmetry;

        Xped::Qbasis<Symmetry, 1> aux;
        std::unique_ptr<Xped::TwoSiteObservable<Symmetry>> ham;

        auto config_file = args.get<std::string>("config_file", "config.toml");
        toml::value data;
        try {
            data = toml::parse("config.toml");
            // std::cout << data << "\n";
        } catch(const toml::syntax_error& err) {
            std::cerr << "Parsing failed:\n" << err.what() << "\n";
            return 1;
        }

        Xped::UnitCell c;
        if(data.at("ipeps").contains("cell")) {
            c = Xped::UnitCell(toml::get<std::vector<std::vector<std::string>>>(toml::find(data.at("ipeps"), "cell")));
        }

        std::map<std::string, std::any> params = Xped::util::params_from_toml(data.at("model").at("params"));

        std::vector<Xped::Opts::Bond> bs;
        for(const auto& elem : data.at("model").at("bonds").as_array()) {
            auto b = Xped::util::enum_from_toml<Xped::Opts::Bond>(elem);
            bs.push_back(b);
        }
        Xped::Opts::Bond bonds;
        if(bs.size() == 0) {
            bonds = Xped::Opts::Bond::V | Xped::Opts::Bond::H;
        } else {
            bonds = bs[0];
            for(std::size_t i = 1; i < bs.size(); ++i) { bonds = bonds | bs[i]; }
        }

        if constexpr(std::is_same_v<Symmetry, Xped::Sym::SU2<Xped::Sym::SpinSU2>>) {
            if(toml::find(data.at("model"), "name").as_string() == "Heisenberg") {
                ham = std::make_unique<Xped::Heisenberg<Symmetry>>(params, c.pattern, bonds);
                aux.push_back({1}, 2);
                aux.push_back({2}, 1);
            } else if(toml::find(data.at("model"), "name").as_string() == "KondoNecklace") {
                ham = std::make_unique<Xped::KondoNecklace<Symmetry>>(params, c.pattern, bonds);
                aux.push_back({1}, 1);
                aux.push_back({3}, 1);
            } else {
                throw std::invalid_argument("Specified model is not implemented.");
            }
        } else if constexpr(std::is_same_v<Symmetry, Xped::Sym::U1<Xped::Sym::SpinU1, double>>) {
            if(toml::find(data.at("model"), "name").as_string() == "Heisenberg") {
                ham = std::make_unique<Xped::Heisenberg<Symmetry>>(params, c.pattern, bonds);
            } else if(toml::find(data.at("model"), "name").as_string() == "KondoNecklace") {
                ham = std::make_unique<Xped::KondoNecklace<Symmetry>>(params, c.pattern, bonds);
            } else {
                throw std::invalid_argument("Specified model is not implemented.");
            }
            aux.push_back({0}, 2);
            aux.push_back({+1}, 1);
            aux.push_back({-1}, 1);
        } else if constexpr(std::is_same_v<Symmetry, Xped::Sym::ZN<Xped::Sym::SpinU1, 36, double>>) {
            if(toml::find(data.at("model"), "name").as_string() == "Heisenberg") {
                ham = std::make_unique<Xped::Heisenberg<Symmetry>>(params, c.pattern, bonds);
            } else if(toml::find(data.at("model"), "name").as_string() == "KondoNecklace") {
                ham = std::make_unique<Xped::KondoNecklace<Symmetry>>(params, c.pattern, bonds);
            } else {
                throw std::invalid_argument("Specified model is not implemented.");
            }
            aux.push_back({0}, 2);
            aux.push_back({+1}, 1);
            aux.push_back({35}, 1);
        } else if constexpr(std::is_same_v<Symmetry, Xped::Sym::U0<double>>) {
            if(toml::find(data.at("model"), "name").as_string() == "Heisenberg") {
                ham = std::make_unique<Xped::Heisenberg<Symmetry>>(params, c.pattern, bonds);
            } else if(toml::find(data.at("model"), "name").as_string() == "KondoNecklace") {
                ham = std::make_unique<Xped::KondoNecklace<Symmetry>>(params, c.pattern, bonds);
            } else {
                throw std::invalid_argument("Specified model is not implemented.");
            }
            aux.push_back({}, toml::find_or(data.at("ipeps"), "D", 2));
        }
        aux.sort();
        auto Psi = std::make_shared<Xped::iPEPS<double, Symmetry, false>>(c, aux, ham->data_h[0].uncoupledDomain()[0]);
        Psi->setRandom();

        Xped::Opts::Optim o_opts = Xped::Opts::optim_from_toml(data.at("optim"));
        Xped::Opts::CTM c_opts = Xped::Opts::ctm_from_toml(data.at("ctm"));
        Xped::iPEPSSolverAD<Scalar, Symmetry> Jack(o_opts, c_opts);

        Xped::SpinBase<Symmetry> B(1, 2);
        Xped::TwoSiteObservable<Symmetry> SzSz(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        for(auto& t : SzSz.data_h) { t = Xped::tprod(B.Sz(), B.Sz()); }
        for(auto& t : SzSz.data_v) { t = Xped::tprod(B.Sz(), B.Sz()); }
        for(auto& t : SzSz.data_d1) { t = Xped::tprod(B.Sz(), B.Sz()); }
        for(auto& t : SzSz.data_d2) { t = Xped::tprod(B.Sz(), B.Sz()); }

        Xped::TwoSiteObservable<Symmetry> SpSm(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        for(auto& t : SpSm.data_h) { t = Xped::tprod(B.Sp(), B.Sm()); }
        for(auto& t : SpSm.data_v) { t = Xped::tprod(B.Sp(), B.Sm()); }
        for(auto& t : SpSm.data_d1) { t = Xped::tprod(B.Sp(), B.Sm()); }
        for(auto& t : SpSm.data_d2) { t = Xped::tprod(B.Sp(), B.Sm()); }

        Xped::TwoSiteObservable<Symmetry> SmSp(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        for(auto& t : SmSp.data_h) { t = Xped::tprod(B.Sm(), B.Sp()); }
        for(auto& t : SmSp.data_v) { t = Xped::tprod(B.Sm(), B.Sp()); }
        for(auto& t : SmSp.data_d1) { t = Xped::tprod(B.Sm(), B.Sp()); }
        for(auto& t : SmSp.data_d2) { t = Xped::tprod(B.Sm(), B.Sp()); }
        Xped::OneSiteObservable<Symmetry> Sz(c.pattern);
        for(auto& t : Sz.data) { t = B.Sz().data.template trim<1>(); }
        Xped::OneSiteObservable<Symmetry> Sx(c.pattern);
        for(auto& t : Sx.data) { t = B.Sx().data.template trim<1>(); }
        // Xped::OneSiteObservable<Symmetry> sz(c.pattern);
        // for(auto& t : sz.data) { t = B.Sz(1).data.template trim<1>(); }
        // Xped::OneSiteObservable<Symmetry> Szsz(c.pattern);
        // for(auto& t : Szsz.data) { t = (B.Sz(0) * B.Sz(1)).data.template trim<1>(); }
        // Xped::OneSiteObservable<Symmetry> Spsm(c.pattern);
        // for(auto& t : Spsm.data) { t = (B.Sp(0) * B.Sm(1)).data.template trim<1>(); }
        // Xped::OneSiteObservable<Symmetry> Smsp(c.pattern);
        // for(auto& t : Smsp.data) { t = (B.Sm(0) * B.Sp(1)).data.template trim<1>(); }

        // Xped::TwoSiteObservable<Symmetry> SS(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        // for(auto& t : SS.data_h) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // for(auto& t : SS.data_v) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // for(auto& t : SS.data_d1) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // for(auto& t : SS.data_d2) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // Xped::OneSiteObservable<Symmetry> Ssq(c.pattern);
        // for(auto& t : Ssq.data) {
        //     t = std::sqrt(3.) * (Xped::SiteOperator<Scalar, Symmetry>::prod(B.Sdag(0), B.S(0), Symmetry::qvacuum())).data.template trim<1>();
        // }
        Jack.callback = [Sx, Sz, SzSz, SpSm, SmSp](XPED_CONST Xped::CTM<Scalar, Symmetry>& env, std::size_t i) mutable {
            fmt::print("Callback at iteration {}\n", i);
            auto o_Sz = avg(env, Sz);
            auto o_Sx = avg(env, Sx);
            for(auto i = 0ul; i < o_Sz.size(); ++i) {
                fmt::print("Sz={}, Sx={}, |S|={}\n", o_Sz[i], o_Sx[i], std::sqrt(o_Sz[i] * o_Sz[i] + o_Sx[i] * o_Sx[i]));
            }
            // auto o_Ssq = avg(env, Ssq);
            // for(const auto d : o_Ssq) { fmt::print("S²={}\n", d); }
            // auto [o_SS_h, o_SS_v, o_SS_d1, o_SS_d2] = avg(env, SS);
            // for(auto i = 0ul; i < o_SS_h.size(); ++i) { fmt::print("SS_h={}, SS_v={}\n", o_SS_h[i], o_SS_v[i]); }
            // for(auto i = 0ul; i < o_SS_d1.size(); ++i) { fmt::print("SS_d1={}, SS_d2={}\n", o_SS_d1[i], o_SS_d2[i]); }
            auto [o_SzSz_h, o_SzSz_v, o_SzSz_d1, o_SzSz_d2] = avg(env, SzSz);
            auto [o_SpSm_h, o_SpSm_v, o_SpSm_d1, o_SpSm_d2] = avg(env, SpSm);
            auto [o_SmSp_h, o_SmSp_v, o_SmSp_d1, o_SmSp_d2] = avg(env, SmSp);
            for(auto i = 0ul; i < o_SzSz_h.size(); ++i) { fmt::print("SS_h={}\n", o_SzSz_h[i] + 0.5 * (o_SpSm_h[i] + o_SmSp_h[i])); }
            for(auto i = 0ul; i < o_SzSz_v.size(); ++i) { fmt::print("SS_v={}\n", o_SzSz_v[i] + 0.5 * (o_SpSm_v[i] + o_SmSp_v[i])); }
            for(auto i = 0ul; i < o_SzSz_d1.size(); ++i) { fmt::print("SS_d1={}\n", o_SzSz_d1[i] + 0.5 * (o_SpSm_d1[i] + o_SmSp_d1[i])); }
            for(auto i = 0ul; i < o_SzSz_d2.size(); ++i) { fmt::print("SS_d2={}\n", o_SzSz_d2[i] + 0.5 * (o_SpSm_d2[i] + o_SmSp_d2[i])); }
            // auto o_Spsm = avg(env, Spsm);
            // for(const auto d : o_Spsm) { fmt::print("Spsm={}\n", d); }
            //     auto o_Smsp = avg(env, Smsp);
            //     for(const auto d : o_Smsp) { fmt::print("Smsp={}\n", d); }
        };
        Jack.solve<double>(Psi, *ham);
        // ceres::GradientProblem problem(new Energy<Scalar, Symmetry>(ham, Psi, chi));

        // std::vector<Scalar> parameters = Psi->data();

        // ceres::GradientProblemSolver::Options options;
        // // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
        // // options.line_search_direction_type = ceres::STEEPEST_DESCENT;
        // options.line_search_direction_type = ceres::LBFGS;
        // // options.line_search_type = ceres::ARMIJO;
        // options.line_search_type = ceres::WOLFE;
        // options.minimizer_progress_to_stdout = true;
        // options.use_approximate_eigenvalue_bfgs_scaling = true;
        // // options.line_search_interpolation_type = ceres::BISECTION;
        // options.max_num_iterations = 500;
        // options.function_tolerance = 1.e-10;
        // options.parameter_tolerance = 0.;
        // options.update_state_every_iteration = true;
        // ceres::GradientProblemSolver::Summary summary;
        // ceres::Solve(options, problem, parameters.data(), &summary);
        // std::cout << summary.FullReport() << "\n";

#ifdef XPED_CACHE_PERMUTE_OUTPUT
        std::cout << "total hits=" << tree_cache</*shift*/ 0, /*Rank*/ 4, /*CoRank*/ 3, Symmetry>.cache.stats().total_hits()
                  << endl; // Hits for any key
        std::cout << "total misses=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
        std::cout << "hit rate=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

#ifdef XPED_USE_MPI
        MPI_Finalize();
#endif
    }
}
