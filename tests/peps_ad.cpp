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

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#ifdef XPED_USE_AD
#    include "stan/math/rev/core/autodiffstackstorage.hpp"
#    include "stan/math/rev/core/chainablestack.hpp"
#    include "stan/math/rev/core/init_chainablestack.hpp"
#    include "stan/math/rev/core/operator_division.hpp"
#    include "stan/math/rev/core/print_stack.hpp"
#    include "stan/math/rev/core/var.hpp"

#    include "ceres/first_order_function.h"
#    include "ceres/gradient_problem.h"
#    include "ceres/gradient_problem_solver.h"
#endif

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

#include "Xped/Util/Stopwatch.hpp"

#include "TOOLS/ArgParser.h"

#ifdef XPED_USE_AD
#    include "Xped/AD/ADTensor.hpp"
template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
inline stan::math::var dot_self(const Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry, true>& v)
{
    stan::math::var res = v.val().squaredNorm();
    stan::math::reverse_pass_callback([res, v]() mutable { v.adj() += (v.val() * (2.0 * res.adj())).eval(); });
    return res;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
inline stan::math::var my_func(const Xped::ArenaTensor<Scalar, Rank, CoRank, Symmetry, true>& t)
{
    return dot_self(t - Scalar(5));
}

template <typename Scalar, typename Symmetry>
inline stan::math::var avg(const Xped::Tensor<Scalar, 0, 2, Symmetry, false>& bra,
                           const Xped::Tensor<Scalar, 2, 2, Symmetry, false>& op,
                           const Xped::Tensor<Scalar, 2, 0, Symmetry, true>& ket)
{
    auto opvec = op * ket;
    auto energy = bra * opvec;
    auto norm = bra * ket;
    auto res = energy / norm;
    return res;
}

template <typename Scalar, typename Symmetry>
inline stan::math::var avg2(const Xped::Tensor<Scalar, 2, 2, Symmetry>& op, const Xped::Tensor<Scalar, 2, 0, Symmetry, true>& ket)
{
    auto opvec = op * ket;
    auto energy = ket.adjoint() * opvec;
    auto norm = ket.adjoint() * ket;
    auto res = energy / norm;
    return res;
}

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
        // Xped::Tensor<Scalar, 2, 3, Symmetry, false> A_({{aux, aux}}, {{aux, aux, phys}}, parameters, NumParameters(), op.world());
        // Xped::Tensor<Scalar, 2, 3, Symmetry, true> A(A_);
        stan::math::nested_rev_autodiff nested;
        stan::math::print_stack(std::cout);
        auto Psi_ad = std::make_shared<Xped::iPEPS<double, Symmetry, true>>(Psi->cell,
                                                                            Psi->ketBasis(0, 0, Xped::iPEPS<double, Symmetry, false>::LEG::LEFT),
                                                                            Psi->ketBasis(0, 0, Xped::iPEPS<double, Symmetry, false>::LEG::PHYS));
        Psi_ad->setZero();
        // Psi->As[0] = A;
        // std::cout << "params=";
        // for(auto i = 0; i < NumParameters(); ++i) { std::cout << parameters[i] << " "; }
        // std::cout << std::endl;
        Psi_ad->set_data(parameters);

        Xped::CTM<double, Symmetry, true> Jack(Psi_ad, chi);
        Jack.init();
        // {
        //     stan::math::nested_rev_autodiff nested;
        //     Jack.init();
        //     double E, E_prev = 1000;
        //     for(std::size_t step = 0; step < 80; ++step) {
        //         SPDLOG_CRITICAL("Pre step={}", step);
        //         Jack.left_move();
        //         Jack.right_move();
        //         Jack.top_move();
        //         Jack.bottom_move();
        //         Jack.computeRDM();
        //         auto [E_h, E_v] = avg(Jack, op);
        //         auto res = (E_h.sum() + E_v.sum()) / Jack.cell.size();
        //         E = res.val();
        //         SPDLOG_CRITICAL("Energy={}, diff={}", E, std::abs(E_prev - E));
        //         if(std::abs(E_prev - E) < 1.e-12) {
        //             SPDLOG_CRITICAL("CTM converged in #{} steps.", step);
        //             break;
        //         }
        //         E_prev = E;
        //     }
        // }
        for(std::size_t step = 0; step < 20; ++step) {
            SPDLOG_CRITICAL("Step={}", step);
            Jack.left_move();
            Jack.right_move();
            Jack.top_move();
            Jack.bottom_move();
        }
        Jack.computeRDM();
        // Jack.solve();
        auto [E_h, E_v] = avg(Jack, op);
        auto res = (E_h.sum() + E_v.sum()) / Jack.cell.size();
        // std::cout << t << std::endl;
        // auto res = avg2(op, t);
        // auto res = avg(t.val().adjoint().eval(), op, t);
        // fmt::print("Energy={}\n", res.val());
        cost[0] = res.val();
        if(gradient != nullptr) {
            SPDLOG_CRITICAL("Backwards pass:");
            stan::math::grad(res.vi_);
            // memcpy(gradient, Psi->data(), NumParameters() * sizeof(Scalar));
            // if constexpr(Psi->CONTIGUOUS_STORAGE()) {
            //     memcpy(gradient, Psi->data(), NumParameters() * sizeof(Scalar));
            // } else {
            // Psi->As[0].adj().print(std::cout, true);
            std::size_t count = 0;
            for(auto it = Psi_ad->gradbegin(); it != Psi_ad->gradend(); ++it) { gradient[count++] = *it; }
            // Eigen::Map<Eigen::VectorXd> tmp(gradient, NumParameters());
            // fmt::print("Gradient={}\n", tmp.lpNorm<Eigen::Infinity>());
            // }
            // std::cout << "gradient=";
            // for(auto i = 0; i < NumParameters(); ++i) { std::cout << gradient[i] << " "; }
            // std::cout << std::endl;
        }
        return true;
    }

    Xped::Tensor<Scalar, 2, 2, Symmetry, false> op;
    std::shared_ptr<Xped::iPEPS<Scalar, Symmetry, false>> Psi;
    std::size_t chi;

    int NumParameters() const override { return Psi->plainSize(); }
};

#endif

int main(int argc, char* argv[])
{
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

    // Xped<double, 2, 2, Symmetry_> t({{B, C}}, {{B, C}}, World);
    // t.setRandom();
    // std::cout << t.print() << std::endl;

    // std::cout << t << std::endl;
    // SPDLOG_INFO("Tensor: \n {}", t);
    // constexpr std::size_t Rank=2;
    // tensortraits<EigenTensorLib>::Ttype<double,Rank> T1(2,3); T1.setRandom();
    // tensortraits<EigenTensorLib>::Ttype<double,Rank> T2(2,3); T2.setRandom();
    // std::array<std::pair<std::size_t,std::size_t>,1> legs = {std::pair<std::size_t,std::size_t>{0,0}};
    // auto C = tensortraits<EigenTensorLib>::contract(T1,T2,legs);
    // auto T = tensortraits<EigenTensorLib>::tensorProd(T1,T1);
    // auto Ts = tensortraits<EigenTensorLib>::shuffle(C, {{1,0}});
    // std::array<std::size_t, 2> dims = {2,3};
    // auto cmap = tensortraits<EigenTensorLib>::cMap(T.data(), dims);
    // auto map = tensortraits<EigenTensorLib>::Map(T.data(), dims);
    // map(0,0)=10;
    // std::cout << cmap(0,0) << std::endl;

    // std::cout << T << std::endl;

    // std::cout << Ts << std::endl;

    // std::cout << std::endl << C << std::endl;

    typedef double Scalar;
    typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
    // typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
    // typedef Xped::Sym::U0<double> Symmetry;
#ifdef XPED_USE_AD
    {
        Xped::Qbasis<Symmetry, 1> aux;
        // aux.push_back({}, Minit);
        aux.push_back({1}, 1);
        // aux.push_back({2}, 1);
        aux.push_back({3}, 1);
        auto J = args.get<double>("J", 1.);
        auto I = args.get<double>("I", 0.);
        auto Jk = args.get<double>("Jk", 1.);
        auto ham = Xped::KondoNecklace<Symmetry>::twoSiteHamiltonian(Jk, J, I);
        // auto ham = Xped::KondoNecklace<Symmetry>::twoSiteHamiltonian(Jk, Jk, J, J, I, I);
        // auto ham = Xped::Heisenberg<Symmetry>::twoSiteHamiltonian();

        // Xped::Qbasis<Symmetry, 1> phys;
        // phys.push_back({}, 4);
        // phys.push_back({1}, 1);
        // phys.push_back({3}, 1);
        // Xped::Pattern p({{'a', 'b'}, {'c', 'd'}});
        Xped::Pattern p({std::vector<char>{'a'}});
        SPDLOG_CRITICAL("Pattern:\n {}", p);
        auto Lx = args.get<std::size_t>("Lx", 1);
        auto Ly = args.get<std::size_t>("Ly", 1);
        auto Psi = std::make_shared<Xped::iPEPS<double, Symmetry, false>>(Xped::UnitCell(Lx, Ly), aux, ham.uncoupledDomain()[0]);
        Psi->setRandom();
        std::cout << Psi->As[0] << std::endl << "done" << std::endl;
        // Xped::Tensor<Scalar, 2, 3, Symmetry> A_({{aux, aux}}, {{aux, aux, phys}}, world);
        // t_.setRandom();
        // Xped::Tensor<Scalar, 2, 0, Symmetry, true> t(t_);
        // std::cout << t << std::endl;
        // Xped::Qbasis<Symmetry, 1, Xped::StanArenaPolicy> phys;
        // phys.push_back({2}, 1);
        // phys.push_back({}, 2);
        // Xped::ArenaTensor<double, 2, 3, Symmetry> t({{B, B}}, {{B, B, phys}}, world);
        // t.setRandom();
        // vec_.print(std::cout, true);
        // Xped::Tensor<Scalar, 2, 2, Symmetry> op({{B, B}}, {{B, B}}, world);
        // op.setRandom();
        // op = op + op.adjoint();
        // auto check = (vec_.adjoint() * op).adjoint().eval();
        // auto check = (op + op.adjoint()) * vec_;
        // std::cout << ham << std::endl;
        // auto [eigvals, eigvecs] = ham.eigh();
        // for(auto i = 0ul; i < eigvals.sector().size(); ++i) {
        //     std::cout << "Q=" << Xped::Sym::format<Symmetry>(eigvals.sector(i)) << ": " << std::setprecision(15)
        //               << eigvals.block(i).diagonal().transpose() << std::endl;
        // }

        std::size_t chi = args.get<std::size_t>("chi", 20);

        ceres::GradientProblem problem(new Energy<Scalar, Symmetry>(ham, Psi, chi));

        std::vector<Scalar> parameters = Psi->data();
        // Scalar* parameters = (Scalar*)malloc(problem.NumParameters() * sizeof(Scalar));
        // for(std::size_t i = 0; i < problem.NumParameters(); ++i) { parameters[i] = Xped::random::threadSafeRandUniform<Scalar>(-1., 1., false); }

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
        // free(parameters);
        // auto [eigvals, eigvecs] = op.eigh();
        // for(auto i = 0ul; i < eigvals.sector().size(); ++i) {
        //     std::cout << "Q=" << Xped::Sym::format<Symmetry>(eigvals.sector(i)) << ": " << std::setprecision(15) << eigvals.block(i)(0, 0)
        //               << std::endl;
        // }
        // t.print(std::cout, true);
        // t += 7.;
        // t.print(std::cout, true);
        // t = 3 * t;
        // t.print(std::cout, true);
        // stan::math::var_value<Xped::ArenaTensor<double, 2, 0, Symmetry>> vec(vec_);
        // // auto res = avg(vec_.adjoint().eval(), op, vec);
        // auto res = avg2(op, vec);
        // stan::math::print_stack(std::cout);
        // stan::math::grad(res.vi_);
        // std::cout << vec.vi_ << std::endl;
        // check.print(std::cout, true);
    }
    // return 0;
#endif

    // Xped::Qbasis<Symmetry, 1> in;
    // in.push_back({0}, 2);
    // in.push_back({-1}, 1);
    // in.push_back({+1}, 1);
    // in.push_back({1}, 1);
    // in.push_back({2}, 1);
    // in.push_back({}, 2);
    // SPDLOG_CRITICAL("Auxiliary basis (dim={}):\n {}", in.fullDim(), in);
    // std::cout << in << std::endl;
    // Xped::Qbasis<Symmetry, 1> phys;
    // phys.push_back({1}, 1);
    // phys.push_back({3}, 1);
    // phys.push_back({}, 2);
    // phys.push_back({+1}, 1);
    // phys.push_back({-1}, 1);
    // Xped::Qbasis<Symmetry, 1> op;
    // op.push_back({1}, 1);
    // op.push_back({3}, 1);
    // op.sort();
    // Xped::Qbasis<Symmetry, 1> aux;
    // aux.push_back({1}, 2);
    // aux.push_back({3}, 1);
    // aux.sort();
    // Xped::Tensor<double, 2, 1, Symmetry> T({{phys, op}}, {{phys}});
    // T.setRandom();
    // // T.print(std::cout, true);
    // Xped::FusionTree<2, Symmetry> fket;
    // fket.q_uncoupled = {{{2}, {1}}};
    // fket.q_coupled = {2};
    // fket.dims = {{1, 1}};
    // fket.computeDim();
    // Xped::FusionTree<1, Symmetry> fbra;
    // fbra.q_uncoupled = {{{2}}};
    // fbra.q_coupled = {2};
    // fbra.dims = {{1}};
    // fbra.computeDim();
    // T.view(fket, fbra).setConstant(1.);

    // fket.q_uncoupled = {{{2}, {3}}};
    // fket.q_coupled = {2};
    // fket.dims = {{1, 1}};
    // fket.computeDim();
    // T.view(fket, fbra).setConstant(std::sqrt(0.75));
    // std::cout << T.permute<0, 1, 2, 0>().plainTensor() << std::endl;

    // Xped::Tensor<double, 2, 1, Symmetry> X({{aux, op}}, {{aux}});
    // X.setZero();
    // // X.print(std::cout, true);
    // fket.q_uncoupled = {{{1}, {1}}};
    // fket.q_coupled = {1};
    // fket.dims = {{2, 1}};
    // fket.computeDim();
    // fbra.q_uncoupled = {{{1}}};
    // fbra.q_coupled = {1};
    // fbra.dims = {{2}};
    // fbra.computeDim();
    // Eigen::Tensor<double, 3> id(2, 1, 2);
    // id.setZero();
    // id(0, 0, 0) = 1.;
    // id(1, 0, 1) = 1.;
    // X.view(fket, fbra) = id;

    // fket.q_uncoupled = {{{1}, {3}}};
    // fket.q_coupled = {3};
    // fket.dims = {{2, 1}};
    // fket.computeDim();
    // fbra.q_uncoupled = {{{3}}};
    // fbra.q_coupled = {3};
    // fbra.dims = {{1}};
    // fbra.computeDim();
    // Eigen::Tensor<double, 3> S1(2, 1, 1);
    // S1.setZero();
    // S1(0, 0, 0) = 1.;
    // S1(1, 0, 0) = 0.;
    // X.view(fket, fbra) = S1;

    // fket.q_uncoupled = {{{3}, {3}}};
    // fket.q_coupled = {1};
    // fket.dims = {{1, 1}};
    // fket.computeDim();
    // fbra.q_uncoupled = {{{1}}};
    // fbra.q_coupled = {1};
    // fbra.dims = {{2}};
    // fbra.computeDim();
    // Eigen::Tensor<double, 3> S2(1, 1, 2);
    // S2.setZero();
    // S2(0, 0, 0) = 0.;
    // S2(0, 0, 1) = -std::sqrt(3.);
    // X.view(fket, fbra) = S2;

    // std::cout << X.permute<0, 1, 0, 2>().plainTensor() << std::endl;

    // Xped::Tensor<double, 2, 2, Symmetry> W = T.permute<0, 0, 2, 1>() * X.permute<+1, 1, 0, 2>();
    // W.print(std::cout, true);
    // std::cout << std::endl << W.plainTensor() << std::endl;

    // auto Psi = std::make_shared<Xped::iPEPS<double, Symmetry, false>>(Xped::UnitCell(3, 2), in, phys);
    // Psi->setRandom();
    // Psi->As(0, 0).print(std::cout, true);

    // auto J = args.get<double>("J", 1.);
    // auto I = args.get<double>("I", 0.);
    // auto Jk = args.get<double>("Jk", 1.);
    // auto ham = Xped::KondoNecklace<Symmetry>::twoSiteHamiltonian(Jk, J, I);
    // std::cout << ham << std::endl;
    // auto [eigvals, eigvecs] = ham.eigh();
    // for(auto i = 0ul; i < eigvals.sector().size(); ++i) {
    //     std::cout << "Q=" << Xped::Sym::format<Symmetry>(eigvals.sector(i)) << ": " << std::setprecision(15)
    //               << eigvals.block(i).diagonal().transpose() << std::endl;
    // }
    // Psi.info();
    // std::size_t chi = args.get<std::size_t>("chi", 100);
    // Xped::CTM<double, Symmetry, false> Jack(Psi, chi);
    // Jack.init();
    // Jack.solve();
    // Jack.info();
    // auto Y = T.adjoint() * T;
    // std::cout << Y.print(true) << std::endl;

#ifdef XPED_CACHE_PERMUTE_OUTPUT
    std::cout << "total hits=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "total misses=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "hit rate=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

    stan::math::recover_memory();
// double trunc;
// auto [U, S, V] = Psi.A.Ac[L / 2].tSVD(10000ul, 1.e-10, trunc, false);
// Mps<Symmetry> Phi(L, qloc, Qtot, Minit, Qinit);
// cout << "<Phi,Psi>=" << dot(Phi, Psi) << endl;
// cout << "<Phi,Phi>=" << dot(Phi, Phi) << endl;
// double normSq = dot(Psi, Psi, DIR);
// std::cout << "<Psi|Psi>=" << normSq << std::endl;
// cout << "Left: <Psi,Psi>=" << normSq << endl;
// Psi.A.Ac[0] = Psi.A.Ac[0] * std::pow(normSq, -0.5);
// cout << "<Psi,Psi>=" << dot(Psi, Psi, DIR) << endl;
#ifdef XPED_USE_MPI
    SPDLOG_CRITICAL("Calling MPI_Finalize()");
    // volatile int i = 0;
    // char hostname[256];
    // gethostname(hostname, sizeof(hostname));
    // printf("PID %d on %s ready for attach\n", getpid(), hostname);
    // fflush(stdout);
    // while(0 == i) sleep(5);
    MPI_Finalize();
#endif
}
