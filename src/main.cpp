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

#include "TOOLS/Stopwatch.h"

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "TOOLS/ArgParser.h"

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
#include "Xped/PEPS/iPEPS.hpp"

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

    auto Minit = args.get<std::size_t>("Minit", 10);

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

    typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
    // typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
    // typedef Xped::Sym::U0<double> Symmetry;

    Xped::Qbasis<Symmetry, 1> in;
    // in.push_back({0}, 2);
    // in.push_back({-1}, 1);
    // in.push_back({+1}, 1);
    in.push_back({1}, 3);
    in.push_back({2}, 2);
    SPDLOG_CRITICAL("Auxiliary basis (dim={}):\n {}", in.fullDim(), in);
    // std::cout << in << std::endl;
    Xped::Qbasis<Symmetry, 1> phys;
    phys.push_back({2}, 1);
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
    Xped::Pattern p({{'a', 'b'}, {'c', 'd'}});
    SPDLOG_CRITICAL("Pattern:\n {}", p);

    Xped::iPEPS<double, Symmetry> Psi(Xped::UnitCell(p), in, phys);
    // Psi.info();
    std::size_t chi = args.get<std::size_t>("chi", 100);
    Xped::CTM<double, Symmetry> Jack(Xped::UnitCell(p), chi);
    Jack.init(Psi);
    Jack.solve(Psi);
    // Jack.info();
    // auto Y = T.adjoint() * T;
    // std::cout << Y.print(true) << std::endl;

#ifdef XPED_CACHE_PERMUTE_OUTPUT
    std::cout << "total hits=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "total misses=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "hit rate=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif
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
