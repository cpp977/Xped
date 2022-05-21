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
#include "Xped/Symmetry/ZN.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/MPS/Mps.hpp"
#include "Xped/MPS/MpsAlgebra.hpp"

#include "Xped/PEPS/Models/KondoNecklace.hpp"
// #include "Xped/PEPS/iPEPS.hpp"

#include "Xped/Util/Stopwatch.hpp"

#include "Xped/IO/Matlab.hpp"

#include "TOOLS/ArgParser.h"

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

    HighFive::File file("/home/user/matlab-tmp/math5.mat", HighFive::File::ReadOnly);
    auto py = file.getGroup("py");

    auto C1_ref = py.getDataSet("C1");
    std::vector<HighFive::Reference> C1;
    C1_ref.read(C1);
    auto g_C1_0 = C1[0].template dereference<HighFive::Group>(py);
    auto C = Xped::IO::loadMatlabTensor<double, 2, 0, Xped::Sym::ZN<Xped::Sym::ChargeU1, 36>, Xped::HeapPolicy>(g_C1_0, py);
    std::cout << C.squaredNorm() << std::endl;
    std::cout << (C.template contract<std::array{1, 2}, std::array{1, 2}, 0>(C.adjoint().eval())).trace() << std::endl;

    auto T1_ref = py.getDataSet("T1");
    std::vector<HighFive::Reference> T1;
    T1_ref.read(T1);
    auto g_T1_0 = T1[0].template dereference<HighFive::Group>(py);
    auto T = Xped::IO::loadMatlabTensor<double, 0, 4, Xped::Sym::ZN<Xped::Sym::ChargeU1, 36>, Xped::HeapPolicy>(g_T1_0, py);
    std::cout << T.squaredNorm() << std::endl;
    std::cout << (T.template contract<std::array{1, 2, 3, 4}, std::array{1, 2, 3, 4}, 0>(T.adjoint().eval())).trace() << std::endl;

    auto CT = C.template contract<std::array{1, -1}, std::array{1, -2, -3, -4}, 4>(T);

    auto H = Xped::IO::loadMatlabTensor<double, 2, 2, Xped::Sym::ZN<Xped::Sym::ChargeU1, 36>, Xped::HeapPolicy>(py.getGroup("H"), py);
    auto [D, U] = H.eigh();
    D.print(std::cout);
    std::cout << H.squaredNorm() << std::endl;

    auto Hcheck = Xped::KondoNecklace<Xped::Sym::U0<>>::twoSiteHamiltonian(0.68, 0.68, 1., 1., 0., 0.4);
    auto [Dcheck, Ucheck] = Hcheck.eigh();
    Dcheck.print(std::cout);
    std::cout << Hcheck.squaredNorm() << std::endl;

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
