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

#ifdef MKL_ILP64
#    pragma message("Xped is using the intel math kernel library (MKL)")
#endif

#ifdef _OPENMP
#    pragma message("Xped is using OpenMP parallelization")
#    include "omp.h"
#endif

#include "Stopwatch.h"

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "ArgParser.h"

#include "Util/Mpi.hpp"

#include "Util/Macros.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    pragma message("Xped is using LRU cache for the output of FusionTree manipulations.")
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Interfaces/PlainInterface.hpp"

#include "Core/Qbasis.hpp"
#include "Symmetry/SU2.hpp"
#include "Symmetry/U0.hpp"
#include "Symmetry/U1.hpp"
#include "Symmetry/kind_dummies.hpp"

#include "Core/AdjointOp.hpp"
#include "Core/Xped.hpp"
#include "MPS/Mps.hpp"
#include "MPS/MpsAlgebra.hpp"

int main(int argc, char* argv[])
{
    std::ios::sync_with_stdio(true);

    ArgParser args(argc, argv);

    spdlog::set_level(spdlog::level::info);
#ifdef XPED_USE_OPENMPI
    MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &xped_rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &xped_np);
    // CTF::World world(argc, argv);
    util::mpi::XpedWorld world(argc.argv);
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
#else
    util::mpi::XpedWorld world;
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log.txt");
#endif

    my_logger->sinks()[0]->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [process %P] %v");
    if(world.rank == 0) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);
        console_sink->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [process %P] %v");
        my_logger->sinks().push_back(console_sink);
    }

    my_logger->info("Number of MPI processes: {}", world.np);
    my_logger->info("I am process number #={}", world.rank);

    // Xped<double, 2, 2, Symmetry_> t({{B, C}}, {{B, C}}, World);
    // t.setRandom();
    // std::cout << t.print() << std::endl;

    // std::cout << t << std::endl;
    // spdlog::get("info")->info("Tensor: \n {}", t);
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

    typedef Sym::SU2<Sym::SpinSU2> Symmetry;
    // typedef Sym::U1<Sym::SpinU1> Symmetry;
    // typedef Sym::U0 Symmetry;
    typedef Symmetry::qType qType;
    auto L = args.get<std::size_t>("L", 10);
    auto D = args.get<int>("D", 1);
    auto Minit = args.get<std::size_t>("Minit", 10);
    auto Qinit = args.get<std::size_t>("Qinit", 10);
    auto reps = args.get<std::size_t>("reps", 10);
    auto DIR = static_cast<DMRG::DIRECTION>(args.get<int>("DIR", 0));
    auto NORM = args.get<bool>("NORM", true);
    auto SWEEP = args.get<bool>("SWEEP", true);
    auto INFO = args.get<bool>("INFO", true);

    qType Qtot = {D};
    Qbasis<Symmetry, 1> qloc_;
    // qloc_.push_back({}, 2);
    qloc_.push_back({2}, 1);
    qloc_.push_back({3}, 1);
    qloc_.push_back({4}, 1);
    // qloc_.push_back({-2}, 1);
    std::vector<Qbasis<Symmetry, 1>> qloc(L, qloc_);
    // Qbasis<Symmetry, 1> in;
    // in.setRandom(Minit);
    // std::cout << in << std::endl;
    // auto out = in.combine(in).forgetHistory();
    // Xped<double, 2, 1, Symmetry> T({{in, in}}, {{out}});
    // T.setRandom();
    // auto F = T.permute<0, 2, 0, 1>();
    // std::cout << T.print(true) << std::endl;
    // auto X = T * T.adjoint();
    // std::cout << X.print(true) << std::endl;

    // auto Y = T.adjoint() * T;
    // std::cout << Y.print(true) << std::endl;
    Stopwatch<> construct;
    Mps<double, Symmetry> Psi(L, qloc, Qtot, Minit, Qinit);
    my_logger->critical(construct.info("Time for constructor"));

    if(INFO) {
        for(size_t l = 0; l <= L; l++) {
            std::stringstream ss;
            ss << Psi.auxBasis(l);
            my_logger->info(ss.str());
        }
    }
    if(NORM) {
        Stopwatch<> norm;
        for(std::size_t i = 0; i < reps; i++) {
            double normSq = dot(Psi, Psi, DIR);
            my_logger->info("<Psi|Psi>= {:03.2f}", normSq);
        }
        my_logger->critical(norm.info("Time for norm"));
    }
    if(SWEEP) {
        Stopwatch<> Sweep;
        if(DIR == DMRG::DIRECTION::RIGHT) {
            for(std::size_t l = 0; l < L; l++) { Psi.rightSweepStep(l, DMRG::BROOM::SVD); }
        } else {
            for(std::size_t l = L - 1; l > 0; l--) { Psi.leftSweepStep(l, DMRG::BROOM::SVD); }
            Psi.leftSweepStep(0, DMRG::BROOM::SVD);
        }
        my_logger->critical(Sweep.info("Time for sweep"));
    }

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
#ifdef XPED_USE_OPENMPI
    my_logger->critical("Calling MPI_Finalize()");
    // volatile int i = 0;
    // char hostname[256];
    // gethostname(hostname, sizeof(hostname));
    // printf("PID %d on %s ready for attach\n", getpid(), hostname);
    // fflush(stdout);
    // while(0 == i) sleep(5);
    MPI_Finalize();
#endif
}
