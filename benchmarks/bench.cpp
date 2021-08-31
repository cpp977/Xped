#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"

#ifdef MKL_ILP64
#    pragma message("Xped is using the intel math kernel library (MKL)")
#endif

#ifdef _OPENMP
#    pragma message("Xped is using OpenMP parallelization")
#    include "omp.h"
const int XPED_MAX_THREADS = omp_get_max_threads();
#else
const int XPED_MAX_THREADS = 1;
#endif

#include "TOOLS/ArgParser.h"
#include "TOOLS/Stopwatch.h"

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "Util/Macros.hpp"

#include "spdlog/spdlog.h"

#include "spdlog/cfg/argv.h" // for loading levels from argv
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "Util/Mpi.hpp"

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
    spdlog::cfg::load_argv_levels(argc, argv);
#ifdef XPED_USE_MPI
    MPI_Init(&argc, &argv);
    util::mpi::XpedWorld world(argc, argv);
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
#else
    util::mpi::XpedWorld world;
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log.txt");
#endif

    std::ios::sync_with_stdio(true);

    ArgParser args(argc, argv);

    // spdlog::set_level(spdlog::level::info);

    my_logger->sinks()[0]->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
    if(world.rank == 0) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        // console_sink->set_level(spdlog::level::warn);
        console_sink->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
        my_logger->sinks().push_back(console_sink);
    }
    spdlog::set_default_logger(my_logger);

    SPDLOG_INFO("Number of MPI processes: {}", world.np);
    SPDLOG_INFO("I am process number #={}", world.rank);

    SPDLOG_INFO("Number of MPI processes: {}", world.np);

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
    auto outfile = args.get<std::string>("outfile", "./out.csv");

    qType Qtot = {D};
    Qbasis<Symmetry, 1> qloc_;
    // qloc_.push_back({}, 2);
    qloc_.push_back({2}, 1);
    // qloc_.push_back({3}, 1);
    // qloc_.push_back({4}, 1);
    // qloc_.push_back({-2}, 1);
    std::vector<Qbasis<Symmetry, 1>> qloc(L, qloc_);

    Stopwatch<> construct;
    Mps<double, Symmetry> Psi(L, qloc, Qtot, Minit, Qinit);
    SPDLOG_CRITICAL(construct.info("Time for constructor"));

    XPED_MPI_BARRIER(world.comm)

    Eigen::ArrayXd norm_times;
    norm_times.resize(reps);
    norm_times.setZero();
    if(INFO) {
        for(size_t l = 0; l <= L; l++) { SPDLOG_INFO("l={} \n {}", l, Psi.auxBasis(l)); }
    }
    if(NORM) {
        norm_times.resize(reps);
        for(std::size_t i = 0; i < reps; i++) {
            Stopwatch<> norm;
            double normsq __attribute__((unused)) = dot(Psi, Psi, DIR);
            norm_times(i) = norm.time();
            SPDLOG_WARN("<Psi|Psi>= {:03.2e}", normsq);
        }
        SPDLOG_CRITICAL("Time for norm: {}", norm_times.sum());
    }
    if(SWEEP) {
        Stopwatch<> Sweep;
        if(DIR == DMRG::DIRECTION::RIGHT) {
            for(std::size_t l = 0; l < L; l++) {
                SPDLOG_WARN("l={}", l);
                Psi.rightSweepStep(l, DMRG::BROOM::SVD);
            }
        } else {
            for(std::size_t l = L - 1; l > 0; l--) { Psi.leftSweepStep(l, DMRG::BROOM::SVD); }
            Psi.leftSweepStep(0, DMRG::BROOM::SVD);
        }
        SPDLOG_CRITICAL(Sweep.info("Time for sweep"));
    }

#ifdef XPED_CACHE_PERMUTE_OUTPUT
    std::cout << "total hits=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "total misses=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "hit rate=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

    XPED_MPI_BARRIER(world.comm)

    if(world.rank == 0) {
        std::filesystem::path p(outfile);
        std::ofstream f;
        if(std::filesystem::exists(p)) {
            f = std::ofstream(p, std::ios::app);
        } else {
            f = std::ofstream(p, std::ios::out);
            f << "Compiler,BLAS,#Processes,#Threads / process,Lib,Algorithm,Nq,M,L,Repetition,Total t [s],Mean t [s],Min t [s],Max t [s]\n";
        }
        f << XPED_COMPILER_STR << "," << XPED_BLAS_STR << "," << world.np << "," << XPED_MAX_THREADS << "," << XPED_DEFAULT_TENSORLIB{}.name()
          << ",norm," << std::to_string(Qinit) << "," << std::to_string(Minit) << "," << std::to_string(L) << "," << std::to_string(reps) << ","
          << std::to_string(norm_times.sum()) << "," << std::to_string(norm_times.mean()) << "," << std::to_string(norm_times.minCoeff()) << ","
          << std::to_string(norm_times.maxCoeff()) << "\n";
        f.close();

        using Row_t = std::vector<variant<std::string, const char*, tabulate::Table>>;
        tabulate::Table t;
        t.add_row({"Compiler",
                   "BLAS",
                   "#Processes",
                   "#Threads / process",
                   "Lib",
                   "Algorithm",
                   "Nq",
                   "M",
                   "L",
                   "Repetitions",
                   "Total t [s]",
                   "Mean t [s]",
                   "Min t [s]",
                   "Max t [s]"});
        t.add_row(Row_t{XPED_COMPILER_STR,
                        XPED_BLAS_STR,
                        std::to_string(world.np),
                        std::to_string(XPED_MAX_THREADS),
                        XPED_DEFAULT_TENSORLIB{}.name(),
                        "norm",
                        std::to_string(Qinit),
                        std::to_string(Minit),
                        std::to_string(L),
                        std::to_string(reps),
                        std::to_string(norm_times.sum()),
                        std::to_string(norm_times.mean()),
                        std::to_string(norm_times.minCoeff()),
                        std::to_string(norm_times.maxCoeff())});
        t.format()
            .font_align(tabulate::FontAlign::center)
            .font_style({tabulate::FontStyle::bold})
            .border_top("-")
            .border_bottom("-")
            .border_left("|")
            .border_right("|")
            .corner(" ");
        t[0].format().padding_top(1).padding_bottom(1).font_style({tabulate::FontStyle::underline}).font_background_color(tabulate::Color::grey);

        std::cout << t << std::endl;
    }
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
