#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "tabulate/table.hpp"
#include "tabulate/tabulate.hpp"

#ifdef _OPENMP
#    include "omp.h"
const int XPED_MAX_THREADS = omp_get_max_threads();
#else
const int XPED_MAX_THREADS = 1;
#endif

#include "TOOLS/ArgParser.h"

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "Xped/Util/Macros.hpp"

#include "spdlog/spdlog.h"

#include "spdlog/cfg/argv.h" // for loading levels from argv
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

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

#include "Xped/Util/Stopwatch.hpp"

int main(int argc, char* argv[])
{
#ifdef XPED_USE_MPI
    int err = MPI_Init(&argc, &argv);
#endif
    {
        spdlog::cfg::load_argv_levels(argc, argv);
#ifdef XPED_USE_MPI
        Xped::mpi::XpedWorld world(argc, argv);
        auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
#else
        Xped::mpi::XpedWorld world;
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

        typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        // typedef Sym::U1<Sym::SpinU1> Symmetry;
        // typedef Sym::U0 Symmetry;
        typedef Symmetry::qType qType;
        auto L = args.get<std::size_t>("L", 10);
        auto D = args.get<int>("D", 1);
        auto Minit = args.get<std::size_t>("Minit", 10);
        auto Qinit = args.get<std::size_t>("Qinit", 10);
        auto reps = args.get<std::size_t>("reps", 10);
        auto DIR = static_cast<Xped::DMRG::DIRECTION>(args.get<int>("DIR", 0));
        auto NORM = args.get<bool>("NORM", true);
        auto SWEEP = args.get<bool>("SWEEP", true);
        auto INFO = args.get<bool>("INFO", true);
        auto outfile = args.get<std::string>("outfile", "./out.csv");

        qType Qtot = {D};
        Xped::Qbasis<Symmetry, 1> qloc_;
        // qloc_.push_back({}, 2);
        qloc_.push_back({2}, 1);
        // qloc_.push_back({3}, 1);
        // qloc_.push_back({4}, 1);
        // qloc_.push_back({-2}, 1);
        std::vector<Xped::Qbasis<Symmetry, 1>> qloc(L, qloc_);

        Stopwatch<> construct;
        Xped::Mps<double, Symmetry> Psi(L, qloc, Qtot, Minit, Qinit);
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
            if(DIR == Xped::DMRG::DIRECTION::RIGHT) {
                for(std::size_t l = 0; l < L; l++) {
                    SPDLOG_CRITICAL("l={}", l);
                    Psi.rightSweepStep(l, Xped::DMRG::BROOM::SVD);
                }
            } else {
                for(std::size_t l = L - 1; l > 0; l--) {
                    SPDLOG_CRITICAL("l={}", l);
                    Psi.leftSweepStep(l, Xped::DMRG::BROOM::SVD);
                }
                Psi.leftSweepStep(0, Xped::DMRG::BROOM::SVD);
            }
            SPDLOG_CRITICAL(Sweep.info("Time for sweep"));
        }

#ifdef XPED_CACHE_PERMUTE_OUTPUT
        std::cout << "total hits=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
        std::cout << "total misses=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
        std::cout << "hit rate=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

        XPED_MPI_BARRIER(world.comm)

        if(world.rank == 0 and NORM) {
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
        XPED_MPI_BARRIER(world.comm)
    }
#ifdef XPED_USE_MPI
    SPDLOG_CRITICAL("Calling MPI_Finalize()");
    err = MPI_Finalize();
    SPDLOG_CRITICAL("Err={}", err);
#endif
}
