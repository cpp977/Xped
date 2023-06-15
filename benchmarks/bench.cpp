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

#include "toml.hpp"

#ifdef _OPENMP
#    include "omp.h"
const int XPED_MAX_THREADS = omp_get_max_threads();
#else
const int XPED_MAX_THREADS = 1;
#endif

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

#include "Xped/Util/Logging.hpp"
#include "Xped/Util/TomlHelpers.hpp"

namespace Xped::Opts {
struct Bench
{
    std::size_t L = 10;
    int D = 1;
    std::size_t Minit = 10;
    std::size_t Qinit = 10;
    std::size_t reps = 10;
    Xped::DMRG::DIRECTION DIR = Xped::DMRG::DIRECTION::LEFT;
    bool NORM = true;
    bool SWEEP = true;
    bool INFO = true;
    std::filesystem::path outfile = std::filesystem::current_path() / "out.csv";
};

inline Bench bench_from_toml(const toml::value& t)
{
    Bench res{};
    if(t.contains("out_file")) {
        std::filesystem::path tmp_of(static_cast<std::string>(t.at("outfile").as_string()));
        if(tmp_of.is_relative()) {
            res.outfile = std::filesystem::current_path() / tmp_of;
        } else {
            res.outfile = tmp_of;
        }
    }
    res.NORM = t.contains("NORM") ? t.at("NORM").as_boolean() : res.NORM;
    res.SWEEP = t.contains("SWEEP") ? t.at("SWEEP").as_boolean() : res.SWEEP;
    res.INFO = t.contains("INFO") ? t.at("INFO").as_boolean() : res.INFO;
    if(t.contains("DIR")) { res.DIR = Xped::util::enum_from_toml<DMRG::DIRECTION>(t.at("DIR")); }
    res.L = t.contains("L") ? (t.at("L").as_integer()) : res.L;
    res.D = t.contains("D") ? (t.at("D").as_integer()) : res.D;
    res.Minit = t.contains("Minit") ? (t.at("Minit").as_integer()) : res.Minit;
    res.Qinit = t.contains("Qinit") ? (t.at("Qinit").as_integer()) : res.Qinit;
    res.reps = t.contains("reps") ? (t.at("reps").as_integer()) : res.reps;
    return res;
}

} // namespace Xped::Opts

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

        std::string config_file = argc > 1 ? argv[1] : "config.toml";
        toml::value data;
        try {
            data = toml::parse(config_file);
            // std::cout << data << "\n";
        } catch(const toml::syntax_error& err) {
            std::cerr << "Parsing failed:\n" << err.what() << "\n";
            return 1;
        }

        Xped::Log::init_logging(world, "bench.txt");
        Xped::Log::globalLevel = Xped::Verbosity::DEBUG;
        if(data.at("Global").contains("LogLevel")) {
            Xped::Log::globalLevel = Xped::util::enum_from_toml<Xped::Verbosity>(data.at("Global").at("LogLevel"));
        }

        Xped::Log::debug("Number of MPI processes: {}", world.np);
        Xped::Log::debug("I am process number #={}", world.rank);

        Xped::Log::debug("Number of MPI processes: {}", world.np);

        typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        // typedef Sym::U1<Sym::SpinU1> Symmetry;
        // typedef Sym::U0 Symmetry;
        typedef Symmetry::qType qType;

        Xped::Opts::Bench opts = Xped::Opts::bench_from_toml(data.at("benchmark"));

        qType Qtot = {opts.D};
        Xped::Qbasis<Symmetry, 1> qloc_;
        // qloc_.push_back({}, 2);
        qloc_.push_back({2}, 1);
        // qloc_.push_back({3}, 1);
        // qloc_.push_back({4}, 1);
        // qloc_.push_back({-2}, 1);
        std::vector<Xped::Qbasis<Symmetry, 1>> qloc(opts.L, qloc_);

        Xped::util::Stopwatch<> construct;
        Xped::Mps<double, Symmetry> Psi(opts.L, qloc, Qtot, opts.Minit, opts.Qinit);
        Xped::Log::debug(construct.info("Time for constructor"));

        XPED_MPI_BARRIER(world.comm)

        Eigen::ArrayXd norm_times;
        norm_times.resize(opts.reps);
        norm_times.setZero();
        if(opts.INFO) {
            for(size_t l = 0; l <= opts.L; l++) { Xped::Log::debug("l={} \n {}", l, Psi.auxBasis(l).info()); }
        }
        if(opts.NORM) {
            norm_times.resize(opts.reps);
            for(std::size_t i = 0; i < opts.reps; i++) {
                Xped::util::Stopwatch<> norm;
                double normsq __attribute__((unused)) = dot(Psi, Psi, opts.DIR);
                norm_times(i) = norm.seconds();
                Xped::Log::on_exit("<Psi|Psi>= {:03.2e}, t={}", normsq, norm_times(i));
            }
            Xped::Log::on_exit("Time for norm: {}", norm_times.sum());
        }
        if(opts.SWEEP) {
            Xped::util::Stopwatch<> Sweep;
            if(opts.DIR == Xped::DMRG::DIRECTION::RIGHT) {
                for(std::size_t l = 0; l < opts.L; l++) {
                    Xped::Log::debug("l={}", l);
                    Psi.rightSweepStep(l, Xped::DMRG::BROOM::SVD);
                }
            } else {
                for(std::size_t l = opts.L - 1; l > 0; l--) {
                    Xped::Log::debug("l={}", l);
                    Psi.leftSweepStep(l, Xped::DMRG::BROOM::SVD);
                }
                Psi.leftSweepStep(0, Xped::DMRG::BROOM::SVD);
            }
            Xped::Log::debug(Sweep.info("Time for sweep"));
        }

#ifdef XPED_CACHE_PERMUTE_OUTPUT
        std::cout << "total hits=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
        std::cout << "total misses=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
        std::cout << "hit rate=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

        XPED_MPI_BARRIER(world.comm)

        if(world.rank == 0 and opts.NORM) {
            std::ofstream f;
            if(std::filesystem::exists(opts.outfile)) {
                f = std::ofstream(opts.outfile, std::ios::app);
            } else {
                f = std::ofstream(opts.outfile, std::ios::out);
                f << "Compiler,BLAS,#Processes,#Threads / process,Lib,Algorithm,Nq,M,L,Repetition,Total t [s],Mean t [s],Min t [s],Max t [s]\n";
            }
            f << XPED_COMPILER_STR << "," << XPED_BLAS_STR << "," << world.np << "," << XPED_MAX_THREADS << "," << XPED_DEFAULT_TENSORLIB{}.name()
              << ",norm," << std::to_string(opts.Qinit) << "," << std::to_string(opts.Minit) << "," << std::to_string(opts.L) << ","
              << std::to_string(opts.reps) << "," << std::to_string(norm_times.sum()) << "," << std::to_string(norm_times.mean()) << ","
              << std::to_string(norm_times.minCoeff()) << "," << std::to_string(norm_times.maxCoeff()) << "\n";
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
                            std::to_string(opts.Qinit),
                            std::to_string(opts.Minit),
                            std::to_string(opts.L),
                            std::to_string(opts.reps),
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
    Xped::Log::debug("Calling MPI_Finalize()");
    err = MPI_Finalize();
    Xped::Log::debug("Err={}", err);
#endif
}
