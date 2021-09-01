#ifdef XPED_USE_MPI
#    include "mpi.h"
#endif

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN

#include "spdlog/spdlog.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#ifdef _OPENMP
#    include "omp.h"
#endif

#ifdef XPED_USE_MPI
#    include "doctest/extensions/doctest_mpi.h"
#endif

#ifdef XPED_USE_MPI
#    include "ctf.hpp"
#endif

int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(true);
#ifdef XPED_USE_MPI
    MPI_Init(&argc, &argv);
    CTF::World world(MPI_COMM_WORLD, argc, argv);

    auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
    my_logger->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
    if(world.rank == 0) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
        my_logger->sinks().push_back(console_sink);
    }
    spdlog::set_default_logger(my_logger);

    SPDLOG_CRITICAL("Driver: Number of MPI processes: {}", world.np);
    SPDLOG_CRITICAL("Driver: I am process #={}", world.rank);
    MPI_Barrier(world.comm);

    doctest::Context ctx;
    ctx.setOption("reporters", "MpiConsoleReporter");
    ctx.setOption("reporters", "MpiFileReporter");
#else
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log.txt");
    my_logger->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
    my_logger->sinks().push_back(console_sink);
    spdlog::set_default_logger(my_logger);

    doctest::Context ctx;
#endif
    ctx.setOption("force-colors", true);
    ctx.applyCommandLine(argc, argv);

    int test_result = ctx.run();

#ifdef XPED_USE_MPI
    MPI_Finalize();
#endif

    return test_result;
}
