#ifdef XPED_USE_MPI
#    include "mpi.h"
#endif

#include "spdlog/spdlog.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

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
    std::cout << "Hello, I am in MPI mode" << std::endl;
    MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &xped_rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &xped_np);
    CTF::World world(MPI_COMM_WORLD, argc, argv);

    // auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/log_" + to_string(World.rank) + ".txt", true);
    // file_sink->set_level(spdlog::level::trace);
    // auto my_logger = std::make_shared<spdlog::logger>("info", {file_sink});
    // spdlog::register_logger(std::make_shared(my_logger));
    spdlog::set_level(spdlog::level::critical);
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
    my_logger->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [process %P] %v");
    if(world.rank == 0) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::critical);
        console_sink->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [process %P] %v");
        my_logger->sinks().push_back(console_sink);
    }

    my_logger->warn("Driver: Number of MPI processes: {}", world.np);
    my_logger->warn("Driver: I am process #={}", world.rank);
    // my_logger->warn("Driver: World #={}", world.comm);
    MPI_Barrier(world.comm);

    doctest::Context ctx;
    ctx.setOption("reporters", "MpiConsoleReporter");
    ctx.setOption("reporters", "MpiFileReporter");
    ctx.setOption("force-colors", true);
    ctx.applyCommandLine(argc, argv);
#else
    spdlog::set_level(spdlog::level::info);
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log.txt");
    my_logger->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [process %P] %v");
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::critical);
    console_sink->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [process %P] %v");
    my_logger->sinks().push_back(console_sink);

    doctest::Context ctx;
    ctx.setOption("force-colors", true);
    ctx.applyCommandLine(argc, argv);
#endif
    int test_result = ctx.run();

#ifdef XPED_USE_MPI
    MPI_Finalize();
#endif

    return test_result;
}
