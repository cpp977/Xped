#include "Xped/Util/Logging.hpp"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace Xped::Log {

inline void init_logging(const mpi::XpedWorld& world, const std::string& name)
{
    auto my_logger = world.np > 1 ? spdlog::basic_logger_mt("xlog", "logs/" + name + "_" + std::to_string(world.rank) + ".txt")
                                  : spdlog::basic_logger_mt("xlog", "logs/" + name + ".txt");
    my_logger->sinks()[0]->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
    my_logger->sinks()[0]->set_level(spdlog::level::trace);
    if(world.rank == 0) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%^---%l---%$] %v");
        console_sink->set_level(spdlog::level::trace);
        my_logger->sinks().push_back(console_sink);
    }
    my_logger->set_level(spdlog::level::trace);
    spdlog::set_default_logger(my_logger);
    spdlog::flush_every(std::chrono::seconds(3));
}

} // namespace Xped::Log

#if __has_include("Logging.gen.cpp")
#    include "Logging.gen.cpp"
#endif
