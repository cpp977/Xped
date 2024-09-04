#include "Xped/Util/Mpi.hpp"

#ifdef XPED_USE_MPI
#    include "mpi.h"
#    include "spdlog/spdlog.h"
#    include "yas/serialize.hpp"

#    include "Xped/Core/Qbasis.hpp"
#    include "Xped/Symmetry/SU2.hpp"
#    include "Xped/Symmetry/U0.hpp"
#    include "Xped/Symmetry/U1.hpp"
#    include "Xped/Util/Macros.hpp"
namespace Xped::mpi {

template <typename T>
void broadcast(T&& t, int process_rank, int root_process, XpedWorld& world)
{
    XPED_MPI_BARRIER(world.comm)
    SPDLOG_TRACE("Entering broadcast.");
    constexpr std::size_t flags = yas::mem /*IO type*/ | yas::binary; /*IO format*/
    yas::shared_buffer buf;
    if(process_rank == root_process) {
        buf = yas::save<flags>(t);
        SPDLOG_TRACE("Root set buf.");
    }

    int size = 0;
    char* plain_buf = nullptr;
    if(process_rank == root_process) {
        size = buf.size;

        plain_buf = (char*)malloc(sizeof(char) * size);
        std::memcpy(plain_buf, buf.data.get(), size);
        SPDLOG_TRACE("Root malloced plain buf.");
    }

    MPI_Bcast(&size, 1, MPI_INT, root_process, world.comm);
    SPDLOG_TRACE("Called MPI_Bcast for size.");
    if(process_rank != root_process) {
        plain_buf = (char*)malloc(sizeof(char) * size);
        SPDLOG_TRACE("Non root malloced plain buf.");
    }

    MPI_Bcast(plain_buf, size, MPI_CHAR, root_process, world.comm);
    SPDLOG_TRACE("Called MPI_Bcast for plain buf.");

    if(process_rank != root_process) {
        buf = yas::shared_buffer(plain_buf, size);
        SPDLOG_TRACE("Non root set buf.");
        yas::load<flags>(buf, t);
    }
    XPED_MPI_BARRIER(world.comm)
    SPDLOG_TRACE("Before free.");
    free(plain_buf);
    XPED_MPI_BARRIER(world.comm)
    SPDLOG_TRACE("After free.");
    XPED_MPI_BARRIER(world.comm)
    SPDLOG_TRACE("Leaving broadcast.");
}

} // namespace Xped::mpi
#else
#    include "Xped/Core/Qbasis.hpp"
#    include "Xped/Symmetry/SU2.hpp"
#    include "Xped/Symmetry/U0.hpp"
#    include "Xped/Symmetry/U1.hpp"

namespace Xped::mpi {

template <typename T>
inline void broadcast(T&&, int, int, XpedWorld&)
{
    return;
}

} // namespace Xped::mpi
#endif

#if __has_include("Mpi.gen.cpp")
#    include "Mpi.gen.cpp"
#endif
