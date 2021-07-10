#ifndef XPED_MPI_H_
#define XPED_MPI_H_

#ifdef XPED_USE_MPI

#    include "ctf.hpp"
#    include "mpi.h"
#    include "yas/serialize.hpp"

namespace util::mpi {

typedef CTF::World XpedWorld;
XpedWorld getUniverse() { return CTF::get_universe(); }

template <typename T>
void broadcast(T&& t, int process_rank, int root_process = 0, XpedWorld world = getUniverse())
{
    constexpr std::size_t flags = yas::mem /*IO type*/ | yas::binary; /*IO format*/
    yas::shared_buffer buf;
    if(process_rank == root_process) { buf = yas::save<flags>(t); }

    int size = 0;
    char* plain_buf = NULL;
    if(process_rank == root_process) {
        size = buf.size;

        plain_buf = (char*)malloc(sizeof(char) * size);
        plain_buf = buf.data.get();
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, world.comm);
    if(process_rank != root_process) { plain_buf = (char*)malloc(sizeof(char) * size); }

    MPI_Bcast(plain_buf, size, MPI_CHAR, 0, world.comm);

    if(process_rank != root_process) {
        buf = yas::shared_buffer(plain_buf, size);
        yas::load<flags>(buf, t);
    }
}
} // namespace util::mpi
#else
namespace util::mpi {
struct XpedWorld
{
    int rank = 0;
    int np = 1;
};
XpedWorld getUniverse()
{
    XpedWorld out;
    return out;
};

template <typename T>
void broadcast(T&&, int, int, XpedWorld)
{
    return;
}
} // namespace util::mpi

#endif
#endif
