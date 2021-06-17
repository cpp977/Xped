#ifndef XPED_MPI_H_
#define XPED_MPI_H_

#ifdef XPED_USE_OPENMPI

#    include "mpi.h"
#    include "yas/serialize.hpp"

namespace util::mpi {
template <typename T>
void broadcast(T&& t, int process_rank, int root_process = 0, MPI_Comm comm = MPI_COMM_WORLD)
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

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(process_rank != root_process) { plain_buf = (char*)malloc(sizeof(char) * size); }

    MPI_Bcast(plain_buf, size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if(process_rank != root_process) {
        buf = yas::shared_buffer(plain_buf, size);
        yas::load<flags>(buf, t);
    }
}
} // namespace util::mpi
#endif
#endif
