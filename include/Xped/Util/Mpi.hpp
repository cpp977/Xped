#ifndef XPED_MPI_H_
#define XPED_MPI_H_

#ifdef XPED_USE_MPI
#    include "ctf.hpp"
namespace util::mpi {

template <typename T>
struct TrivialDeleter
{
    void operator()(T*) {}
};

typedef CTF::World XpedWorld;
inline XpedWorld& getUniverse() { return CTF::get_universe(); }

template <typename T>
void broadcast(T&& t, int process_rank, int root_process = 0, XpedWorld& world = getUniverse());

} // namespace util::mpi
#else
namespace util::mpi {

template <typename T>
struct TrivialDeleter
{
    void operator()(T*) {}
};

struct XpedWorld
{
    int rank = 0;
    int np = 1;
    int comm = 1000;
};
inline XpedWorld universe{};
inline XpedWorld& getUniverse() { return universe; };

template <typename T>
void broadcast(T&&, int, int, XpedWorld&);

} // namespace util::mpi

#endif

#ifndef XPED_COMPILED_LIB
#    include "src/Mpi.hpp"
#endif
#endif
