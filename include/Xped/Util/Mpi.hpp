#ifndef XPED_MPI_H_
#define XPED_MPI_H_

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#ifdef XPED_USE_MPI
#    include "ctf.hpp"
namespace Xped::mpi {

template <typename T>
struct TrivialDeleter
{
    void operator()(T*) {}
};

typedef CTF::World XpedWorld;
inline XpedWorld& getUniverse() { return CTF::get_universe(); }

template <typename T>
void broadcast(T&& t, int process_rank, int root_process = 0, XpedWorld& world = getUniverse());

} // namespace Xped::mpi
#else
namespace Xped::mpi {

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

    bool operator==(const XpedWorld& other) const { return comm == other.comm; }
    // auto operator<=>(const XpedWorld&) const = default;

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("x_world", ("rank", rank), ("np", np), ("comm", comm));
    }
};
inline XpedWorld universe{};
inline XpedWorld& getUniverse() { return universe; };

template <typename T>
void broadcast(T&&, int, int, XpedWorld&);

} // namespace Xped::mpi

#endif

#ifndef XPED_COMPILED_LIB
#    include "Util/Mpi.cpp"
#endif
#endif
