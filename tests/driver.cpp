#include "mpi.h"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

int main(int argc, char** argv)
{
    int xped_rank, xped_np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &xped_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &xped_np);
    std::cout << "Number of MPI processes: " << xped_np << std::endl;
    doctest::Context context;

    context.applyCommandLine(argc, argv);

    int res = context.run(); // run

    if(context.shouldExit()) // important - query flags (and --exit) rely on the user doing this
        return res; // propagate the result of the tests

    MPI_Finalize();

    return res;
}
