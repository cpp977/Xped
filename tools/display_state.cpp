#include <cstddef>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "toml.hpp"

#ifdef _OPENMP
#    include "omp.h"
#endif

#include "stan/math/rev.hpp"

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/TomlHelpers.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
#endif

#include "Xped/Symmetry/CombSym.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"

#include "Xped/PEPS/iPEPS.hpp"
#include "Xped/Util/Logging.hpp"

int main(int argc, char* argv[])
{
    {
#ifdef XPED_USE_MPI
        MPI_Init(&argc, &argv);
        Xped::mpi::XpedWorld world(argc, argv);
#else
        Xped::mpi::XpedWorld world;
#endif
        // std::ios::sync_with_stdio(true);

        // using Scalar = std::complex<double>;
        using Scalar = double;
        // using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
        // using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>;
        // using Symmetry =
        // Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
        // using Symmetry = Xped::Sym::Combined<Xped::Sym::U1<Xped::Sym::SpinU1>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;
        using Symmetry = Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;
        // using Symmetry = Xped::Sym::Combined<Xped::Sym::ZN<Xped::Sym::SpinU1, 36>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;

        // typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        // typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        // typedef Xped::Sym::ZN<Xped::Sym::SpinU1, 36, double> Symmetry;
        // using Symmetry = Xped::Sym::U0<double>;

        std::string state_file = argc > 1 ? argv[1] : "state.psi";

        Xped::iPEPS<Scalar, Symmetry> Psi;
        constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
        Xped::Log::on_entry("Load iPEPS from native file {}.", state_file);

        try {
            yas::load<flags>(state_file.c_str(), Psi);
        } catch(const yas::serialization_exception& se) {
            fmt::print(
                "Error while deserializing file ({}) with initial wavefunction.\nThis might be because of incompatible symmetries between this simulation and the loaded wavefunction.",
                state_file);
            std::cout << std::flush;
            throw;
        } catch(const yas::io_exception& ie) {
            fmt::print("Error while loading file ({}) with initial wavefunction.\n", state_file);
            std::cout << std::flush;
            throw;
        } catch(const std::exception& e) {
            fmt::print("Unknown error while loading file ({}) with initial wavefunction.\n", state_file);
            std::cout << std::flush;
            throw;
        }
        std::cout << Psi.cell().pattern << std::endl;
        Psi.debug_info();
    }

#ifdef XPED_USE_MPI
    MPI_Finalize();
#endif
}
