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

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

#include "Xped/Util/Macros.hpp"

#include "Xped/Util/Logging.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 1000000)
#endif

#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "Xped/PEPS/CTMSolver.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/PEPS/SimpleUpdate.hpp"
#include "Xped/PEPS/TimePropagator.hpp"
#include "Xped/PEPS/iPEPS.hpp"
#include "Xped/PEPS/iPEPSSolverAD.hpp"
#include "Xped/PEPS/iPEPSSolverImag.hpp"

#include "Xped/PEPS/Models/Heisenberg.hpp"
#include "Xped/PEPS/Models/Hubbard.hpp"
#include "Xped/PEPS/Models/Kondo.hpp"
#include "Xped/PEPS/Models/KondoNecklace.hpp"

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
        // using Symmetry = Xped::Sym::ZN<Xped::Sym::SpinU1, 36>;
        // using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
        using Symmetry = Xped::Sym::Combined<Xped::Sym::ZN<Xped::Sym::SpinU1, 36>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;
        // using Symmetry = Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;

        // using Symmetry =
        // Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
        // using Symmetry = Xped::Sym::Combined<Xped::Sym::U1<Xped::Sym::SpinU1>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;
        // using Symmetry = Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
        // using Symmetry = Xped::Sym::Combined<Xped::Sym::ZN<Xped::Sym::SpinU1, 36>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;

        // typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        // typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        // typedef Xped::Sym::ZN<Xped::Sym::SpinU1, 36, double> Symmetry;
        // using Symmetry = Xped::Sym::U0<double>;

        std::unique_ptr<Xped::TwoSiteObservable<double, Symmetry>> ham;

        std::string config_file = argc > 1 ? argv[1] : "config.toml";

        toml::value data;
        try {
            data = toml::parse(config_file);
            // std::cout << data << "\n";
        } catch(const toml::syntax_error& err) {
            std::cerr << "Parsing failed:\n" << err.what() << "\n";
            return 1;
        }

        Xped::Pattern pat;
        Xped::UnitCell c;
        if(data.at("ipeps").contains("pattern")) {
            pat = Xped::Pattern(toml::get<std::vector<std::vector<std::size_t>>>(toml::find(data.at("ipeps"), "cell")));
            c = Xped::UnitCell(pat);
        } else if(data.at("ipeps").contains("cell")) {
            auto [Lx, Ly] = toml::get<std::pair<int, int>>(toml::find(data.at("ipeps"), "cell"));
            c = Xped::UnitCell(Lx, Ly);
        }

        Xped::TMatrix<Symmetry::qType> charges(c.pattern);
        charges.setConstant(Symmetry::qvacuum());
        if(data.at("ipeps").contains("charges")) {
            auto int_ch = toml::get<std::vector<std::vector<std::vector<int>>>>(toml::find(data.at("ipeps"), "charges"));
            for(int x = 0; x < c.Lx; ++x) {
                for(int y = 0; y < c.Ly; ++y) { charges(x, y) = int_ch[x][y]; }
            }
        }

        std::size_t D = toml::get_or<std::size_t>(toml::find(data.at("ipeps"), "D"), 2ul);

        Xped::TMatrix<Xped::Qbasis<Symmetry, 1>> left_aux(c.pattern), top_aux(c.pattern);
        if(data.at("ipeps").contains("aux_bases")) {
            auto left =
                toml::get<std::vector<std::vector<std::pair<std::vector<int>, int>>>>(toml::find(data.at("ipeps").at("aux_bases"), "left_basis"));
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) {
                for(const auto& [q, dim_q] : left[i]) { left_aux[i].push_back(q, dim_q); }
                left_aux[i].sort();
            }

            auto top =
                toml::get<std::vector<std::vector<std::pair<std::vector<int>, int>>>>(toml::find(data.at("ipeps").at("aux_bases"), "top_basis"));
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) {
                for(const auto& [q, dim_q] : top[i]) { top_aux[i].push_back(q, dim_q); }
                top_aux[i].sort();
            }
        }

        std::map<std::string, Xped::Param> params = Xped::util::params_from_toml(data.at("model").at("params"));

        std::vector<Xped::Opts::Bond> bs;
        for(const auto& elem : data.at("model").at("bonds").as_array()) {
            auto b = Xped::util::enum_from_toml<Xped::Opts::Bond>(elem);
            bs.push_back(b);
        }
        Xped::Opts::Bond bonds;
        if(bs.size() == 0) {
            bonds = Xped::Opts::Bond::V | Xped::Opts::Bond::H;
        } else {
            bonds = bs[0];
            for(std::size_t i = 1; i < bs.size(); ++i) { bonds = bonds | bs[i]; }
        }

        if(toml::find(data.at("model"), "name").as_string() == "Heisenberg") {
            ham = std::make_unique<Xped::Heisenberg<Symmetry>>(params, c.pattern, bonds);
        } else if(toml::find(data.at("model"), "name").as_string() == "KondoNecklace") {
            ham = std::make_unique<Xped::KondoNecklace<Symmetry>>(params, c.pattern, bonds);
        } else if(toml::find(data.at("model"), "name").as_string() == "Hubbard") {
            ham = std::make_unique<Xped::Hubbard<Symmetry>>(params, c.pattern, bonds);
        } else if(toml::find(data.at("model"), "name").as_string() == "Kondo") {
            ham = std::make_unique<Xped::Kondo<Symmetry>>(params, c.pattern, bonds);
        } else {
            throw std::invalid_argument("Specified model is not implemented.");
        }
        ham->setDefaultObs();

        Xped::Opts::CTM ctm_opts = Xped::Opts::ctm_from_toml(data.at("ctm"));
        Xped::Opts::Imag imag_opts = Xped::Opts::imag_from_toml(data.at("imag"));
        Xped::Opts::Optim o_opts = Xped::Opts::optim_from_toml(data.at("optim"));

        Xped::Log::init_logging(world, (o_opts.working_directory / o_opts.logging_directory).string() + "/" + ham->file_name() + ".txt");

        Xped::TMatrix<Xped::Qbasis<Symmetry, 1>> phys_basis(c.pattern);
        phys_basis.setConstant(ham->data_h[0].uncoupledDomain()[0]);
        auto Psi = std::make_shared<Xped::iPEPS<Scalar, Symmetry, false>>(c, D, left_aux, top_aux, phys_basis, charges);
        Psi->setRandom();

        auto Ds = imag_opts.Ds;
        auto chis = imag_opts.chis;

        constexpr Xped::Opts::CTMCheckpoint cp_opts{
            .GROW_ALL = true, .MOVE = true, .CORNER = true, .PROJECTORS = true, .RENORMALIZE = true, .RDM = true};
        constexpr std::size_t TRank = 2;

        for(auto iD = 0ul; auto D : Ds) {
            ctm_opts.chi = chis[iD][0];
            imag_opts.Ds = std::vector(1, D);
            imag_opts.chis = std::vector(1, std::vector(1, chis[iD][0]));
            if(iD > 0) {
                imag_opts.load = o_opts.working_directory.string() + "/" + ham->file_name() +
                                 fmt::format("_D={}_chi={}_seed={}_id={}.psi", Ds[iD - 1], chis[iD - 1][0], o_opts.seed, o_opts.id);
            }
            Xped::iPEPSSolverImag<Scalar, Symmetry> Lucy(imag_opts, ctm_opts, Psi, *ham);
            Lucy.solve();
            o_opts.load = (imag_opts.working_directory.string() + "/" + ham->file_name() + fmt::format("_D={}_id={}.psi", D, imag_opts.id));
            Xped::iPEPSSolverAD<Scalar, Symmetry, cp_opts, TRank> Jack(o_opts, ctm_opts, Psi, *ham);
            Jack.solve<double>();
            ++iD;
        }

#ifdef XPED_CACHE_PERMUTE_OUTPUT
        std::cout << "total hits=" << tree_cache</*shift*/ 0, /*Rank*/ 4, /*CoRank*/ 3, Symmetry>.cache.stats().total_hits()
                  << endl; // Hits for any key
        std::cout << "total misses=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
        std::cout << "hit rate=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif
    }

#ifdef XPED_USE_MPI
    MPI_Finalize();
#endif
}
