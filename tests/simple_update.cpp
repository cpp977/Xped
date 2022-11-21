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

#include "Xped/Util/Logging.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100000)
#endif

#include "Xped/PEPS/CTMSolver.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/PEPS/SimpleUpdate.hpp"
#include "Xped/PEPS/TimePropagator.hpp"
#include "Xped/PEPS/iPEPS.hpp"
#include "Xped/PEPS/iPEPSSolverImag.hpp"

#include "Xped/PEPS/Models/Heisenberg.hpp"
#include "Xped/PEPS/Models/Hubbard.hpp"
#include "Xped/PEPS/Models/KondoNecklace.hpp"

#include "TOOLS/ArgParser.h"

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

        Xped::Log::init_logging(world, "log");

        ArgParser args(argc, argv);

        Xped::Log::globalLevel = args.get<Xped::Verbosity>("verb", Xped::Verbosity::DEBUG);

        typedef double Scalar;
        // using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
        // using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
        // using Symmetry =
        //     Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
        using Symmetry = Xped::Sym::Combined<Xped::Sym::U1<Xped::Sym::SpinU1>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;
        // typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        // typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        // typedef Xped::Sym::ZN<Xped::Sym::SpinU1, 36, double> Symmetry;
        // typedef Xped::Sym::U0<double> Symmetry;

        std::unique_ptr<Xped::TwoSiteObservable<Symmetry>> ham;

        auto config_file = args.get<std::string>("config_file", "config.toml");
        toml::value data;
        try {
            data = toml::parse(config_file);
            // std::cout << data << "\n";
        } catch(const toml::syntax_error& err) {
            std::cerr << "Parsing failed:\n" << err.what() << "\n";
            return 1;
        }

        Xped::UnitCell c;
        if(data.at("ipeps").contains("cell")) {
            c = Xped::UnitCell(toml::get<std::vector<std::vector<std::size_t>>>(toml::find(data.at("ipeps"), "cell")));
        }

        Xped::TMatrix<Symmetry::qType> charges(c.pattern);
        charges.setConstant(Symmetry::qvacuum());
        if(data.at("ipeps").contains("charges")) {
            auto int_ch = toml::get<std::vector<std::vector<std::vector<int>>>>(toml::find(data.at("ipeps"), "charges"));
            for(int x = 0; x < c.Lx; ++x) {
                for(int y = 0; y < c.Ly; ++y) { charges(x, y) = int_ch[x][y]; }
            }
        }

        Xped::TMatrix<Xped::Qbasis<Symmetry, 1>> left_aux(c.pattern), top_aux(c.pattern), right_aux(c.pattern), bottom_aux(c.pattern);
        if(data.at("ipeps").at("aux_bases").contains("left_basis")) {
            auto left =
                toml::get<std::vector<std::vector<std::pair<std::vector<int>, int>>>>(toml::find(data.at("ipeps").at("aux_bases"), "left_basis"));
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) {
                for(const auto& [q, dim_q] : left[i]) { left_aux[i].push_back(q, dim_q); }
                left_aux[i].sort();
            }
        } else {
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) { left_aux[i].setRandom(toml::get<std::size_t>(toml::find(data.at("ipeps"), "D"))); }
        }

        if(data.at("ipeps").at("aux_bases").contains("top_basis")) {
            auto top =
                toml::get<std::vector<std::vector<std::pair<std::vector<int>, int>>>>(toml::find(data.at("ipeps").at("aux_bases"), "top_basis"));
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) {
                for(const auto& [q, dim_q] : top[i]) { top_aux[i].push_back(q, dim_q); }
                top_aux[i].sort();
            }
        } else {
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) { top_aux[i].setRandom(toml::get<std::size_t>(toml::find(data.at("ipeps"), "D"))); }
        }

        if(data.at("ipeps").at("aux_bases").contains("right_basis")) {
            auto right =
                toml::get<std::vector<std::vector<std::pair<std::vector<int>, int>>>>(toml::find(data.at("ipeps").at("aux_bases"), "right_basis"));
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) {
                for(const auto& [q, dim_q] : right[i]) { right_aux[i].push_back(q, dim_q); }
                right_aux[i].sort();
            }
        } else {
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) { right_aux[i].setRandom(toml::get<std::size_t>(toml::find(data.at("ipeps"), "D"))); }
        }

        if(data.at("ipeps").at("aux_bases").contains("bottom_basis")) {
            auto bottom =
                toml::get<std::vector<std::vector<std::pair<std::vector<int>, int>>>>(toml::find(data.at("ipeps").at("aux_bases"), "bottom_basis"));
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) {
                for(const auto& [q, dim_q] : bottom[i]) { bottom_aux[i].push_back(q, dim_q); }
                bottom_aux[i].sort();
            }
        } else {
            for(std::size_t i = 0; i < c.uniqueSize(); ++i) { bottom_aux[i].setRandom(toml::get<std::size_t>(toml::find(data.at("ipeps"), "D"))); }
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
        } else {
            throw std::invalid_argument("Specified model is not implemented.");
        }
        ham->setDefaultObs();

        Xped::TMatrix<Xped::Qbasis<Symmetry, 1>> phys_basis(c.pattern);
        phys_basis.setConstant(ham->data_h[0].uncoupledDomain()[0]);
        auto Psi = std::make_shared<Xped::iPEPS<double, Symmetry, false>>(c, left_aux, top_aux, right_aux, bottom_aux, phys_basis, charges);
        // Psi->loadFromMatlab(std::filesystem::path("/home/user/matlab-tmp/heisenberg_D2.mat"), "cpp");
        Psi->setRandom();
        // Psi->info();
        Xped::Opts::CTM ctm_opts = Xped::Opts::ctm_from_toml(data.at("ctm"));
        Xped::Opts::Imag imag_opts = Xped::Opts::imag_from_toml(data.at("imag"));

        Xped::iPEPSSolverImag<Scalar, Symmetry> Lucy(imag_opts, ctm_opts, Psi, *ham);
        Lucy.solve();
    }
#ifdef XPED_CACHE_PERMUTE_OUTPUT
    std::cout << "total hits=" << tree_cache</*shift*/ 0, /*Rank*/ 4, /*CoRank*/ 3, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "total misses=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "hit rate=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

#ifdef XPED_USE_MPI
    MPI_Finalize();
#endif
}
