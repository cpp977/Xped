#include <cstddef>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "toml.hpp"

#ifdef _OPENMP
#    include "omp.h"
#endif

#include "stan/math/rev.hpp"

#include "ceres/first_order_function.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_solver.h"

#include "Xped/Util/Permutations.hpp"

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/TomlHelpers.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100000)
#endif

#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/MPS/Mps.hpp"
#include "Xped/MPS/MpsAlgebra.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/Models/Heisenberg.hpp"
#include "Xped/PEPS/Models/Hubbard.hpp"
#include "Xped/PEPS/Models/KondoNecklace.hpp"
#include "Xped/PEPS/iPEPS.hpp"

#include "Xped/AD/ADTensor.hpp"

#include "Xped/Util/Stopwatch.hpp"

#include "TOOLS/ArgParser.h"

#include "Xped/PEPS/iPEPSSolverAD.hpp"

int main(int argc, char* argv[])
{
    {
#ifdef XPED_USE_MPI
        MPI_Init(&argc, &argv);
        Xped::mpi::XpedWorld world(argc, argv);
        auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
#else
        Xped::mpi::XpedWorld world;
        auto my_logger = spdlog::basic_logger_mt("info", "logs/log.txt");
#endif
        std::ios::sync_with_stdio(true);

        ArgParser args(argc, argv);

        my_logger->sinks()[0]->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
        if(world.rank == 0) {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("%v");
            my_logger->sinks().push_back(console_sink);
        }
        spdlog::set_default_logger(my_logger);

        SPDLOG_INFO("Number of MPI processes: {}", world.np);
        SPDLOG_INFO("I am process number #={}", world.rank);
        SPDLOG_INFO("Number of MPI processes: {}", world.np);

        typedef double Scalar;
        // using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
        // using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
        // using Symmetry =
        //     Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
        // typedef Xped::Sym::SU2<Xped::Sym::SpinSU2> Symmetry;
        // typedef Xped::Sym::U1<Xped::Sym::SpinU1> Symmetry;
        // typedef Xped::Sym::ZN<Xped::Sym::SpinU1, 36, double> Symmetry;
        typedef Xped::Sym::U0<double> Symmetry;

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
            c = Xped::UnitCell(toml::get<std::vector<std::vector<std::string>>>(toml::find(data.at("ipeps"), "cell")));
        }

        Xped::TMatrix<Symmetry::qType> charges(c.pattern);
        charges.setConstant(Symmetry::qvacuum());
        if(data.at("ipeps").contains("charges")) {
            auto int_ch = toml::get<std::vector<std::vector<int>>>(toml::find(data.at("ipeps"), "charges"));
            for(int x = 0; x < c.Lx; ++x) {
                for(int y = 0; y < c.Ly; ++y) { charges(x, y) = {int_ch[x][y]}; }
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
        Psi->setRandom();

        Xped::Opts::Optim o_opts = Xped::Opts::optim_from_toml(data.at("optim"));
        Xped::Opts::CTM c_opts = Xped::Opts::ctm_from_toml(data.at("ctm"));
        constexpr Xped::Opts::CTMCheckpoint cp_opts{.MOVE = true, .CORNER = true};
        constexpr std::size_t TRank = 2;
        Xped::iPEPSSolverAD<Scalar, Symmetry, cp_opts, TRank> Jack(o_opts, c_opts, Psi, *ham);

        // Xped::FermionBase<Symmetry> F(1);
        // Xped::OneSiteObservable<Symmetry> n(c.pattern);
        // for(auto& t : n.data) { t = F.n().data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> nup(c.pattern);
        // for(auto& t : nup.data) { t = F.n(Xped::SPIN_INDEX::UP).data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> ndn(c.pattern);
        // for(auto& t : ndn.data) { t = F.n(Xped::SPIN_INDEX::DN).data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> d(c.pattern);
        // for(auto& t : d.data) { t = F.d().data.template trim<2>(); }

        // Xped::SpinBase<Symmetry> B(1, 2);
        // Xped::TwoSiteObservable<Symmetry> SzSz(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        // for(auto& t : SzSz.data_h) { t = Xped::tprod(B.Sz(), B.Sz()); }
        // for(auto& t : SzSz.data_v) { t = Xped::tprod(B.Sz(), B.Sz()); }
        // for(auto& t : SzSz.data_d1) { t = Xped::tprod(B.Sz(), B.Sz()); }
        // for(auto& t : SzSz.data_d2) { t = Xped::tprod(B.Sz(), B.Sz()); }

        // Xped::TwoSiteObservable<Symmetry> SpSm(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        // for(auto& t : SpSm.data_h) { t = Xped::tprod(B.Sp(), B.Sm()); }
        // for(auto& t : SpSm.data_v) { t = Xped::tprod(B.Sp(), B.Sm()); }
        // for(auto& t : SpSm.data_d1) { t = Xped::tprod(B.Sp(), B.Sm()); }
        // for(auto& t : SpSm.data_d2) { t = Xped::tprod(B.Sp(), B.Sm()); }

        // Xped::TwoSiteObservable<Symmetry> SmSp(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        // for(auto& t : SmSp.data_h) { t = Xped::tprod(B.Sm(), B.Sp()); }
        // for(auto& t : SmSp.data_v) { t = Xped::tprod(B.Sm(), B.Sp()); }
        // for(auto& t : SmSp.data_d1) { t = Xped::tprod(B.Sm(), B.Sp()); }
        // for(auto& t : SmSp.data_d2) { t = Xped::tprod(B.Sm(), B.Sp()); }
        // Xped::OneSiteObservable<Symmetry> Sz(c.pattern);
        // for(auto& t : Sz.data) { t = B.Sz().data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> Sx(c.pattern);
        // for(auto& t : Sx.data) { t = B.Sx().data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> sz(c.pattern);
        // for(auto& t : sz.data) { t = B.Sz(1).data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> Szsz(c.pattern);
        // for(auto& t : Szsz.data) { t = (B.Sz(0) * B.Sz(1)).data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> Spsm(c.pattern);
        // for(auto& t : Spsm.data) { t = (B.Sp(0) * B.Sm(1)).data.template trim<2>(); }
        // Xped::OneSiteObservable<Symmetry> Smsp(c.pattern);
        // for(auto& t : Smsp.data) { t = (B.Sm(0) * B.Sp(1)).data.template trim<2>(); }

        // Xped::TwoSiteObservable<Symmetry> SS(c.pattern, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2);
        // for(auto& t : SS.data_h) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // for(auto& t : SS.data_v) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // for(auto& t : SS.data_d1) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // for(auto& t : SS.data_d2) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
        // Xped::OneSiteObservable<Symmetry> Ssq(c.pattern);
        // for(auto& t : Ssq.data) {
        //     t = std::sqrt(3.) * (Xped::SiteOperator<Scalar, Symmetry>::prod(B.Sdag(0), B.S(0), Symmetry::qvacuum())).data.template trim<2>();
        // }
        // Jack.callback = [Sz, Sx, SzSz, SpSm, SmSp](XPED_CONST Xped::CTM<Scalar, Symmetry, TRank>& env, std::size_t i) mutable {
        //     fmt::print("  Callback at iteration {}\n", i);
        //     auto o_Sz = avg(env, Sz);
        //     auto o_Sx = avg(env, Sx);
        //     for(auto i = 0ul; i < o_Sz.size(); ++i) {
        //         fmt::print("  Sz={}, Sx={}, |S|={}\n", o_Sz[i], o_Sx[i], std::sqrt(o_Sz[i] * o_Sz[i] + o_Sx[i] * o_Sx[i]));
        //         // fmt::print("Sz={}\n", o_Sz[i]);
        //     }
        //     // auto o_Ssq = avg(env, Ssq);
        //     // for(const auto d : o_Ssq) { fmt::print("S²={}\n", d); }
        //     // auto [o_SS_h, o_SS_v, o_SS_d1, o_SS_d2] = avg(env, SS);
        //     // for(auto i = 0ul; i < o_SS_h.size(); ++i) { fmt::print("SS_h={}, SS_v={}\n", o_SS_h[i], o_SS_v[i]); }
        //     // for(auto i = 0ul; i < o_SS_d1.size(); ++i) { fmt::print("SS_d1={}, SS_d2={}\n", o_SS_d1[i], o_SS_d2[i]); }
        //     auto [o_SzSz_h, o_SzSz_v, o_SzSz_d1, o_SzSz_d2] = avg(env, SzSz);
        //     auto [o_SpSm_h, o_SpSm_v, o_SpSm_d1, o_SpSm_d2] = avg(env, SpSm);
        //     auto [o_SmSp_h, o_SmSp_v, o_SmSp_d1, o_SmSp_d2] = avg(env, SmSp);
        //     for(auto i = 0ul; i < o_SzSz_h.size(); ++i) { fmt::print("  SS_h={}\n", o_SzSz_h[i] + 0.5 * (o_SpSm_h[i] + o_SmSp_h[i])); }
        //     for(auto i = 0ul; i < o_SzSz_v.size(); ++i) { fmt::print("  SS_v={}\n", o_SzSz_v[i] + 0.5 * (o_SpSm_v[i] + o_SmSp_v[i])); }
        //     for(auto i = 0ul; i < o_SzSz_d1.size(); ++i) { fmt::print("  SS_d1={}\n", o_SzSz_d1[i] + 0.5 * (o_SpSm_d1[i] + o_SmSp_d1[i])); }
        //     for(auto i = 0ul; i < o_SzSz_d2.size(); ++i) { fmt::print("  SS_d2={}\n", o_SzSz_d2[i] + 0.5 * (o_SpSm_d2[i] + o_SmSp_d2[i])); }
        //     // auto o_Spsm = avg(env, Spsm);
        //     // for(const auto d : o_Spsm) { fmt::print("Spsm={}\n", d); }
        //     //     auto o_Smsp = avg(env, Smsp);
        //     //     for(const auto d : o_Smsp) { fmt::print("Smsp={}\n", d); }
        // };
        // Jack.callback = [n, nup, ndn, d](XPED_CONST Xped::CTM<Scalar, Symmetry, TRank>& env, std::size_t i) mutable {
        //     fmt::print("Callback at iteration {}\n", i);
        //     auto o_n = avg(env, n);
        //     auto o_nup = avg(env, nup);
        //     auto o_ndn = avg(env, ndn);
        //     auto o_d = avg(env, d);
        //     for(auto i = 0ul; i < o_n.size(); ++i) { fmt::print("n={}, nup={}, ndn={}, d={}\n", o_n[i], o_nup[i], o_ndn[i], o_d[i]); }
        // };
        // Jack.callback = ham.getObservables(std::filesystem::path("obs/obs.h5"));
        Jack.solve<double>();
#ifdef XPED_CACHE_PERMUTE_OUTPUT
        std::cout << "total hits=" << tree_cache</*shift*/ 0, /*Rank*/ 4, /*CoRank*/ 3, Symmetry>.cache.stats().total_hits()
                  << endl; // Hits for any key
        std::cout << "total misses=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
        std::cout << "hit rate=" << tree_cache<0, 4, 3, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif

#ifdef XPED_USE_MPI
        MPI_Finalize();
#endif
    }
}
