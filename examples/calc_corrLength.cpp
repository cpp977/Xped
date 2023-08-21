#include "Xped/Util/Macros.hpp"

#include "Xped/Util/Logging.hpp"
#include "Xped/Util/Permutations.hpp"
#include "Xped/Util/Stopwatch.hpp"

#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/TomlHelpers.hpp"
#include "Xped/Util/YasHelpers.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100000)
#endif

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/CorrelationLength.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/PEPS/Models/Hamiltonian.hpp"

#include "Xped/PEPS/Models/Heisenberg.hpp"
#include "Xped/PEPS/Models/Hubbard.hpp"
#include "Xped/PEPS/Models/Kondo.hpp"
#include "Xped/PEPS/Models/KondoNecklace.hpp"

int main(int argc, char* argv[])
{
    // using Scalar = std::complex<double>;
    using Scalar = double;
    // using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
    using Symmetry = Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>;
    // using Symmetry = Xped::Sym::ZN<Xped::Sym::SpinU1, 36>;
    // using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
    // using Symmetry = Xped::Sym::Combined<Xped::Sym::ZN<Xped::Sym::SpinU1, 36>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 36>>;
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
        pat = Xped::Pattern(toml::get<std::vector<std::vector<std::size_t>>>(toml::find(data.at("ipeps"), "pattern")));
        c = Xped::UnitCell(pat);
    } else if(data.at("ipeps").contains("cell")) {
        auto [Lx, Ly] = toml::get<std::pair<int, int>>(toml::find(data.at("ipeps"), "cell"));
        c = Xped::UnitCell(Lx, Ly);
    }

    std::map<std::string, Xped::Param> params = Xped::util::params_from_toml(data.at("model").at("params"), c);

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

    std::unique_ptr<Xped::Hamiltonian<double, Symmetry>> ham;
    if(toml::find(data.at("model"), "name").as_string() == "Heisenberg") {
        ham = std::make_unique<Xped::Heisenberg<Symmetry>>(params, c.pattern, bonds);
    } else if(toml::find(data.at("model"), "name").as_string() == "KondoNecklace") {
        ham = std::make_unique<Xped::KondoNecklace<Symmetry>>(params, c.pattern, bonds);
    } else if(toml::find(data.at("model"), "name").as_string() == "Hubbard") {
        ham = std::make_unique<Xped::Hubbard<Symmetry>>(params, c.pattern, bonds);
    } // else if(toml::find(data.at("model"), "name").as_string() == "Kondo") {
    //     ham = std::make_unique<Xped::Kondo<Symmetry>>(params, c.pattern, bonds);
    // }
    else {
        throw std::invalid_argument("Specified model is not implemented.");
    }

    Xped::Opts::CTM ctm_opts = Xped::Opts::ctm_from_toml(data.at("ctm"));

    std::string load_psi = toml::get<std::string>(toml::find(data.at("global"), "load_psi"));
    Xped::Log::on_entry(ctm_opts.verbosity, "Load initial iPEPS from native file {}.", load_psi);
    constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
    Xped::iPEPS<Scalar, Symmetry> tmp_Psi;
    try {
        yas::load<flags>(load_psi.c_str(), tmp_Psi);
    } catch(const yas::serialization_exception& se) {
        fmt::print(
            "Error while deserializing file ({}) with initial wavefunction.\nThis might be because of incompatible symmetries between this simulation and the loaded wavefunction.",
            load_psi);
        std::cout << std::flush;
        throw;
    } catch(const yas::io_exception& ie) {
        fmt::print("Error while loading file ({}) with initial wavefunction.\n", load_psi);
        std::cout << std::flush;
        throw;
    } catch(const std::exception& e) {
        fmt::print("Unknown error while loading file ({}) with initial wavefunction.\n", load_psi);
        std::cout << std::flush;
        throw;
    }
    auto Psi = std::make_shared<Xped::iPEPS<Scalar, Symmetry>>(std::move(tmp_Psi));

    Xped::CTM<Scalar, Symmetry> Jack(Psi, ctm_opts.chi, ctm_opts.init);
    Jack.init();
    Xped::Log::on_entry("Performing CTM with χ={}", ctm_opts.chi);
    for(std::size_t step = 0; step < ctm_opts.max_presteps; ++step) {
        Xped::util::Stopwatch<> move_t;
        Jack.grow_all();
        Jack.computeRDM();
        auto hamobs = ham->asObservable();
        auto [E_h, E_v, E_d1, E_d2] = avg(Jack, hamobs);
        auto E = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jack.cell().uniqueSize();
        fmt::print("{: >3} {:2d}: E={:2.8f}, t={}\n", "▷", step, E, move_t.time_string());
    }
    using qType = typename Symmetry::qType;
    Xped::correlation_length(Jack, Xped::Opts::Orientation::H, 0, 0);
    // Xped::correlation_length(Jack, Xped::Opts::Orientation::H, 0, 1);
    // Xped::correlation_length(Jack, Xped::Opts::Orientation::H, 1, 0);
    // Xped::correlation_length(Jack, Xped::Opts::Orientation::H, 1, 1);
}
