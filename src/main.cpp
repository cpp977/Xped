#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "nlohmann/json.hpp"

#ifdef _OPENMP
#    include "omp.h"
#endif

using std::cout;
using std::endl;
using std::size_t;
using std::string;

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
XPED_INIT_TREE_CACHE_VARIABLE(tree_cache, 100)
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
// #include "Xped/MPS/Mps.hpp"
// #include "Xped/MPS/MpsAlgebra.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/ExactContractions.hpp"
#include "Xped/PEPS/Models/Heisenberg.hpp"
#include "Xped/PEPS/iPEPS.hpp"
// #include "Xped/Physics/FermionBase.hpp"
// #include "Xped/Physics/SpinBase.hpp"
// #include "Xped/Physics/SpinlessFermionBase.hpp"

#include "Xped/Util/Stopwatch.hpp"

#include "Xped/IO/Json.hpp"
#include "Xped/IO/Matlab.hpp"

int main(int argc, char* argv[])
{
#ifdef XPED_USE_MPI
    MPI_Init(&argc, &argv);
    Xped::mpi::XpedWorld world(argc, argv);
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log_" + to_string(world.rank) + ".txt");
#else
    Xped::mpi::XpedWorld world;
    auto my_logger = spdlog::basic_logger_mt("info", "logs/log.txt");
#endif
    spdlog::set_default_logger(my_logger);
    std::ios::sync_with_stdio(true);

    // using Symmetry1 = Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>, Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>;
    // using Symmetry1 = Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>;
    // using Symmetry2 = Xped::Sym::ZN<Xped::Sym::ChargeU1, 2>;
    // Xped::FermionBase<Symmetry2> F1(1);
    // Xped::FermionBase<Symmetry2> F2(2);
    // auto hopping = (-1. * (tprod(F1.cdag(Xped::SPIN_INDEX::UP) * F1.sign_local(), F1.c(Xped::SPIN_INDEX::UP)) +
    //                        tprod(F1.cdag(Xped::SPIN_INDEX::DN) * F1.sign_local(), F1.c(Xped::SPIN_INDEX::DN)) -
    //                        tprod(F1.c(Xped::SPIN_INDEX::UP) * F1.sign_local(), F1.cdag(Xped::SPIN_INDEX::UP)) -
    //                        tprod(F1.c(Xped::SPIN_INDEX::DN) * F1.sign_local(), F1.cdag(Xped::SPIN_INDEX::DN))))
    //                    .eval();
    // std::cout << hopping << std::endl;
    // // std::cout << F2.cdag(Xped::SPIN_INDEX::UP, 1) << std::endl;
    // // std::cout << F2.cdag(Xped::SPIN_INDEX::DN, 0) << std::endl;
    // // std::cout << F2.cdag(Xped::SPIN_INDEX::DN, 1) << std::endl;
    // auto cdagc =
    //     -1. * (F2.cdag(Xped::SPIN_INDEX::UP, 0) * F2.c(Xped::SPIN_INDEX::UP, 1) - F2.c(Xped::SPIN_INDEX::UP, 0) * F2.cdag(Xped::SPIN_INDEX::UP, 1)
    //     +
    //            F2.cdag(Xped::SPIN_INDEX::DN, 0) * F2.c(Xped::SPIN_INDEX::DN, 1) - F2.c(Xped::SPIN_INDEX::DN, 0) * F2.cdag(Xped::SPIN_INDEX::DN,
    //            1));
    // std::cout << cdagc << std::endl;

    // auto cdagc2 =
    //     (-1. * (tprod(F1.cdag(Xped::SPIN_INDEX::UP), F1.c(Xped::SPIN_INDEX::UP)) - tprod(F1.c(Xped::SPIN_INDEX::UP), F1.cdag(Xped::SPIN_INDEX::UP))
    //     +
    //             tprod(F1.cdag(Xped::SPIN_INDEX::DN), F1.c(Xped::SPIN_INDEX::DN)) - tprod(F1.c(Xped::SPIN_INDEX::DN),
    //             F1.cdag(Xped::SPIN_INDEX::DN))))
    //         .eval();

    // std::cout << cdagc2 << std::endl;
    // std::cout << cdagc2.adjoint().eval() << std::endl;

    my_logger->sinks()[0]->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
    if(world.rank == 0) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%H:%M:%S] [%n] [%^---%L---%$] [process %P] %v");
        my_logger->sinks().push_back(console_sink);
    }
    spdlog::set_default_logger(my_logger);

    SPDLOG_INFO("Number of MPI processes: {}", world.np);
    SPDLOG_INFO("I am process number #={}", world.rank);

    SPDLOG_INFO("Number of MPI processes: {}", world.np);

    std::string config_file = argc > 1 ? argv[1] : "config.toml";
    toml::value data;
    try {
        data = toml::parse(config_file);
        // std::cout << data << "\n";
    } catch(const toml::syntax_error& err) {
        std::cerr << "Parsing failed:\n" << err.what() << "\n";
        return 1;
    }

    std::filesystem::path p2json;

    if(data["json"].contains("load")) {
        std::filesystem::path tmp_of(static_cast<std::string>(data["json"].at("load").as_string()));
        if(tmp_of.is_relative()) {
            p2json = std::filesystem::current_path() / tmp_of;
        } else {
            p2json = tmp_of;
        }
    }

    using Symmetry = Xped::Sym::U0<>;
    using Scalar = double;
    Xped::UnitCell c(1, 1);
    Xped::TMatrix<Symmetry::qType> charges(c.pattern);
    charges.setConstant(Symmetry::qvacuum());
    std::map<std::string, Xped::Param> params = {
        {"Jxy", Xped::Param{1.}}, {"Jz", Xped::Param{1.}}, {"J", Xped::Param{1.}}, {"Bz", Xped::Param{0.}}, {"J2", Xped::Param{0.}}};
    Xped::Opts::Bond bonds = Xped::Opts::Bond::V | Xped::Opts::Bond::H;
    std::unique_ptr<Xped::TwoSiteObservable<double, Symmetry>> ham;
    ham = std::make_unique<Xped::Heisenberg<Symmetry>>(params, c.pattern, bonds);
    Xped::TMatrix<Xped::Qbasis<Symmetry, 1>> phys_basis(c.pattern);
    phys_basis.setConstant(ham->data_h[0].uncoupledDomain()[0]);

    std::size_t D = 2;
    if(data["random"].contains("D")) { D = static_cast<std::size_t>(data["random"].at("D").as_integer()); }
    Xped::TMatrix<Xped::Qbasis<Symmetry, 1>> left_aux(c.pattern), top_aux(c.pattern);
    Xped::iPEPS<Scalar, Symmetry, false> Psi(c, D, left_aux, top_aux, phys_basis, charges, Xped::Opts::DiscreteSym::C4v);
    std::size_t seed = 1;
    if(data["random"].contains("seed")) { seed = static_cast<std::size_t>(data["random"].at("seed").as_integer()); }
    Psi.setRandom(seed);
    Psi.normalize();
    // Psi.As[0].print(std::cout, true);
    // std::cout << std::endl;
    // auto norm1 = fourByfour(Psi);
    // Psi.As[0] = Psi.As[0] * 2.;
    // Psi.Adags[0] = Psi.As[0].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>();
    // auto energy2 = fourByfour(Psi, *ham);
    // fmt::print("energy1={}, energy2={}\n", energy1, energy2);
    // using Symmetry1 = Xped::Sym::ZN<Xped::Sym::SpinU1, 36>;
    // auto A1 = Xped::IO::loadSU2JsonTensor<Symmetry1>(p2json);
    // using Symmetry2 = Xped::Sym::U0<>;
    // auto A2 = Xped::IO::loadSU2JsonTensor<Symmetry2>(p2json);
    // using Symmetry3 = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
    // auto A3 = Xped::IO::loadSU2JsonTensor<Symmetry3>(p2json);

    // static thread_local std::mt19937 engine(std::random_device{}());
    // T.setRandom(engine);
    // HighFive::File file("/home/user/matlab-tmp/math5.mat", HighFive::File::ReadOnly);
    // auto py = file.getGroup("py");

    // auto C1_ref = py.getDataSet("C1");
    // std::vector<HighFive::Reference> C1;
    // C1_ref.read(C1);
    // auto g_C1_0 = C1[0].template dereference<HighFive::Group>(py);
    // auto C = Xped::IO::loadMatlabTensor<double, 2, 0, Xped::Sym::ZN<Xped::Sym::ChargeU1, 36>, Xped::HeapPolicy>(g_C1_0, py);
    // // std::cout << C.squaredNorm() << std::endl;
    // // std::cout << (C.template contract<std::array{1, 2}, std::array{1, 2}, 0>(C.adjoint().eval())).trace() << std::endl;

    // auto T1_ref = py.getDataSet("T1");
    // std::vector<HighFive::Reference> T1;
    // T1_ref.read(T1);
    // auto g_T1_0 = T1[0].template dereference<HighFive::Group>(py);
    // auto T = Xped::IO::loadMatlabTensor<double, 0, 4, Xped::Sym::ZN<Xped::Sym::ChargeU1, 36>, Xped::HeapPolicy>(g_T1_0, py);
    // // std::cout << T.squaredNorm() << std::endl;
    // // std::cout << (T.template contract<std::array{1, 2, 3, 4}, std::array{1, 2, 3, 4}, 0>(T.adjoint().eval())).trace() << std::endl;

    // auto CT = C.template contract<std::array{1, -1}, std::array{1, -2, -3, -4}, 4>(T);

    // auto H = Xped::IO::loadMatlabTensor<double, 2, 2, Xped::Sym::ZN<Xped::Sym::ChargeU1, 36>, Xped::HeapPolicy>(py.getGroup("H"), py);
    // auto [D, U] = H.eigh();
    // // D.print(std::cout);
    // // std::cout << H.squaredNorm() << std::endl;

    // // using Symmetry = Xped::Sym::U0<double>;
    // using Symmetry = Xped::Sym::U1<Xped::Sym::SpinU1, double>;
    // Xped::Qbasis<Symmetry, 1> b;
    // b.push_back({2}, 1);
    // Xped::SiteOperator<double, Symmetry> S({3}, b);
    // S.setZero();
    // S({2}, {2})(0, 0) = std::sqrt(0.75);
    // auto S2 = std::sqrt(3.) * Xped::SiteOperator<double, Symmetry>::prod(S.adjoint(), S, {1});
    // auto SdxS = std::sqrt(3.) * Xped::SiteOperator<double, Symmetry>::outerprod(S.adjoint(), S, {1});
    // auto SxSd = std::sqrt(3.) * Xped::SiteOperator<double, Symmetry>::outerprod(S, S.adjoint(), {1});
    // Xped::SpinBase<Symmetry> B(1, Dloc);
    // std::cout << B.Sz() << std::endl;
    // Xped::Tensor<double, 2, 2, Symmetry> ham = Xped::tprod(B.Sz(), B.Sz()) + 0.5 * (Xped::tprod(B.Sp(), B.Sm()) + Xped::tprod(B.Sm(), B.Sp()));
    // auto [Es, Us] = ham.eigh();
    // Es.print(std::cout);
    // std::cout << B.Sz(0) << std::endl;
    // std::cout << B.Sz(1) << std::endl;
    // std::cout << t << std::endl;
    // SPDLOG_INFO("Tensor: \n {}", t);
    // constexpr std::size_t Rank=2;
    // tensortraits<EigenTensorLib>::Ttype<double,Rank> T1(2,3); T1.setRandom();
    // tensortraits<EigenTensorLib>::Ttype<double,Rank> T2(2,3); T2.setRandom();
    // std::array<std::pair<std::size_t,std::size_t>,1> legs = {std::pair<std::size_t,std::size_t>{0,0}};
    // auto C = tensortraits<EigenTensorLib>::contract(T1,T2,legs);
    // auto T = tensortraits<EigenTensorLib>::tensorProd(T1,T1);
    // auto Ts = tensortraits<EigenTensorLib>::shuffle(C, {{1,0}});
    // std::array<std::size_t, 2> dims = {2,3};
    // auto cmap = tensortraits<EigenTensorLib>::cMap(T.data(), dims);
    // auto map = tensortraits<EigenTensorLib>::Map(T.data(), dims);
    // map(0,0)=10;
    // std::cout << cmap(0,0) << std::endl;

    // std::cout << T << std::endl;

    // std::cout << Ts << std::endl;

    // std::cout << std::endl << C << std::endl;

#ifdef XPED_CACHE_PERMUTE_OUTPUT
    std::cout << "total hits=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "total misses=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "hit rate=" << tree_cache<1, 2, 1, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif
// double trunc;
// auto [U, S, V] = Psi.A.Ac[L / 2].tSVD(10000ul, 1.e-10, trunc, false);
// Mps<Symmetry> Phi(L, qloc, Qtot, Minit, Qinit);
// cout << "<Phi,Psi>=" << dot(Phi, Psi) << endl;
// cout << "<Phi,Phi>=" << dot(Phi, Phi) << endl;
// double normSq = dot(Psi, Psi, DIR);
// std::cout << "<Psi|Psi>=" << normSq << std::endl;
// cout << "Left: <Psi,Psi>=" << normSq << endl;
// Psi.A.Ac[0] = Psi.A.Ac[0] * std::pow(normSq, -0.5);
// cout << "<Psi,Psi>=" << dot(Psi, Psi, DIR) << endl;
#ifdef XPED_USE_MPI
    SPDLOG_CRITICAL("Calling MPI_Finalize()");
    // volatile int i = 0;
    // char hostname[256];
    // gethostname(hostname, sizeof(hostname));
    // printf("PID %d on %s ready for attach\n", getpid(), hostname);
    // fflush(stdout);
    // while(0 == i) sleep(5);
    MPI_Finalize();
#endif
}
