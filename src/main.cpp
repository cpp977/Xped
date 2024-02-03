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

    // using Symmetry = Xped::Sym::U0<>;
    using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
    using Scalar = double;
    static thread_local std::mt19937 engine(std::random_device{}());
    engine.seed(0);
    std::size_t doublets = atoi(argv[1]);
    Xped::Qbasis<Symmetry> B;
    B.push_back({1}, 1);
    B.push_back({2}, doublets);
    // B.push_back({3}, 1);
    // B.push_back({4}, 1);
    B.sort();
    // B.push_back({}, doublets);
    Xped::Qbasis<Symmetry> loc;
    loc.push_back({2}, 1);

    using PEPSTensor = Xped::Tensor<Scalar, 4, 1, Symmetry>;
    // Xped::Tensor<Scalar, 4, 1, Symmetry> T1({{B,B,B,B}}, {{loc}});
    // T1.setRandom(engine);
    // fmt::print("Identity: 0, 1, 3, 4:\n");
    // T1.print(std::cout, true);
    // std::cout << std::endl;


    // fmt::print("Z2 discrete sym svs:\n");
    // S.print(std::cout, true);
    // std::cout << std::endl;

    // auto sym = (rot01*T1 + rot10 * T1).eval();

    // fmt::print("Symmetric\n");
    // sym.print(std::cout, true);
    // std::cout << std::endl;
    // auto T2 = rot * T1;
    // fmt::print("Swap: 2, 3, 1, 0:\n");
    // T2.print(std::cout, true);
    // std::cout << std::endl;

    // auto R1 = T1.getRotOperator<0, 2, 3, 1>();
    // fmt::print("Swap2: 2, 3, 1, 0:\n");
    // T2p.print(std::cout, true);
    // std::cout << std::endl;

    // auto res = (T2 - T2p).norm();
    // fmt::print("check={}\n", res);
    // fmt::print("U-D: 2,1,0,3:\n");
    // auto R1 = T1.getRotOperator<2, 1, 0, 3>();

    // // fmt::print("L-R: 0,3,2,1:\n");
    // auto R2 = T1.getRotOperator<0, 3, 2, 1>();

    // // fmt::print("90deg CWR: 3,0,1,2:\n");
    // auto R3 = T1.getRotOperator<3, 0, 1, 2>();

    // // fmt::print("90deg CCWR: 1,2,3,0:\n");
    // auto R4 = T1.getRotOperator<1, 2, 3, 0>();

    // // fmt::print("180deg R: 2,3,0,1:\n");
    // auto R5 = T1.getRotOperator<2, 3, 0, 1>();

    // // fmt::print("Diag1 Refl: 1,0,3,2:\n");
    // auto R6 = T1.getRotOperator<1, 0, 3, 2>();

    // // fmt::print("Diag1 Refl: 3,2,1,0:\n");
    // auto R7 = T1.getRotOperator<3, 2, 1, 0>();
    // auto R8 = T1.getRotOperator<0, 1, 2, 3>();
    // auto res = (0.125 * (R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8)).eval();
    // double trunc = 0;
    // auto [U,S,Vdag] = res.tSVD(10000ul, 1.e-12, trunc);
    // std::map<int, int> ref{{1,2}, {2,10}, {3,30}, {4,68}, {5, -1}};

    auto R1 = PEPSTensor::RotOperator<2, 1, 0, 3>({{B, B, B, B}}, {{loc}});

    // fmt::print("L-R: 0,3,2,1:\n");
    auto R2 = PEPSTensor::RotOperator<0, 3, 2, 1>({{B, B, B, B}}, {{loc}});

    // fmt::print("90deg CWR: 3,0,1,2:\n");
    auto R3 = PEPSTensor::RotOperator<3, 0, 1, 2>({{B, B, B, B}}, {{loc}});

    // fmt::print("90deg CCWR: 1,2,3,0:\n");
    auto R4 = PEPSTensor::RotOperator<1, 2, 3, 0>({{B, B, B, B}}, {{loc}});

    // fmt::print("180deg R: 2,3,0,1:\n");
    auto R5 = PEPSTensor::RotOperator<2, 3, 0, 1>({{B, B, B, B}}, {{loc}});

    // fmt::print("Diag1 Refl: 1,0,3,2:\n");
    auto R6 = PEPSTensor::RotOperator<1, 0, 3, 2>({{B, B, B, B}}, {{loc}});

    // fmt::print("Diag1 Refl: 3,2,1,0:\n");
    auto R7 = PEPSTensor::RotOperator<3, 2, 1, 0>({{B, B, B, B}}, {{loc}});
    auto R8 = PEPSTensor::RotOperator<0, 1, 2, 3>({{B, B, B, B}}, {{loc}});
    auto res = (0.125 * (R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8)).eval();
    double trunc = 0;
    auto [U,S,Vdag] = res.tSVD(10000ul, 1.e-12, trunc);
    std::map<int, int> ref{{1,2}, {2,10}, {3,30}, {4,68}, {5, -1}};
    fmt::print("# svs={} vs ref={}\n", S.block(0).rows(), ref[doublets]);
    // // // fmt::print("Trees:\n{}\n", Tsym.coupledDomain().printTrees());

    // auto computeMap = [](auto A) {
    //     auto comp = [](Scalar s1, Scalar s2) {
    //         if(std::abs(s1 - s2) < 1.e-12) { return false; }
    //         return s1 < s2;
    //     };
    //     std::map<Scalar, std::size_t, decltype(comp)> unique(comp);
    //     // std::unordered_map<Scalar, std::size_t> unique;
    //     std::size_t count_tot = 0ul;
    //     std::size_t count_unique = 0ul;
    //     std::pair<std::size_t, std::vector<std::size_t>> res_map;
    //     res_map.second.resize(A.plainSize());
    //     for(auto it = A.begin(); it != A.end(); ++it) {
    //         if(auto it_unique = unique.find(*it); it_unique == unique.end()) {
    //             res_map.second[count_tot] = count_unique;
    //             unique.insert(std::make_pair(*it, count_unique));
    //             ++count_unique;
    //         } else {
    //             res_map.second[count_tot] = it_unique->second;
    //         }
    //         ++count_tot;
    //     }
    //     res_map.first = count_unique;
    //     fmt::print("#unique elements={}, res_map:\n{}\n", res_map.first, res_map.second);
    //     return res_map;
    // };

    // auto res = computeMap(Tsym);
    // fmt::print("res={}\n", res);
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
