#include <array>
#include <iostream>

#include "spdlog/spdlog.h"

#include "Xped/PEPS/CTM.hpp"

#include "Xped/Core/CoeffUnaryOp.hpp"
#include "Xped/PEPS/Models/KondoNecklace.hpp"
#include "Xped/PEPS/PEPSContractions.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::init()
{
    C1s.resize(cell_.pattern);
    C2s.resize(cell_.pattern);
    C3s.resize(cell_.pattern);
    C4s.resize(cell_.pattern);
    T1s.resize(cell_.pattern);
    T2s.resize(cell_.pattern);
    T3s.resize(cell_.pattern);
    T4s.resize(cell_.pattern);

    Svs.resize(cell_.pattern);

    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);
            switch(init_m) {
            case INIT::FROM_TRIVIAL: {
                C1s[pos] =
                    Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD>({{}}, {{Qbasis<Symmetry, 1>::TrivialBasis(), Qbasis<Symmetry, 1>::TrivialBasis()}});
                C1s[pos].setRandom();
                // C1s[pos] = C1s[pos].square();
                C2s[pos] =
                    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                C2s[pos].setRandom();
                // C2s[pos] = C2s[pos].square();
                C3s[pos] =
                    Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis(), Qbasis<Symmetry, 1>::TrivialBasis()}}, {{}});
                C3s[pos].setRandom();
                // C3s[pos] = C3s[pos].square();
                C4s[pos] =
                    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                C4s[pos].setRandom();
                // C4s[pos] = C4s[pos].square();
                T1s[pos] = Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                                     {{Qbasis<Symmetry, 1>::TrivialBasis(),
                                                                       A->ketBasis(x, y + 1, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::UP),
                                                                       A->braBasis(x, y + 1, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::UP)}});
                T1s[pos].setRandom();
                // T1s[pos] = T1s[pos].square();
                T2s[pos] = Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>({{A->ketBasis(x - 1, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::RIGHT),
                                                                       A->braBasis(x - 1, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::RIGHT),
                                                                       Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                                     {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                T2s[pos].setRandom();
                // T2s[pos] = T2s[pos].square();
                T3s[pos] = Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>({{A->ketBasis(x, y - 1, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::DOWN),
                                                                       A->braBasis(x, y - 1, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::DOWN),
                                                                       Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                                     {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                T3s[pos].setRandom();
                // T3s[pos] = T3s[pos].square();
                T4s[pos] = Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                                     {{Qbasis<Symmetry, 1>::TrivialBasis(),
                                                                       A->ketBasis(x + 1, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::LEFT),
                                                                       A->braBasis(x + 1, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::LEFT)}});
                T4s[pos].setRandom();
                // T4s[pos] = T4s[pos].square();

                break;
            }
            case INIT::FROM_A: {
                auto fuse_ll =
                    Tensor<Scalar, 1, 2, Symmetry, false>::Identity({{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::LEFT)
                                                                          .combine(A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::LEFT))
                                                                          .forgetHistory()}},
                                                                    {{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::LEFT),
                                                                      A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::LEFT)}});
                auto fuse_uu =
                    Tensor<Scalar, 1, 2, Symmetry, false>::Identity({{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::UP)
                                                                          .combine(A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::UP))
                                                                          .forgetHistory()}},
                                                                    {{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::UP),
                                                                      A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::UP)}});
                auto fuse_rr =
                    Tensor<Scalar, 2, 1, Symmetry, false>::Identity({{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::RIGHT),
                                                                      A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::RIGHT)}},
                                                                    {{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::RIGHT)
                                                                          .combine(A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::RIGHT))
                                                                          .forgetHistory()}});
                auto fuse_dd =
                    Tensor<Scalar, 2, 1, Symmetry, false>::Identity({{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::DOWN),
                                                                      A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::DOWN)}},
                                                                    {{A->ketBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::DOWN)
                                                                          .combine(A->braBasis(x, y, iPEPS<Scalar, Symmetry, ENABLE_AD>::LEG::DOWN))
                                                                          .forgetHistory()}});

                C1s[pos] = A->As[pos]
                               .template contract<std::array{1, 2, -1, -2, 3}, std::array{1, 2, 3, -3, -4}, 2>(A->Adags[pos])
                               .template contract<std::array{1, -1, 2, -2}, std::array{1, 2, -3}, 2>(fuse_rr)
                               .template contract<std::array{1, 2, -2}, std::array{1, 2, -1}, 0>(fuse_dd);
                C2s[pos] = A->As[pos]
                               .template contract<std::array{-1, 1, 2, -2, 3}, std::array{-3, 1, 3, 2, -4}, 2>(A->Adags[pos])
                               .template contract<std::array{1, -1, 2, -2}, std::array{-3, 1, 2}, 2>(fuse_ll)
                               .template contract<std::array{1, 2, -1}, std::array{1, 2, -2}, 1>(fuse_dd);
                C3s[pos] = A->As[pos]
                               .template contract<std::array{-1, -2, 1, 2, 3}, std::array{-3, -4, 3, 1, 2}, 2>(A->Adags[pos])
                               .template contract<std::array{1, -1, 2, -2}, std::array{-3, 1, 2}, 2>(fuse_ll)
                               .template contract<std::array{1, 2, -2}, std::array{-1, 1, 2}, 2>(fuse_uu);
                C4s[pos] = A->As[pos]
                               .template contract<std::array{1, -1, -2, 2, 3}, std::array{1, -3, 3, -4, 2}, 2>(A->Adags[pos])
                               .template contract<std::array{1, -1, 2, -2}, std::array{-3, 1, 2}, 2>(fuse_uu)
                               .template contract<std::array{1, 2, -1}, std::array{1, 2, -2}, 1>(fuse_rr);

                T1s[pos] = A->As[pos]
                               .template contract<std::array{-1, 1, -2, -3, 2}, std::array{-4, 1, 2, -5, -6}, 3>(A->Adags[pos])
                               .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_ll)
                               .template contract<std::array{1, -3, 2, -4, -1}, std::array{1, 2, -2}, 1>(fuse_rr);
                T2s[pos] = A->As[pos]
                               .template contract<std::array{-1, -2, 1, -3, 2}, std::array{-4, -5, 2, 1, -6}, 3>(A->Adags[pos])
                               .template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{-5, 1, 2}, 4>(fuse_uu)
                               .template contract<std::array{-1, 1, -2, 2, -3}, std::array{1, 2, -4}, 3>(fuse_dd);
                T3s[pos] = A->As[pos]
                               .template contract<std::array{-1, -2, -3, 1, 2}, std::array{-4, -5, 2, -6, 1}, 3>(A->Adags[pos])
                               .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_ll)
                               .template contract<std::array{-1, 1, -2, 2, -3}, std::array{1, 2, -4}, 3>(fuse_rr);
                T4s[pos] = A->As[pos]
                               .template contract<std::array{1, -1, -2, -3, 2}, std::array{1, -4, 2, -5, -6}, 3>(A->Adags[pos])
                               .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_uu)
                               .template contract<std::array{-3, 1, -4, 2, -1}, std::array{1, 2, -2}, 1>(fuse_dd);
                break;
            }
            }
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::solve()
{
    info();
    stan::math::print_stack(std::cout);
    for(std::size_t step = 0; step < opts.max_steps; ++step) {
        SPDLOG_CRITICAL("Step={}", step);
        left_move();
        right_move();
        top_move();
        bottom_move();
        // info();
        // computeRDM();
        // auto [E_h, E_v] = avg(*this, KondoNecklace::twoSiteHamiltonian(1.));
        // std::cout << "Energy (horizontal):\n" << E_h << std::endl;
        // std::cout << "Energy (vertical)  :\n" << E_v << std::endl;
        // SPDLOG_CRITICAL("E={}", (E_h.sum() + E_v.sum()) / cell_.size());
        // checkConvergence(1.e-8);
    }
    computeRDM();
    stan::math::print_stack(std::cout);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::computeRDM()
{
    computeRDM_h();
    computeRDM_v();
    HAS_RDM = true;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::info() const
{
    std::string mode_string = "";
    switch(init_m) {
    case INIT::FROM_TRIVIAL: {
        mode_string = "FROM_TRIVIAL";
        break;
    }
    case INIT::FROM_A: {
        mode_string = "FROM_A";
        break;
    }
    }

    std::cout << "CTM(χ=" << chi << "): UnitCell=(" << cell_.Lx << "x" << cell_.Ly << ")"
              << ", init=" << mode_string << std::endl;
    // std::cout << "Tensors:" << std::endl;
    // for(int x = 0; x < cell_.Lx; x++) {
    //     for(int y = 0; y < cell_.Lx; y++) {
    //         if(not cell_.pattern.isUnique(x, y)) {
    //             std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
    //             continue;
    //         }
    //         std::cout << "Cell site: (" << x << "," << y << "), C1: " << C1s(x, y).block(0)(0, 0) << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), C2: " << C2s(x, y).block(0)(0, 0) << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), C3: " << C3s(x, y).block(0)(0, 0) << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), C4: " << C4s(x, y).block(0)(0, 0) << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), T1: " << T1s(x, y).block(0)(0, 0) << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), T2: " << T2s(x, y).block(0)(0, 0) << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), T3: " << T3s(x, y).block(0)(0, 0) << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), T4: " << T4s(x, y).block(0)(0, 0) << std::endl << std::endl;
    //     }
    // }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
bool CTM<Scalar, Symmetry, ENABLE_AD>::checkConvergence(typename ScalarTraits<Scalar>::Real epsilon)
{
    for(int x = 0; x < cell_.Lx; ++x) {
        for(int y = 0; y < cell_.Ly; ++y) {
            auto C1T4 = C1s(x, y - 1).template contract<std::array{1, -1}, std::array{1, -2, -3, -4}, 1>(T4s(x, y));
            auto C1T4C4 = C1T4.template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 2>(C4s(x, y + 1));

            auto corner = contractCorner(x, y, CORNER::UPPER_LEFT);
            auto cornerC4 = corner.template contract<std::array{1, -1, -2, -3, -4, -5}, std::array{1, -6}, 5>(C4s(x - 1, y + 1));
            auto C1T4C4check = cornerC4.template contract<std::array{1, 2, -1, -2, -3, 3}, std::array{1, 2, 3, -4}, 2>(T3s(x, y + 1));
            Scalar diff = (C1T4C4 - C1T4C4check).norm();
            SPDLOG_INFO("x,y={},{} diff={}", x, y, diff);
            if(std::abs(diff) > epsilon) {
                // C1T4C4.print(std::cout, true);
                // C1T4C4check.print(std::cout, true);
                // (C1T4C4 - C1T4C4check).eval().print(std::cout, true);
            }
        }
    }
    return true;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::computeRDM_h()
{
    rho_h = TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>>(cell_.pattern);
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(rho_h.isChanged(x, y)) { continue; }
            auto C1T1 = C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1>(T1s(x, y - 1));
            SPDLOG_INFO("Computed C1T1");
            // auto tmp = C1s(x - 1, y - 1).template permute<-1, 0, 1>() * T1s(x, y - 1);
            auto T4C1T1 = T4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3>(C1T1);
            // std::cout << "T4C1T1 trees" << std::endl;
            // for(const auto& [q, tree] : T4C1T1.coupledDomain().trees) {
            //     std::cout << "Q=" << q << std::endl;
            //     for(const auto& t : tree) { std::cout << t.draw() << std::endl; }
            //     std::cout << std::endl;
            // }
            SPDLOG_INFO("Computed T4C1T1");
            // auto tmp2 = T4s(x - 1, y).template permute<-2, 1, 2, 3, 0>() * tmp;
            auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4>(A->As(x, y));
            SPDLOG_INFO("Computed T4C1T1A");
            // auto UL = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A->As(x, y);
            auto C4T3 = C4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1>(T3s(x, y + 1));
            SPDLOG_INFO("Computed C4T3");
            auto C4T3Ad = C4T3.template contract<std::array{-1, -2, 1, -3}, std::array{-4, -5, -6, -7, 1}, 3>(A->Adags(x, y));
            SPDLOG_INFO("Computed C4T3Ad");
            auto left_half = T4C1T1A.template contract<std::array{1, 2, -1, 3, -2, 4, -3}, std::array{1, 4, -4, 2, 3, -5, -6}, 3>(C4T3Ad);
            SPDLOG_INFO("Computed left_half");

            auto T1C2 = T1s(x + 1, y - 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3>(C2s(x + 2, y - 1));
            SPDLOG_INFO("Computed T1C2");
            auto T1C2T2 = T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, 1, -6}, 3>(T2s(x + 2, y));
            SPDLOG_INFO("Computed T1C2T2");
            auto AT1C2T2 = A->As(x + 1, y).template contract<std::array{-1, 1, 2, -2, -3}, std::array{-4, 1, -5, 2, -6, -7}, 3>(T1C2T2);
            SPDLOG_INFO("Computed AT1C1T2");
            auto T3C3 = T3s(x + 1, y + 1).template contract<std::array{-1, -2, -3, 1}, std::array{-4, 1}, 3>(C3s(x + 2, y + 1));
            SPDLOG_INFO("Computed T3C3");
            auto T3C3Ad = T3C3.template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, -6, -7, 1}, 3>(A->Adags(x + 1, y));
            SPDLOG_INFO("Computed T3C3Ad");
            auto right_half = AT1C2T2.template contract<std::array{-1, 1, -2, -3, 2, 3, 4}, std::array{1, -4, 4, -5, 2, -6, 3}, 3>(T3C3Ad);
            SPDLOG_INFO("Computed right_half");
            rho_h(x, y) = left_half.template contract<std::array{1, 2, -3, 3, -1, 4}, std::array{2, -4, 1, 3, 4, -2}, 2>(right_half);
            SPDLOG_CRITICAL("x,y={},{} Tr_rho_h={}", x, y, rho_h(x, y).val().trace());
            // assert(rho_h(x, y).val().trace() > 0 and "Negative norm detected");
            rho_h(x, y) = rho_h(x, y) * (1. / rho_h(x, y).trace());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::computeRDM_v()
{
    rho_v = TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>>(cell_.pattern);
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(rho_v.isChanged(x, y)) { continue; }
            auto C1T1 = C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1>(T1s(x, y - 1));
            SPDLOG_INFO("Computed C1T1");
            auto T4C1T1 = T4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3>(C1T1);
            SPDLOG_INFO("Computed T4C1T1");
            auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4>(A->As(x, y));
            SPDLOG_INFO("Computed T4C1T1A");
            auto C2T2 = C2s(x + 1, y - 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1>(T2s(x + 1, y));
            SPDLOG_INFO("Computed C4T3");
            auto AdC2T2 = A->Adags(x, y).template contract<std::array{-1, -2, -3, 1, -4}, std::array{-5, -6, 1, -7}, 4>(C2T2);
            SPDLOG_INFO("Computed C4T3Ad");
            auto upper_half = T4C1T1A.template contract<std::array{-1, 1, 2, 3, 4, -2, -3}, std::array{1, 3, -4, -5, 2, 4, -6}, 3>(AdC2T2);
            SPDLOG_INFO("Computed left_half");

            auto C4T3 = C4s(x - 1, y + 2).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1>(T3s(x, y + 2));
            SPDLOG_INFO("Computed T1C2");
            auto T4C4T3 = T4s(x - 1, y + 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3>(C4T3);
            SPDLOG_INFO("Computed T1C2T2");
            auto AT4C4T3 = A->As(x, y + 1).template contract<std::array{1, -1, -2, 2, -3}, std::array{-4, 1, -5, 2, -6, -7}, 3>(T4C4T3);
            SPDLOG_INFO("Computed AT1C1T2");
            auto T2C3 = T2s(x + 1, y + 1).template contract<std::array{-1, -2, -3, 1}, std::array{1, -4}, 3>(C3s(x + 1, y + 2));
            SPDLOG_INFO("Computed T3C3");
            auto T2C3Ad = T2C3.template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, -6, 1, -7}, 3>(A->Adags(x, y + 1));
            SPDLOG_INFO("Computed T3C3Ad");
            // std::cout << AT4C4T3.template permute<0, 2, 3, 5, 0, 1, 4, 6>().coupledCodomain() << std::endl
            //           << T2C3Ad.template permute<-1, 4, 6, 3, 1, 0, 2, 5>().coupledDomain() << std::endl;
            auto lower_half = AT4C4T3.template contract<std::array{-1, 1, -2, -3, 2, 3, 4}, std::array{1, -4, 4, 2, -5, -6, 3}, 3>(T2C3Ad);
            SPDLOG_INFO("Computed right_half");

            rho_v(x, y) = upper_half.template contract<std::array{1, 2, -3, -1, 3, 4}, std::array{2, -4, 1, 4, 3, -2}, 2>(lower_half);
            SPDLOG_CRITICAL("x,y={},{} Tr_rho_v={}", x, y, rho_v(x, y).val().trace());
            // assert(rho_v(x, y).val().trace() > 0 and "Negative norm detected");
            rho_v(x, y) = rho_v(x, y) * (1. / rho_v(x, y).trace());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::left_move()
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD>> C1_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>> T4_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> C4_new(cell_.pattern);

    C1s.resetChange();
    C4s.resetChange();
    T4s.resetChange();
    SPDLOG_INFO("left move");
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, DIRECTION::LEFT); // move assignment
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(C1_new.isChanged(x, y - 1) == T4_new.isChanged(x, y) and C1_new.isChanged(x, y - 1) == C4_new.isChanged(x, y + 1));
            if(C1_new.isChanged(x, y - 1)) { continue; }
            std::tie(C1_new(x, y - 1), T4_new(x, y), C4_new(x, y + 1)) = renormalize_left(x, y, P1, P2);
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(C1s.isChanged(x, y) == T4s.isChanged(x, y) and C1s.isChanged(x, y) == C4s.isChanged(x, y));
            if(C1s.isChanged(x, y)) { continue; }
            C1s(x, y) = std::as_const(C1_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC1new: ([{}],[{}⊗{}])",
                        x,
                        y,
                        C1s(x, y).coupledDomain().dim(),
                        C1s(x, y).uncoupledCodomain()[0].dim(),
                        C1s(x, y).uncoupledCodomain()[1].dim());
            T4s(x, y) = std::as_const(T4_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT4new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        T4s(x, y).coupledDomain().dim(),
                        T4s(x, y).uncoupledCodomain()[0].dim(),
                        T4s(x, y).uncoupledCodomain()[1].dim(),
                        T4s(x, y).uncoupledCodomain()[2].dim());

            C4s(x, y) = std::as_const(C4_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC4new: ([{}],[{}])", x, y, C4s(x, y).uncoupledDomain()[0].dim(), C4s(x, y).uncoupledCodomain()[0].dim());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::right_move()
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> C2_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>> T2_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD>> C3_new(cell_.pattern);

    C2s.resetChange();
    C3s.resetChange();
    T2s.resetChange();
    SPDLOG_INFO("right move");
    for(int x = cell_.Lx; x >= 0; --x) {
        for(int y = 0; y < cell_.Ly; y++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, DIRECTION::RIGHT); // move assignment
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(C2_new.isChanged(x, y - 1) == T2_new.isChanged(x, y) and C2_new.isChanged(x, y - 1) == C3_new.isChanged(x, y + 1));
            if(C2_new.isChanged(x, y - 1)) { continue; }
            std::tie(C2_new(x, y - 1), T2_new(x, y), C3_new(x, y + 1)) = renormalize_right(x, y, P1, P2);
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(C2s.isChanged(x, y) == T2s.isChanged(x, y) and C2s.isChanged(x, y) == C3s.isChanged(x, y));
            if(C2s.isChanged(x, y)) { continue; }
            C2s(x, y) = std::as_const(C2_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC2new: ([{}],[{}])", x, y, C2s(x, y).uncoupledDomain()[0].dim(), C2s(x, y).uncoupledCodomain()[0].dim());
            T2s(x, y) = std::as_const(T2_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT2new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        T2s(x, y).uncoupledDomain()[0].dim(),
                        T2s(x, y).uncoupledDomain()[1].dim(),
                        T2s(x, y).uncoupledDomain()[2].dim(),
                        T2s(x, y).coupledCodomain().dim());
            C3s(x, y) = std::as_const(C3_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC3new: ([{}⊗{}],[{}])",
                        x,
                        y,
                        C3s(x, y).uncoupledDomain()[0].dim(),
                        C3s(x, y).uncoupledDomain()[1].dim(),
                        C3s(x, y).coupledCodomain().dim());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::top_move()
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD>> C1_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>> T1_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> C2_new(cell_.pattern);

    C1s.resetChange();
    C2s.resetChange();
    T1s.resetChange();
    SPDLOG_INFO("top move");
    for(int y = 0; y < cell_.Ly; y++) {
        for(int x = 0; x < cell_.Lx; x++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, DIRECTION::TOP); // move assignment
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(C1_new.isChanged(x - 1, y) == C2_new.isChanged(x + 1, y) and C1_new.isChanged(x - 1, y) == T1_new.isChanged(x, y));
            if(C1_new.isChanged(x - 1, y)) { continue; }
            std::tie(C1_new(x - 1, y), T1_new(x, y), C2_new(x + 1, y)) = renormalize_top(x, y, P1, P2);
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(C1s.isChanged(x, y) == C2s.isChanged(x, y) and C1s.isChanged(x, y) == T1s.isChanged(x, y));
            if(C1s.isChanged(x, y)) { continue; }
            C1s(x, y) = std::as_const(C1_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC1new: ([{}],[{}⊗{}])",
                        x,
                        y,
                        C1s(x, y).coupledDomain().dim(),
                        C1s(x, y).uncoupledCodomain()[0].dim(),
                        C1s(x, y).uncoupledCodomain()[1].dim());
            T1s(x, y) = std::as_const(T1_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT1new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        T1s(x, y).coupledDomain().dim(),
                        T1s(x, y).uncoupledCodomain()[0].dim(),
                        T1s(x, y).uncoupledCodomain()[1].dim(),
                        T1s(x, y).uncoupledCodomain()[2].dim());
            C2s(x, y) = std::as_const(C2_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC2new: ([{}],[{}])", x, y, C2s(x, y).uncoupledDomain()[0].dim(), C2s(x, y).uncoupledCodomain()[0].dim());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void CTM<Scalar, Symmetry, ENABLE_AD>::bottom_move()
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> C4_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>> T3_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD>> C3_new(cell_.pattern);

    C4s.resetChange();
    C3s.resetChange();
    T3s.resetChange();
    SPDLOG_INFO("bottom move");
    for(int y = cell_.Ly; y >= 0; --y) {
        for(int x = 0; x < cell_.Lx; x++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, DIRECTION::BOTTOM); // move assignment
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(C4_new.isChanged(x - 1, y) == C3_new.isChanged(x + 1, y) and C4_new.isChanged(x - 1, y) == T3_new.isChanged(x, y));
            if(C4_new.isChanged(x - 1, y)) { continue; }
            std::tie(C4_new(x - 1, y), T3_new(x, y), C3_new(x + 1, y)) = renormalize_bottom(x, y, P1, P2);
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(C4s.isChanged(x, y) == C3s.isChanged(x, y) and C4s.isChanged(x, y) == T3s.isChanged(x, y));
            if(C4s.isChanged(x, y)) { continue; }
            C4s(x, y) = std::as_const(C4_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC4new: ([{}],[{}])", x, y, C4s(x, y).uncoupledDomain()[0].dim(), C4s(x, y).uncoupledCodomain()[0].dim());
            T3s(x, y) = std::as_const(T3_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT3new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        T3s(x, y).uncoupledDomain()[0].dim(),
                        T3s(x, y).uncoupledDomain()[1].dim(),
                        T3s(x, y).uncoupledDomain()[2].dim(),
                        T3s(x, y).coupledCodomain().dim());
            C3s(x, y) = std::as_const(C3_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC3new: ([{}⊗{}],[{}])",
                        x,
                        y,
                        C3s(x, y).uncoupledDomain()[0].dim(),
                        C3s(x, y).uncoupledDomain()[1].dim(),
                        C3s(x, y).coupledCodomain().dim());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::pair<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>, Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>>
CTM<Scalar, Symmetry, ENABLE_AD>::get_projectors(const int x, const int y, const DIRECTION dir) XPED_CONST
{
    Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD> P1;
    Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD> P2;
    Tensor<Scalar, 3, 3, Symmetry, ENABLE_AD> Q1, Q2, Q3, Q4;
    switch(dir) {
    case DIRECTION::LEFT: {
        switch(proj_m) {
        case PROJECTION::CORNER: {
            Q1 = contractCorner(x, y, CORNER::UPPER_LEFT);
            Q4 = contractCorner(x, y + 1, CORNER::LOWER_LEFT);
            // SPDLOG_INFO("Q1: ({}[{}],{}[{}])",
            //                 Q1.coupledDomain().fullDim(),
            //                 Q1.coupledDomain().dim(),
            //                 Q1.coupledCodomain().fullDim(),
            //                 Q1.coupledCodomain().dim());
            // SPDLOG_INFO("Q4: ({}[{}],{}[{}])",
            //                 Q4.coupledDomain().fullDim(),
            //                 Q4.coupledDomain().dim(),
            //                 Q4.coupledCodomain().fullDim(),
            //                 Q4.coupledCodomain().dim());
            std::tie(P1, P2) = decompose(Q4, Q1, chi);
            break;
        }
        case PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    case DIRECTION::RIGHT: {
        switch(proj_m) {
        case PROJECTION::CORNER: {
            Q2 = contractCorner(x, y, CORNER::UPPER_RIGHT);
            Q3 = contractCorner(x, y + 1, CORNER::LOWER_RIGHT);
            // SPDLOG_INFO("Q2: ({}[{}],{}[{}])",
            //                 Q2.coupledDomain().fullDim(),
            //                 Q2.coupledDomain().dim(),
            //                 Q2.coupledCodomain().fullDim(),
            //                 Q2.coupledCodomain().dim());
            // SPDLOG_INFO("Q3: ({}[{}],{}[{}])",
            //                 Q3.coupledDomain().fullDim(),
            //                 Q3.coupledDomain().dim(),
            //                 Q3.coupledCodomain().fullDim(),
            //                 Q3.coupledCodomain().dim());
            std::tie(P1, P2) = decompose(Q2, Q3, chi);
            break;
        }
        case PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    case DIRECTION::TOP: {
        switch(proj_m) {
        case PROJECTION::CORNER: {
            Q1 = contractCorner(x, y, CORNER::UPPER_LEFT);
            Q2 = contractCorner(x + 1, y, CORNER::UPPER_RIGHT);
            // SPDLOG_INFO("Q1: ({}[{}],{}[{}])",
            //                 Q1.coupledDomain().fullDim(),
            //                 Q1.coupledDomain().dim(),
            //                 Q1.coupledCodomain().fullDim(),
            //                 Q1.coupledCodomain().dim());
            // SPDLOG_INFO("Q2: ({}[{}],{}[{}])",
            //                 Q2.coupledDomain().fullDim(),
            //                 Q2.coupledDomain().dim(),
            //                 Q2.coupledCodomain().fullDim(),
            //                 Q2.coupledCodomain().dim());
            std::tie(P1, P2) = decompose(Q1, Q2, chi);
            break;
        }
        case PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    case DIRECTION::BOTTOM: {
        switch(proj_m) {
        case PROJECTION::CORNER: {
            Q4 = contractCorner(x, y, CORNER::LOWER_LEFT);
            Q3 = contractCorner(x + 1, y, CORNER::LOWER_RIGHT);
            // SPDLOG_INFO("Q4: ({}[{}],{}[{}])",
            //                 Q4.coupledDomain().fullDim(),
            //                 Q4.coupledDomain().dim(),
            //                 Q4.coupledCodomain().fullDim(),
            //                 Q4.coupledCodomain().dim());
            // SPDLOG_INFO("Q3: ({}[{}],{}[{}])",
            //                 Q3.coupledDomain().fullDim(),
            //                 Q3.coupledDomain().dim(),
            //                 Q3.coupledCodomain().fullDim(),
            //                 Q3.coupledCodomain().dim());
            std::tie(P1, P2) = decompose(Q3, Q4, chi);
            break;
        }
        case PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    }
    return std::make_pair(P1, P2);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::tuple<Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD>, Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>, Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>>
CTM<Scalar, Symmetry, ENABLE_AD>::renormalize_left(const int x,
                                                   const int y,
                                                   XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>>& P1,
                                                   XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>>& P2,
                                                   bool NORMALIZE) XPED_CONST
{
    Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD> C1_new;
    Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD> T4_new;
    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD> C4_new;
    C1_new = (P1(x, y - 1) * (C1s(x - 1, y - 1).template permute<-1, 0, 1>() * T1s(x, y - 1)).template permute<-2, 0, 2, 3, 1>())
                 .template permute<+1, 0, 1>();
    if(NORMALIZE) C1_new = C1_new * (1. / C1_new.maxNorm());
    C4_new = ((C4s(x - 1, y + 1) * T3s(x, y + 1).template permute<2, 2, 3, 0, 1>()).template permute<0, 1, 0, 2, 3>() * P2(x, y))
                 .template permute<0, 1, 0>();
    if(NORMALIZE) C4_new = C4_new * (1. / C4_new.maxNorm());
    auto tmp2 = P1(x, y).template permute<-2, 0, 2, 3, 1>() * T4s(x - 1, y).template permute<0, 1, 0, 2, 3>();
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 4, 1>() * A->As(x, y).template permute<0, 0, 3, 1, 2, 4>();
    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 3, 1, 6>() * A->Adags(x, y).template permute<0, 0, 4, 2, 1, 3>())
                    .template permute<+1, 0, 3, 5, 1, 2, 4>();
    T4_new = (tmp4 * P2(x, y - 1)).template permute<+2, 3, 0, 1, 2>();
    if(NORMALIZE) T4_new = T4_new * (1. / T4_new.maxNorm());
    return std::make_tuple(C1_new, T4_new, C4_new);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::tuple<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>, Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>, Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD>>
CTM<Scalar, Symmetry, ENABLE_AD>::renormalize_right(const int x,
                                                    const int y,
                                                    XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>>& P1,
                                                    XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>>& P2,
                                                    bool NORMALIZE) XPED_CONST
{
    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD> C2_new;
    Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD> T2_new;
    Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD> C3_new;
    C2_new = (T1s(x, y - 1).template permute<-2, 0, 2, 3, 1>() * C2s(x + 1, y - 1)).template permute<+2, 0, 3, 1, 2>() * P2(x, y - 1);
    if(NORMALIZE) C2_new = C2_new * (1. / C2_new.maxNorm());
    C3_new = (P1(x, y) *
              (C3s(x + 1, y + 1).template permute<+1, 0, 1>() * T3s(x, y + 1).template permute<+2, 3, 2, 0, 1>()).template permute<-2, 0, 2, 3, 1>())
                 .template permute<-1, 0, 1>();
    if(NORMALIZE) C3_new = C3_new * (1. / C3_new.maxNorm());
    auto tmp2 = P1(x, y - 1).template permute<-2, 0, 2, 3, 1>() * T2s(x + 1, y).template permute<+2, 2, 3, 0, 1>();
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A->As(x, y).template permute<0, 1, 2, 0, 3, 4>();
    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A->Adags(x, y).template permute<0, 1, 3, 2, 0, 4>())
                    .template permute<+1, 0, 2, 4, 1, 3, 5>();
    T2_new = (tmp4 * P2(x, y)).template permute<0, 1, 2, 0, 3>();
    if(NORMALIZE) T2_new = T2_new * (1. / T2_new.maxNorm());
    return std::make_tuple(C2_new, T2_new, C3_new);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::tuple<Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD>, Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>, Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>>
CTM<Scalar, Symmetry, ENABLE_AD>::renormalize_top(const int x,
                                                  const int y,
                                                  XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>>& P1,
                                                  XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>>& P2,
                                                  bool NORMALIZE) XPED_CONST
{
    Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD> C1_new;
    Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD> T1_new;
    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD> C2_new;
    C1_new = ((T4s(x - 1, y).template permute<-2, 1, 2, 3, 0>() * C1s(x - 1, y - 1).template permute<-1, 0, 1>()).template permute<+2, 0, 3, 1, 2>() *
              P2(x - 1, y))
                 .template permute<+1, 0, 1>();
    if(NORMALIZE) C1_new = C1_new * (1. / C1_new.maxNorm());
    C2_new = (P1(x, y) * (C2s(x + 1, y - 1) * T2s(x + 1, y).template permute<+2, 2, 0, 1, 3>()).template permute<-2, 0, 1, 2, 3>());
    if(NORMALIZE) C2_new = C2_new * (1. / C2_new.maxNorm());
    auto tmp2 = P1(x - 1, y).template permute<-2, 0, 2, 3, 1>() * T1s(x, y - 1);
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A->As(x, y);
    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A->Adags(x, y)).template permute<+1, 0, 3, 5, 1, 2, 4>();
    T1_new = (tmp4 * P2(x, y)).template permute<+2, 0, 3, 1, 2>();
    if(NORMALIZE) T1_new = T1_new * (1. / T1_new.maxNorm());
    return std::make_tuple(C1_new, T1_new, C2_new);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::tuple<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>, Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>, Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD>>
CTM<Scalar, Symmetry, ENABLE_AD>::renormalize_bottom(const int x,
                                                     const int y,
                                                     XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>>& P1,
                                                     XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>>& P2,
                                                     bool NORMALIZE) XPED_CONST
{
    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD> C4_new;
    Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD> T3_new;
    Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD> C3_new;
    C4_new = (P1(x - 1, y) * (T4s(x - 1, y).template permute<-2, 0, 2, 3, 1>() * C4s(x - 1, y + 1)).template permute<0, 3, 1, 2, 0>())
                 .template permute<0, 1, 0>();
    if(NORMALIZE) C4_new = C4_new * (1. / C4_new.maxNorm());
    C3_new =
        ((T2s(x + 1, y) * C3s(x + 1, y + 1).template permute<+1, 0, 1>()).template permute<+2, 2, 3, 0, 1>() * P2(x, y)).template permute<-1, 0, 1>();
    if(NORMALIZE) C3_new = C3_new * (1. / C3_new.maxNorm());
    auto tmp2 = P1(x, y).template permute<-2, 0, 2, 3, 1>() * T3s(x, y + 1).template permute<+2, 3, 2, 0, 1>();
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A->As(x, y).template permute<0, 2, 3, 0, 1, 4>();

    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A->Adags(x, y).template permute<0, 3, 4, 2, 0, 1>())
                    .template permute<+1, 0, 3, 5, 1, 2, 4>();
    T3_new = (tmp4 * P2(x - 1, y)).template permute<0, 1, 2, 3, 0>();
    if(NORMALIZE) T3_new = T3_new * (1. / T3_new.maxNorm());
    return std::make_tuple(C4_new, T3_new, C3_new);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
Tensor<Scalar, 3, 3, Symmetry, ENABLE_AD> CTM<Scalar, Symmetry, ENABLE_AD>::contractCorner(const int x, const int y, const CORNER corner) XPED_CONST
{
    Tensor<Scalar, 3, 3, Symmetry, ENABLE_AD> Q;
    switch(corner) {
    case CORNER::UPPER_LEFT: {
        auto tmp = C1s(x - 1, y - 1).template permute<-1, 0, 1>() * T1s(x, y - 1);
        auto tmp2 = T4s(x - 1, y).template permute<-2, 1, 2, 3, 0>() * tmp;
        auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A->As(x, y);
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A->Adags(x, y)).template permute<+1, 0, 3, 5, 1, 2, 4>();

        // auto C1T1 = C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1>(T1s(x, y - 1));
        // auto T4C1T1 = T4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3>(C1T1);
        // auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4>(A->As(x, y));
        // auto Qcheck = T4C1T1A.template contract<std::array{-1, 1, -4, 2, -5, -2, 3}, std::array{1, 2, 3, -6, -3}, 3>(A->Adags(x, y));
        // Scalar diff = (Q - Qcheck).norm();
        // SPDLOG_WARN("upper left corner check at x,y={},{}: {}", x, y, diff);
        // ooooo -->--
        // o Q o
        // ooooo ==>==
        // |  ||
        // ^  ^^
        // |  ||
        break;
    }
    case CORNER::LOWER_LEFT: {
        auto tmp = C4s(x - 1, y + 1) * T3s(x, y + 1).template permute<2, 2, 3, 0, 1>();
        auto tmp2 = T4s(x - 1, y).template permute<-2, 0, 2, 3, 1>() * tmp;
        auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A->As(x, y).template permute<0, 0, 3, 1, 2, 4>();
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A->Adags(x, y).template permute<0, 0, 4, 2, 1, 3>())
                .template permute<+1, 1, 3, 5, 0, 2, 4>();

        // auto C4T3 = C4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1>(T3s(x, y + 1));
        // auto T4C4T3 = T4s(x - 1, y).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3>(C4T3);
        // auto T4C4T3A = T4C4T3.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{1, -5, -6, 2, -7}, 4>(A->As(x, y));
        // auto Qcheck = T4C4T3A.template contract<std::array{-4, 1, 2, -1, -5, -2, 3}, std::array{1, -6, 3, -3, 2}, 3>(A->Adags(x, y));
        // Scalar diff = (Q - Qcheck).norm();
        // SPDLOG_WARN("lower left corner check at x,y={},{}: {}", x, y, diff);

        // |  ||
        // ^  ^^
        // |  ||
        // ooooo --<--
        // o Q o
        // ooooo ==<==
        break;
    }
    case CORNER::UPPER_RIGHT: {
        auto tmp = T1s(x, y - 1).template permute<-2, 0, 2, 3, 1>() * C2s(x + 1, y - 1);
        auto tmp2 = tmp * T2s(x + 1, y).template permute<+2, 2, 3, 0, 1>();
        auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A->As(x, y).template permute<0, 1, 2, 0, 3, 4>();
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A->Adags(x, y).template permute<0, 1, 3, 2, 0, 4>())
                .template permute<+1, 0, 2, 4, 1, 3, 5>();

        // auto T1C2 = T1s(x, y - 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3>(C2s(x + 1, y - 1));
        // auto T1C2T2 = T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, 1, -6}, 3>(T2s(x + 1, y));
        // auto T1C2T2A = T1C2T2.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, 1, 2, -6, -7}, 4>(A->As(x, y));
        // auto Qcheck = T1C2T2A.template contract<std::array{-1, 1, 2, -4, -2, -5, 3}, std::array{-3, 1, 3, 2, -6}, 3>(A->Adags(x, y));
        // Scalar diff = (Q - Qcheck).norm();
        // SPDLOG_WARN("upper right corner check at x,y={},{}: {}", x, y, diff);

        // -->--ooooo
        //      o Q o
        // ==>==ooooo
        //      |  ||
        //      v  vv
        //      |  ||

        break;
    }
    case CORNER::LOWER_RIGHT: {
        auto tmp = C3s(x + 1, y + 1).template permute<+1, 0, 1>() * T3s(x, y + 1).template permute<+2, 3, 2, 0, 1>();
        auto tmp2 = T2s(x + 1, y) * tmp;
        auto tmp3 = tmp2.template permute<-1, 2, 1, 3, 5, 0, 4>() * A->As(x, y).template permute<0, 2, 3, 0, 1, 4>();
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A->Adags(x, y).template permute<0, 3, 4, 2, 0, 1>())
                .template permute<+1, 0, 3, 5, 1, 2, 4>();

        // auto C3T3 = C3s(x + 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, -3, -4, 1}, 1>(T3s(x, y + 1));
        // auto T2C3T3 = T2s(x + 1, y).template contract<std::array{-1, -2, -3, 1}, std::array{1, -4, -5, -6}, 3>(C3T3);
        // auto T2C3T3A = T2C3T3.template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, -6, 1, 2, -7}, 4>(A->As(x, y));
        // auto Qcheck = T2C3T3A.template contract<std::array{1, -1, 2, -4, -5, -2, 3}, std::array{-6, -3, 3, 1, 2}, 3>(A->Adags(x, y));
        // Scalar diff = (Q - Qcheck).norm();
        // SPDLOG_WARN("lower right corner check at x,y={},{}: {}", x, y, diff);

        //      |  ||
        //      v  vv
        //      |  ||
        // --<--ooooo
        //      o Q o
        // ==<==ooooo
        break;
    }
    }
    return Q;
}

} // namespace Xped
