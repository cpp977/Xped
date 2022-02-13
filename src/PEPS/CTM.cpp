#include <iostream>

#include "spdlog/spdlog.h"

#include "Xped/PEPS/CTM.hpp"

#include "Xped/Core/CoeffUnaryOp.hpp"
#include "Xped/PEPS/PEPSContractions.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
void CTM<Scalar, Symmetry>::init(const iPEPS<Scalar, Symmetry>& A)
{
    C1s.resize(cell.pattern);
    C2s.resize(cell.pattern);
    C3s.resize(cell.pattern);
    C4s.resize(cell.pattern);
    T1s.resize(cell.pattern);
    T2s.resize(cell.pattern);
    T3s.resize(cell.pattern);
    T4s.resize(cell.pattern);

    Svs.resize(cell.pattern);

    for(int x = 0; x < cell.Lx; x++) {
        for(int y = 0; y < cell.Ly; y++) {
            if(not cell.pattern.isUnique(x, y)) { continue; }
            auto pos = cell.pattern.uniqueIndex(x, y);
            switch(init_m) {
            case INIT::FROM_TRIVIAL: {
                C1s[pos] = Tensor<Scalar, 0, 2, Symmetry>({{}}, {{Qbasis<Symmetry, 1>::TrivialBasis(), Qbasis<Symmetry, 1>::TrivialBasis()}});
                C1s[pos].setRandom();
                C2s[pos] = Tensor<Scalar, 1, 1, Symmetry>({{Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                C2s[pos].setRandom();
                C3s[pos] = Tensor<Scalar, 2, 0, Symmetry>({{Qbasis<Symmetry, 1>::TrivialBasis(), Qbasis<Symmetry, 1>::TrivialBasis()}}, {{}});
                C3s[pos].setRandom();
                C4s[pos] = Tensor<Scalar, 1, 1, Symmetry>({{Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                C4s[pos].setRandom();
                T1s[pos] = Tensor<Scalar, 1, 3, Symmetry>({{Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                          {{Qbasis<Symmetry, 1>::TrivialBasis(),
                                                            A.ketBasis(x, y + 1, iPEPS<Scalar, Symmetry>::LEG::UP),
                                                            A.braBasis(x, y + 1, iPEPS<Scalar, Symmetry>::LEG::UP)}});
                T1s[pos].setRandom();
                T2s[pos] = Tensor<Scalar, 3, 1, Symmetry>({{A.ketBasis(x - 1, y, iPEPS<Scalar, Symmetry>::LEG::RIGHT),
                                                            A.braBasis(x - 1, y, iPEPS<Scalar, Symmetry>::LEG::RIGHT),
                                                            Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                          {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                T2s[pos].setRandom();
                T3s[pos] = Tensor<Scalar, 3, 1, Symmetry>({{A.ketBasis(x, y - 1, iPEPS<Scalar, Symmetry>::LEG::DOWN),
                                                            A.braBasis(x, y - 1, iPEPS<Scalar, Symmetry>::LEG::DOWN),
                                                            Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                          {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                T3s[pos].setRandom();
                T4s[pos] = Tensor<Scalar, 1, 3, Symmetry>({{Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                          {{Qbasis<Symmetry, 1>::TrivialBasis(),
                                                            A.ketBasis(x + 1, y, iPEPS<Scalar, Symmetry>::LEG::LEFT),
                                                            A.braBasis(x + 1, y, iPEPS<Scalar, Symmetry>::LEG::LEFT)}});
                T4s[pos].setRandom();

                break;
            }
            case INIT::FROM_A: {
                throw("This mode is not implemented");
                break;
            }
            }
        }
    }
}

template <typename Scalar, typename Symmetry>
void CTM<Scalar, Symmetry>::solve(const iPEPS<Scalar, Symmetry>& A)
{
    info();
    for(std::size_t step = 0; step < opts.max_iter; ++step) {
        std::cout << "Step=" << step << std::endl;
        left_move(A);
        right_move(A);
        top_move(A);
        bottom_move(A);
    }
}

template <typename Scalar, typename Symmetry>
void CTM<Scalar, Symmetry>::info() const
{
    std::cout << "CTM(χ=" << chi << "): UnitCell=(" << cell.Lx << "x" << cell.Ly << ")" << std::endl;
    // std::cout << "Tensors:" << std::endl;
    // for(int x = 0; x < cell.Lx; x++) {
    //     for(int y = 0; y < cell.Lx; y++) {
    //         if(not cell.pattern.isUnique(x, y)) {
    //             std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
    //             continue;
    //         }
    //         std::cout << "Cell site: (" << x << "," << y << "), C1:" << std::endl << C1s(x, y) << std::endl << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), C4:" << std::endl << C4s(x, y) << std::endl << std::endl;
    //         std::cout << "Cell site: (" << x << "," << y << "), T4:" << std::endl << T4s(x, y) << std::endl;
    //     }
    // }
}

template <typename Scalar, typename Symmetry>
void CTM<Scalar, Symmetry>::left_move(const iPEPS<Scalar, Symmetry>& A)
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> P1(cell.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> P2(cell.pattern);

    TMatrix<Tensor<Scalar, 0, 2, Symmetry>> C1_new(cell.pattern);
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> T4_new(cell.pattern);
    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> C4_new(cell.pattern);

    C1s.resetChange();
    C4s.resetChange();
    T4s.resetChange();
    SPDLOG_CRITICAL("left move");
    for(int x = 0; x < cell.Lx; x++) {
        for(int y = 0; y < cell.Ly; y++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, A, DIRECTION::LEFT); // move assignment
        }
        for(int y = 0; y < cell.Ly; y++) {
            assert(C1_new.isChanged(x, y - 1) == T4_new.isChanged(x, y) and C1_new.isChanged(x, y - 1) == C4_new.isChanged(x, y + 1));
            if(C1_new.isChanged(x, y - 1)) { continue; }
            std::tie(C1_new(x, y - 1), T4_new(x, y), C4_new(x, y + 1)) = renormalize_left(x, y, A, P1, P2);
        }
        for(int y = 0; y < cell.Ly; y++) {
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

template <typename Scalar, typename Symmetry>
void CTM<Scalar, Symmetry>::right_move(const iPEPS<Scalar, Symmetry>& A)
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> P1(cell.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> P2(cell.pattern);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> C2_new(cell.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> T2_new(cell.pattern);
    TMatrix<Tensor<Scalar, 2, 0, Symmetry>> C3_new(cell.pattern);

    C2s.resetChange();
    C3s.resetChange();
    T2s.resetChange();
    SPDLOG_CRITICAL("right move");
    for(int x = cell.Lx; x >= 0; --x) {
        for(int y = 0; y < cell.Ly; y++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, A, DIRECTION::RIGHT); // move assignment
        }
        for(int y = 0; y < cell.Ly; y++) {
            assert(C2_new.isChanged(x, y - 1) == T2_new.isChanged(x, y) and C2_new.isChanged(x, y - 1) == C3_new.isChanged(x, y + 1));
            if(C2_new.isChanged(x, y - 1)) { continue; }
            std::tie(C2_new(x, y - 1), T2_new(x, y), C3_new(x, y + 1)) = renormalize_right(x, y, A, P1, P2);
        }
        for(int y = 0; y < cell.Ly; y++) {
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

template <typename Scalar, typename Symmetry>
void CTM<Scalar, Symmetry>::top_move(const iPEPS<Scalar, Symmetry>& A)
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> P1(cell.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> P2(cell.pattern);

    TMatrix<Tensor<Scalar, 0, 2, Symmetry>> C1_new(cell.pattern);
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> T1_new(cell.pattern);
    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> C2_new(cell.pattern);

    C1s.resetChange();
    C2s.resetChange();
    T1s.resetChange();
    SPDLOG_CRITICAL("top move");
    for(int y = 0; y < cell.Ly; y++) {
        for(int x = 0; x < cell.Lx; x++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, A, DIRECTION::TOP); // move assignment
        }
        for(int x = 0; x < cell.Lx; x++) {
            assert(C1_new.isChanged(x - 1, y) == C2_new.isChanged(x + 1, y) and C1_new.isChanged(x - 1, y) == T1_new.isChanged(x, y));
            if(C1_new.isChanged(x - 1, y)) { continue; }
            std::tie(C1_new(x - 1, y), T1_new(x, y), C2_new(x + 1, y)) = renormalize_top(x, y, A, P1, P2);
        }
        for(int x = 0; x < cell.Lx; x++) {
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

template <typename Scalar, typename Symmetry>
void CTM<Scalar, Symmetry>::bottom_move(const iPEPS<Scalar, Symmetry>& A)
{
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> P1(cell.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> P2(cell.pattern);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> C4_new(cell.pattern);
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> T3_new(cell.pattern);
    TMatrix<Tensor<Scalar, 2, 0, Symmetry>> C3_new(cell.pattern);

    C4s.resetChange();
    C3s.resetChange();
    T3s.resetChange();
    SPDLOG_CRITICAL("bottom move");
    for(int y = cell.Ly; y >= 0; --y) {
        for(int x = 0; x < cell.Lx; x++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors(x, y, A, DIRECTION::BOTTOM); // move assignment
        }
        for(int x = 0; x < cell.Lx; x++) {
            assert(C4_new.isChanged(x - 1, y) == C3_new.isChanged(x + 1, y) and C4_new.isChanged(x - 1, y) == T3_new.isChanged(x, y));
            if(C4_new.isChanged(x - 1, y)) { continue; }
            std::tie(C4_new(x - 1, y), T3_new(x, y), C3_new(x + 1, y)) = renormalize_bottom(x, y, A, P1, P2);
        }
        for(int x = 0; x < cell.Lx; x++) {
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

template <typename Scalar, typename Symmetry>
std::pair<Tensor<Scalar, 1, 3, Symmetry>, Tensor<Scalar, 3, 1, Symmetry>>
CTM<Scalar, Symmetry>::get_projectors(const int x, const int y, const iPEPS<Scalar, Symmetry>& A, const DIRECTION dir) const
{
    Tensor<Scalar, 1, 3, Symmetry> P1;
    Tensor<Scalar, 3, 1, Symmetry> P2;
    Tensor<Scalar, 3, 3, Symmetry> Q1, Q2, Q3, Q4;
    switch(dir) {
    case DIRECTION::LEFT: {
        switch(proj_m) {
        case PROJECTION::CORNER: {
            Q1 = contractCorner(x, y, A, CORNER::UPPER_LEFT);
            Q4 = contractCorner(x, y + 1, A, CORNER::LOWER_LEFT);
            // SPDLOG_CRITICAL("Q1: ({}[{}],{}[{}])",
            //                 Q1.coupledDomain().fullDim(),
            //                 Q1.coupledDomain().dim(),
            //                 Q1.coupledCodomain().fullDim(),
            //                 Q1.coupledCodomain().dim());
            // SPDLOG_CRITICAL("Q4: ({}[{}],{}[{}])",
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
            Q2 = contractCorner(x, y, A, CORNER::UPPER_RIGHT);
            Q3 = contractCorner(x, y + 1, A, CORNER::LOWER_RIGHT);
            // SPDLOG_CRITICAL("Q2: ({}[{}],{}[{}])",
            //                 Q2.coupledDomain().fullDim(),
            //                 Q2.coupledDomain().dim(),
            //                 Q2.coupledCodomain().fullDim(),
            //                 Q2.coupledCodomain().dim());
            // SPDLOG_CRITICAL("Q3: ({}[{}],{}[{}])",
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
            Q1 = contractCorner(x, y, A, CORNER::UPPER_LEFT);
            Q2 = contractCorner(x + 1, y, A, CORNER::UPPER_RIGHT);
            // SPDLOG_CRITICAL("Q1: ({}[{}],{}[{}])",
            //                 Q1.coupledDomain().fullDim(),
            //                 Q1.coupledDomain().dim(),
            //                 Q1.coupledCodomain().fullDim(),
            //                 Q1.coupledCodomain().dim());
            // SPDLOG_CRITICAL("Q2: ({}[{}],{}[{}])",
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
            Q4 = contractCorner(x, y, A, CORNER::LOWER_LEFT);
            Q3 = contractCorner(x + 1, y, A, CORNER::LOWER_RIGHT);
            // SPDLOG_CRITICAL("Q4: ({}[{}],{}[{}])",
            //                 Q4.coupledDomain().fullDim(),
            //                 Q4.coupledDomain().dim(),
            //                 Q4.coupledCodomain().fullDim(),
            //                 Q4.coupledCodomain().dim());
            // SPDLOG_CRITICAL("Q3: ({}[{}],{}[{}])",
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

template <typename Scalar, typename Symmetry>
std::tuple<Tensor<Scalar, 0, 2, Symmetry>, Tensor<Scalar, 1, 3, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>>
CTM<Scalar, Symmetry>::renormalize_left(const int x,
                                        const int y,
                                        const iPEPS<Scalar, Symmetry>& A,
                                        const TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                                        const TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) const
{
    Tensor<Scalar, 0, 2, Symmetry> C1_new;
    Tensor<Scalar, 1, 3, Symmetry> T4_new;
    Tensor<Scalar, 1, 1, Symmetry> C4_new;
    C1_new = (P1(x, y - 1) * (C1s(x - 1, y - 1).template permute<-1, 0, 1>() * T1s(x, y - 1)).template permute<-2, 0, 2, 3, 1>())
                 .template permute<+1, 0, 1>();
    C1_new = C1_new * (1. / C1_new.norm());
    C4_new = ((C4s(x - 1, y + 1) * T3s(x, y + 1).template permute<2, 2, 3, 0, 1>()).template permute<0, 1, 0, 2, 3>() * P2(x, y))
                 .template permute<0, 1, 0>();
    C4_new = C4_new * (1. / C4_new.norm());
    auto tmp2 = P1(x, y).template permute<-2, 0, 2, 3, 1>() * T4s(x - 1, y).template permute<0, 1, 0, 2, 3>();
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 4, 1>() * A.As(x, y).template permute<0, 0, 3, 1, 2, 4>();
    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 3, 1, 6>() * A.Adags(x, y).template permute<0, 0, 4, 2, 1, 3>())
                    .template permute<+1, 0, 3, 5, 1, 2, 4>();
    T4_new = (tmp4 * P2(x, y - 1)).template permute<+2, 3, 0, 1, 2>();
    T4_new = T4_new * (1. / T4_new.norm());
    return std::make_tuple(C1_new, T4_new, C4_new);
}

template <typename Scalar, typename Symmetry>
std::tuple<Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 3, 1, Symmetry>, Tensor<Scalar, 2, 0, Symmetry>>
CTM<Scalar, Symmetry>::renormalize_right(const int x,
                                         const int y,
                                         const iPEPS<Scalar, Symmetry>& A,
                                         const TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                                         const TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) const
{
    Tensor<Scalar, 1, 1, Symmetry> C2_new;
    Tensor<Scalar, 3, 1, Symmetry> T2_new;
    Tensor<Scalar, 2, 0, Symmetry> C3_new;
    C2_new = (T1s(x, y - 1).template permute<-2, 0, 2, 3, 1>() * C2s(x + 1, y - 1)).template permute<+2, 0, 3, 1, 2>() * P2(x, y - 1);
    C2_new = C2_new * (1. / C2_new.norm());
    C3_new = (P1(x, y) *
              (C3s(x + 1, y + 1).template permute<+1, 0, 1>() * T3s(x, y + 1).template permute<+2, 3, 2, 0, 1>()).template permute<-2, 0, 2, 3, 1>())
                 .template permute<-1, 0, 1>();
    C3_new = C3_new * (1. / C3_new.norm());
    auto tmp2 = P1(x, y - 1).template permute<-2, 0, 2, 3, 1>() * T2s(x + 1, y).template permute<+2, 2, 3, 0, 1>();
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A.As(x, y).template permute<0, 1, 2, 0, 3, 4>();
    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A.Adags(x, y).template permute<0, 1, 3, 2, 0, 4>())
                    .template permute<+1, 0, 2, 4, 1, 3, 5>();
    T2_new = (tmp4 * P2(x, y)).template permute<0, 1, 2, 0, 3>();
    T2_new = T2_new * (1. / T2_new.norm());
    return std::make_tuple(C2_new, T2_new, C3_new);
}

template <typename Scalar, typename Symmetry>
std::tuple<Tensor<Scalar, 0, 2, Symmetry>, Tensor<Scalar, 1, 3, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>>
CTM<Scalar, Symmetry>::renormalize_top(const int x,
                                       const int y,
                                       const iPEPS<Scalar, Symmetry>& A,
                                       const TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                                       const TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) const
{
    Tensor<Scalar, 0, 2, Symmetry> C1_new;
    Tensor<Scalar, 1, 3, Symmetry> T1_new;
    Tensor<Scalar, 1, 1, Symmetry> C2_new;
    C1_new = ((T4s(x - 1, y).template permute<-2, 1, 2, 3, 0>() * C1s(x - 1, y - 1).template permute<-1, 0, 1>()).template permute<+2, 0, 3, 1, 2>() *
              P2(x - 1, y))
                 .template permute<+1, 0, 1>();
    C1_new = C1_new * (1. / C1_new.norm());
    C2_new = (P1(x, y) * (C2s(x + 1, y - 1) * T2s(x + 1, y).template permute<+2, 2, 0, 1, 3>()).template permute<-2, 0, 1, 2, 3>());
    C2_new = C2_new * (1. / C2_new.norm());
    auto tmp2 = P1(x - 1, y).template permute<-2, 0, 2, 3, 1>() * T1s(x, y - 1);
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A.As(x, y);
    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A.Adags(x, y)).template permute<+1, 0, 3, 5, 1, 2, 4>();
    T1_new = (tmp4 * P2(x, y)).template permute<+2, 0, 3, 1, 2>();
    T1_new = T1_new * (1. / T1_new.norm());
    return std::make_tuple(C1_new, T1_new, C2_new);
}

template <typename Scalar, typename Symmetry>
std::tuple<Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 3, 1, Symmetry>, Tensor<Scalar, 2, 0, Symmetry>>
CTM<Scalar, Symmetry>::renormalize_bottom(const int x,
                                          const int y,
                                          const iPEPS<Scalar, Symmetry>& A,
                                          const TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                                          const TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) const
{
    Tensor<Scalar, 1, 1, Symmetry> C4_new;
    Tensor<Scalar, 3, 1, Symmetry> T3_new;
    Tensor<Scalar, 2, 0, Symmetry> C3_new;
    C4_new = (P1(x - 1, y) * (T4s(x - 1, y).template permute<-2, 0, 2, 3, 1>() * C4s(x - 1, y + 1)).template permute<0, 3, 1, 2, 0>())
                 .template permute<0, 1, 0>();
    C4_new = C4_new * (1. / C4_new.norm());
    C3_new =
        ((T2s(x + 1, y) * C3s(x + 1, y + 1).template permute<+1, 0, 1>()).template permute<+2, 2, 3, 0, 1>() * P2(x, y)).template permute<-1, 0, 1>();
    C3_new = C3_new * (1. / C3_new.norm());
    auto tmp2 = P1(x, y).template permute<-2, 0, 2, 3, 1>() * T3s(x, y - 1).template permute<+2, 3, 2, 0, 1>();
    auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A.As(x, y).template permute<0, 2, 3, 0, 1, 4>();

    auto tmp4 = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A.Adags(x, y).template permute<0, 3, 4, 2, 0, 1>())
                    .template permute<+1, 0, 3, 5, 1, 2, 4>();
    T3_new = (tmp4 * P2(x - 1, y)).template permute<0, 1, 2, 3, 0>();
    T3_new = T3_new * (1. / T3_new.norm());
    return std::make_tuple(C4_new, T3_new, C3_new);
}

template <typename Scalar, typename Symmetry>
Tensor<Scalar, 3, 3, Symmetry>
CTM<Scalar, Symmetry>::contractCorner(const int x, const int y, const iPEPS<Scalar, Symmetry>& A, const CORNER corner) const
{
    Tensor<Scalar, 3, 3, Symmetry> Q;
    switch(corner) {
    case CORNER::UPPER_LEFT: {
        auto tmp = C1s(x - 1, y - 1).template permute<-1, 0, 1>() * T1s(x, y - 1);
        auto tmp2 = T4s(x - 1, y).template permute<-2, 1, 2, 3, 0>() * tmp;
        auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A.As(x, y);
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A.Adags(x, y)).template permute<+1, 0, 3, 5, 1, 2, 4>();
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
        auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A.As(x, y).template permute<0, 0, 3, 1, 2, 4>();
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A.Adags(x, y).template permute<0, 0, 4, 2, 1, 3>())
                .template permute<+1, 1, 3, 5, 0, 2, 4>();
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
        auto tmp3 = tmp2.template permute<-1, 0, 2, 3, 5, 1, 4>() * A.As(x, y).template permute<0, 1, 2, 0, 3, 4>();
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A.Adags(x, y).template permute<0, 1, 3, 2, 0, 4>())
                .template permute<+1, 0, 2, 4, 1, 3, 5>();
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
        auto tmp3 = tmp2.template permute<-1, 2, 1, 3, 5, 0, 4>() * A.As(x, y).template permute<0, 2, 3, 0, 1, 4>();
        Q = (tmp3.template permute<0, 0, 2, 4, 5, 1, 3, 6>() * A.Adags(x, y).template permute<0, 3, 4, 2, 0, 1>())
                .template permute<+1, 0, 3, 5, 1, 2, 4>();
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
