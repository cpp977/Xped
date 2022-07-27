#include "Xped/PEPS/LinearAlgebra.hpp"

#include <limits>

#include "spdlog/spdlog.h"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
std::pair<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>>, TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>>>
avg(XPED_CONST CTM<Scalar, Symmetry, ENABLE_AD, CPOpts>& env, XPED_CONST Tensor<Scalar, 2, 2, Symmetry, false>& op)
{
    assert(env.RDM_COMPUTED());
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_h(env.cell().pattern);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_v(env.cell().pattern);
    // PlainInterface::MType<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_h(env.cell().rows(), env.cell().cols());
    // PlainInterface::MType<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_v(env.cell().rows(), env.cell().cols());
    for(int x = 0; x < env.cell().rows(); ++x) {
        for(int y = 0; y < env.cell().cols(); ++y) {
            if(not env.cell().pattern.isUnique(x, y)) { continue; }
            o_h(x, y) = env.rho_h(x, y).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op).trace();
            o_v(x, y) = env.rho_v(x, y).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op).trace();
        }
    }
    return std::make_pair(o_h, o_v);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> avg(XPED_CONST CTM<Scalar, Symmetry, ENABLE_AD, CPOpts>& env,
                                                                    OneSiteObservable<Symmetry>& op)
{
    assert(env.RDM_COMPUTED());
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o(env.cell().pattern);
    o.setConstant(0.);
    if(not op.MEASURE) { return o; }

    auto shifted_op = op.shiftQN(env.Psi()->charges());

    for(int x = 0; x < env.cell().rows(); ++x) {
        for(int y = 0; y < env.cell().cols(); ++y) {
            if(not env.cell().pattern.isUnique(x, y)) { continue; }
            // auto C1T1 = env.C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, ENABLE_AD>(env.T1s(x, y - 1));
            // auto T4C1T1 = env.T4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(C1T1);
            // auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, ENABLE_AD>(env.A->As(x, y));
            // auto T4C1T1AH = T4C1T1A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{1, -7}, 6, ENABLE_AD>(op.data(x, y));
            // auto Q1H = T4C1T1AH.template contract<std::array{-1, 1, -4, 2, -5, -2, 3}, std::array{1, 2, 3, -6, -3}, 3, ENABLE_AD>(env.A->Adags(x,
            // y));

            // auto C2T2 = env.C2s(x + 1, y - 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 3>(env.T2s(x + 1, y));
            // auto C2T2C3 = C2T2.template contract<std::array{-1, -2, -3, 1}, std::array{1, -1}, 3>(env.C3s(x + 1, y + 1));
            // auto C2T2C3T3 = C2T2C3.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, -6, 1}, 3>(env.T3s(x, y + 1));
            // auto C2T2C3T3C4 = C2T2C3T3.template contract<std::array{-1, -2, -3, -5, -6, 1}, std::array{-4, 1}, 3>(env.C4s(x - 1, y + 1));
            // auto Q1 = env.contractCorner(x, y, Opts::CORNER::UPPER_LEFT);
            // // auto norm = (C2T2C3T3C4 * Q1).trace();
            // // o(x, y) = (C2T2C3T3C4 * Q1H).trace() / norm;
            // auto norm = (Q1 * C2T2C3T3C4).trace();
            // o(x, y) = (Q1H * C2T2C3T3C4).trace() / norm;

            o(x, y) = 0.5 * (env.rho1_h(x, y).template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y)).trace() +
                             env.rho1_v(x, y).template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y)).trace());

            if constexpr(ENABLE_AD) {
                op.obs(x, y) = o(x, y).val();
            } else {
                op.obs(x, y) = o(x, y);
            }
        }
    }
    return o;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>>, 4> avg(XPED_CONST CTM<Scalar, Symmetry, ENABLE_AD, CPOpts>& env,
                                                                                   TwoSiteObservable<Symmetry>& op)
{
    assert(env.RDM_COMPUTED());
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_h(env.cell().pattern);
    o_h.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_v(env.cell().pattern);
    o_v.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_d1(env.cell().pattern);
    o_d1.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_d2(env.cell().pattern);
    o_d2.setConstant(0.);
    if(not op.MEASURE) { return std::array{o_h, o_v, o_d1, o_d2}; }

    auto shifted_op = op.shiftQN(env.Psi()->charges());

    for(int x = 0; x < env.cell().rows(); ++x) {
        for(int y = 0; y < env.cell().cols(); ++y) {
            if(not env.cell().pattern.isUnique(x, y)) { continue; }
            if(op.data_h.size() > 0) {
                o_h(x, y) = env.rho_h(x, y).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(shifted_op.data_h(x, y)).trace();
            }
            if(op.data_v.size() > 0) {
                o_v(x, y) = env.rho_v(x, y).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(shifted_op.data_v(x, y)).trace();
            }
            if constexpr(ENABLE_AD) {
                if(op.data_h.size() > 0) { op.obs_h(x, y) = o_h(x, y).val(); }
                if(op.data_v.size() > 0) { op.obs_v(x, y) = o_v(x, y).val(); }
            } else {
                if(op.data_h.size() > 0) { op.obs_h(x, y) = o_h(x, y); }
                if(op.data_v.size() > 0) { op.obs_v(x, y) = o_v(x, y); }
            }
            if(op.data_d1.size() > 0 or op.data_d2.size() > 0) {
                auto Q1 = env.contractCorner(x, y, Opts::CORNER::UPPER_LEFT);
                auto Q2 = env.contractCorner(x + 1, y, Opts::CORNER::UPPER_RIGHT);
                auto Q3 = env.contractCorner(x + 1, y + 1, Opts::CORNER::LOWER_RIGHT);
                auto Q4 = env.contractCorner(x, y + 1, Opts::CORNER::LOWER_LEFT);
                auto norm = (Q1 * Q2 * Q3 * Q4).trace();

                if(op.data_d1.size() > 0) {
                    [[maybe_unused]] double dumb;
                    auto [Hu, Hs, Hvdag] =
                        shifted_op.data_d1(x, y).template permute<0, 0, 2, 1, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 1.e-14, dumb);
                    Hu = Hu * Hs.sqrt();
                    Hvdag = Hs.sqrt() * Hvdag;

                    auto C1T1 =
                        env.C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, ENABLE_AD>(env.T1s(x, y - 1));
                    auto T4C1T1 = env.T4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(C1T1);
                    auto T4C1T1A =
                        T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, ENABLE_AD>(env.A->As(x, y));
                    auto T4C1T1AH = T4C1T1A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{1, -7, -8}, 6, ENABLE_AD>(Hu);
                    auto Q1H = T4C1T1AH.template contract<std::array{-1, 1, -4, 2, -5, -2, 3, -7}, std::array{1, 2, 3, -6, -3}, 3, ENABLE_AD>(
                        env.A->Adags(x, y));

                    auto C3T3 =
                        env.C3s(x + 2, y + 2).template contract<std::array{-1, 1}, std::array{-2, -3, -4, 1}, 1, ENABLE_AD>(env.T3s(x + 1, y + 2));
                    auto T2C3T3 = env.T2s(x + 2, y + 1).template contract<std::array{-1, -2, -3, 1}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(C3T3);
                    auto T2C3T3A = T2C3T3.template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, -6, 1, 2, -7}, 4, ENABLE_AD>(
                        env.A->As(x + 1, y + 1));
                    auto T2C3T3AH = T2C3T3A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{-7, 1, -8}, 6, ENABLE_AD>(Hvdag);
                    auto Q3H = T2C3T3AH.template contract<std::array{1, -1, 2, -4, -5, -2, -7, 3}, std::array{-6, -3, 3, 1, 2}, 3, ENABLE_AD>(
                        env.A->Adags(x + 1, y + 1));
                    Q1H = Q4 * Q1H;
                    Q3H = Q2 * Q3H;
                    o_d1(x, y) =
                        Q1H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{4, 5, 6, 1, 2, 3, 7}, 0, ENABLE_AD>(Q3H).trace() / norm;
                    if constexpr(ENABLE_AD) {
                        op.obs_d1(x, y) = o_d1(x, y).val();
                    } else {
                        op.obs_d1(x, y) = o_d1(x, y);
                    }
                }
                if(op.data_d2.size() > 0) {
                    // Q1 = env.contractCorner(x - 1, y, Opts::CORNER::UPPER_LEFT);
                    // Q2 = env.contractCorner(x, y, Opts::CORNER::UPPER_RIGHT);
                    // Q3 = env.contractCorner(x, y + 1, Opts::CORNER::LOWER_RIGHT);
                    // Q4 = env.contractCorner(x - 1, y + 1, Opts::CORNER::LOWER_LEFT);
                    // norm = (Q1 * Q2 * Q3 * Q4).trace();

                    [[maybe_unused]] double dumb;
                    auto [Hu, Hs, Hvdag] =
                        shifted_op.data_d2(x + 1, y).template permute<0, 0, 2, 1, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 1.e-14, dumb);
                    Hu = Hu * Hs.sqrt();
                    Hvdag = Hs.sqrt() * Hvdag;

                    auto T1C2 =
                        env.T1s(x + 1, y - 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3, ENABLE_AD>(env.C2s(x + 2, y - 1));
                    auto T1C2T2 = T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, 1, -6}, 3, ENABLE_AD>(env.T2s(x + 2, y));
                    auto T1C2T2A =
                        T1C2T2.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, 1, 2, -6, -7}, 4, ENABLE_AD>(env.A->As(x + 1, y));
                    auto T1C2T2AH = T1C2T2A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{-7, 1, -8}, 6, ENABLE_AD>(Hvdag);
                    auto Q2H = T1C2T2AH.template contract<std::array{-1, 1, 2, -4, -2, -5, -7, 3}, std::array{-3, 1, 3, 2, -6}, 3, ENABLE_AD>(
                        env.A->Adags(x + 1, y));

                    auto C4T3 =
                        env.C4s(x - 1, y + 2).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1, ENABLE_AD>(env.T3s(x, y + 2));
                    auto T4C4T3 = env.T4s(x - 1, y + 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(C4T3);
                    auto T4C4T3A =
                        T4C4T3.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{1, -5, -6, 2, -7}, 4, ENABLE_AD>(env.A->As(x, y + 1));
                    auto T4C4T3AH = T4C4T3A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{1, -7, -8}, 6, ENABLE_AD>(Hu);
                    auto Q4H = T4C4T3AH.template contract<std::array{-4, 1, 2, -1, -5, -2, 3, -7}, std::array{1, -6, 3, -3, 2}, 3, ENABLE_AD>(
                        env.A->Adags(x, y + 1));
                    Q2H = Q1 * Q2H;
                    Q4H = Q3 * Q4H;
                    o_d2(x, y) =
                        Q2H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{4, 5, 6, 1, 2, 3, 7}, 0, ENABLE_AD>(Q4H).trace() / norm;
                    // o_d2(x, y) =
                    //     Q2H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{1, 2, 3, 4, 5, 6, 7}, 0, ENABLE_AD>(Q4H).trace() / norm;
                    if constexpr(ENABLE_AD) {
                        op.obs_d2(x, y) = o_d2(x, y).val();
                    } else {
                        op.obs_d2(x, y) = o_d2(x, y);
                    }
                }
            }
        }
    }
    return std::array{o_h, o_v, o_d1, o_d2};
}

} // namespace Xped
