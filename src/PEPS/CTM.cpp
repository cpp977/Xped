#include <array>
#include <iostream>

#include "spdlog/spdlog.h"

#include <highfive/H5File.hpp>

#include "Xped/PEPS/CTM.hpp"

#include "Xped/Core/CoeffUnaryOp.hpp"
#include "Xped/IO/Matlab.hpp"
#include "Xped/PEPS/PEPSContractions.hpp"

#include "Xped/AD/reverse_pass_callback_alloc.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::CTM(const CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, false>& other)
{
    cell_ = other.cell();
    chi_ = other.chi_;
    // init_m = other.init_m;
    // proj_m = other.proj_m;
    // opts = other.opts;
    HAS_RDM = false;

    if(other.A != nullptr) { A = std::make_shared<iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>>(*other.A); }
    CA1s = other.CA1s;
    CA2s = other.CA2s;
    CA3s = other.CA3s;
    CA4s = other.CA4s;
    TA1s = other.TA1s;
    TA2s = other.TA2s;
    TA3s = other.TA3s;
    TA4s = other.TA4s;

    if constexpr(TRank == 1) {
        Ms.resize(cell_.pattern);
        if(other.A != nullptr) { computeMs(); }
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::init()
{
    CA1s.resize(cell_.pattern);
    CA2s.resize(cell_.pattern);
    CA3s.resize(cell_.pattern);
    CA4s.resize(cell_.pattern);
    TA1s.resize(cell_.pattern);
    TA2s.resize(cell_.pattern);
    TA3s.resize(cell_.pattern);
    TA4s.resize(cell_.pattern);

    Svs.resize(cell_.pattern);

    if constexpr(TRank == 1) {
        SPDLOG_CRITICAL("Start M computation.");
        Ms.resize(cell_.pattern);
        computeMs();
        SPDLOG_CRITICAL("Finished M computation.");
    }

    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);
            switch(init_m) {
            case Opts::CTM_INIT::FROM_TRIVIAL: {
                CA1s[pos] =
                    Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD>({{}}, {{Qbasis<Symmetry, 1>::TrivialBasis(), Qbasis<Symmetry, 1>::TrivialBasis()}});
                CA1s[pos].setIdentity();
                CA2s[pos] =
                    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                CA2s[pos].setIdentity();
                CA3s[pos] =
                    Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis(), Qbasis<Symmetry, 1>::TrivialBasis()}}, {{}});
                CA3s[pos].setIdentity();
                CA4s[pos] =
                    Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                CA4s[pos].setIdentity();
                if constexpr(TRank == 2) /*Stadard bra ket case */ {
                    TA1s[pos] = Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>(
                        {{Qbasis<Symmetry, 1>::TrivialBasis()}},
                        {{Qbasis<Symmetry, 1>::TrivialBasis(), A->ketBasis(x, y + 1, Opts::Leg::Top), A->braBasis(x, y + 1, Opts::Leg::Top)}});
                    TA1s[pos].setIdentity();
                    TA2s[pos] = Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>(
                        {{A->ketBasis(x - 1, y, Opts::Leg::Right), A->braBasis(x - 1, y, Opts::Leg::Right), Qbasis<Symmetry, 1>::TrivialBasis()}},
                        {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                    TA2s[pos].setIdentity();
                    TA3s[pos] = Tensor<Scalar, 3, 1, Symmetry, ENABLE_AD>(
                        {{A->ketBasis(x, y - 1, Opts::Leg::Bottom), A->braBasis(x, y - 1, Opts::Leg::Bottom), Qbasis<Symmetry, 1>::TrivialBasis()}},
                        {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                    TA3s[pos].setIdentity();
                    TA4s[pos] = Tensor<Scalar, 1, 3, Symmetry, ENABLE_AD>(
                        {{Qbasis<Symmetry, 1>::TrivialBasis()}},
                        {{Qbasis<Symmetry, 1>::TrivialBasis(), A->ketBasis(x + 1, y, Opts::Leg::Left), A->braBasis(x + 1, y, Opts::Leg::Left)}});
                    TA4s[pos].setIdentity();
                } else if constexpr(TRank == 1) {
                    TA1s[pos] = Tensor<Scalar, 1, 2, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                                          {{Qbasis<Symmetry, 1>::TrivialBasis(), Ms(x, y + 1).uncoupledDomain()[1]}});
                    TA1s[pos].setIdentity();
                    TA2s[pos] = Tensor<Scalar, 2, 1, Symmetry, ENABLE_AD>(
                        {{Ms(x - 1, y).uncoupledCodomain()[0], Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                    TA2s[pos].setIdentity();
                    TA3s[pos] = Tensor<Scalar, 2, 1, Symmetry, ENABLE_AD>(
                        {{Ms(x, y - 1).uncoupledCodomain()[1], Qbasis<Symmetry, 1>::TrivialBasis()}}, {{Qbasis<Symmetry, 1>::TrivialBasis()}});
                    TA3s[pos].setIdentity();
                    TA4s[pos] = Tensor<Scalar, 1, 2, Symmetry, ENABLE_AD>({{Qbasis<Symmetry, 1>::TrivialBasis()}},
                                                                          {{Qbasis<Symmetry, 1>::TrivialBasis(), Ms(x + 1, y).uncoupledDomain()[0]}});
                    TA4s[pos].setIdentity();
                }
                break;
            }
            case Opts::CTM_INIT::FROM_A: {
                auto fuse_ll = Tensor<Scalar, 1, 2, Symmetry, false>::Identity(
                    {{A->ketBasis(x, y, Opts::Leg::Left).combine(A->braBasis(x, y, Opts::Leg::Left)).forgetHistory()}},
                    {{A->ketBasis(x, y, Opts::Leg::Left), A->braBasis(x, y, Opts::Leg::Left)}},
                    A->As(x, y).world());
                auto fuse_uu = Tensor<Scalar, 1, 2, Symmetry, false>::Identity(
                    {{A->ketBasis(x, y, Opts::Leg::Top).combine(A->braBasis(x, y, Opts::Leg::Top)).forgetHistory()}},
                    {{A->ketBasis(x, y, Opts::Leg::Top), A->braBasis(x, y, Opts::Leg::Top)}},
                    A->As(x, y).world());
                auto fuse_rr = Tensor<Scalar, 2, 1, Symmetry, false>::Identity(
                    {{A->ketBasis(x, y, Opts::Leg::Right), A->braBasis(x, y, Opts::Leg::Right)}},
                    {{A->ketBasis(x, y, Opts::Leg::Right).combine(A->braBasis(x, y, Opts::Leg::Right)).forgetHistory()}},
                    A->As(x, y).world());
                auto fuse_dd = Tensor<Scalar, 2, 1, Symmetry, false>::Identity(
                    {{A->ketBasis(x, y, Opts::Leg::Bottom), A->braBasis(x, y, Opts::Leg::Bottom)}},
                    {{A->ketBasis(x, y, Opts::Leg::Bottom).combine(A->braBasis(x, y, Opts::Leg::Bottom)).forgetHistory()}},
                    A->As(x, y).world());

                CA1s[pos] = A->As[pos]
                                .template contract<std::array{1, 2, -1, -2, 3}, std::array{1, 2, 3, -3, -4}, 2>(A->Adags[pos].twist(3).twist(4))
                                .template contract<std::array{1, -1, 2, -2}, std::array{1, 2, -3}, 2>(fuse_rr)
                                .template contract<std::array{1, 2, -2}, std::array{1, 2, -1}, 0>(fuse_dd);
                CA2s[pos] = A->As[pos]
                                .template contract<std::array{-1, 1, 2, -2, 3}, std::array{-3, 1, 3, 2, -4}, 2>(A->Adags[pos].twist(4))
                                .template contract<std::array{1, -1, 2, -2}, std::array{-3, 1, 2}, 2>(fuse_ll.twist(1).twist(2))
                                .template contract<std::array{1, 2, -1}, std::array{1, 2, -2}, 1>(fuse_dd);
                CA3s[pos] = A->As[pos]
                                .template contract<std::array{-1, -2, 1, 2, 3}, std::array{-3, -4, 3, 1, 2}, 2>(A->Adags[pos])
                                .template contract<std::array{1, -1, 2, -2}, std::array{-3, 1, 2}, 2>(fuse_ll.twist(1).twist(2))
                                .template contract<std::array{1, 2, -2}, std::array{-1, 1, 2}, 2>(fuse_uu.twist(1).twist(2));
                CA4s[pos] = A->As[pos]
                                .template contract<std::array{1, -1, -2, 2, 3}, std::array{1, -3, 3, -4, 2}, 2>(A->Adags[pos].twist(3))
                                .template contract<std::array{1, -1, 2, -2}, std::array{-3, 1, 2}, 2>(fuse_uu.twist(1).twist(2))
                                .template contract<std::array{1, 2, -1}, std::array{1, 2, -2}, 1>(fuse_rr);

                if constexpr(TRank == 2) {
                    TA1s[pos] = A->As[pos]
                                    .template contract<std::array{-1, 1, -2, -3, 2}, std::array{-4, 1, 2, -5, -6}, 3>(A->Adags[pos].twist(3).twist(4))
                                    .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_ll.twist(1).twist(2))
                                    .template contract<std::array{1, -3, 2, -4, -1}, std::array{1, 2, -2}, 1>(fuse_rr);
                    TA2s[pos] = A->As[pos]
                                    .template contract<std::array{-1, -2, 1, -3, 2}, std::array{-4, -5, 2, 1, -6}, 3>(A->Adags[pos].twist(4))
                                    .template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{-5, 1, 2}, 4>(fuse_uu.twist(1).twist(2))
                                    .template contract<std::array{-1, 1, -2, 2, -3}, std::array{1, 2, -4}, 3>(fuse_dd);
                    TA3s[pos] = A->As[pos]
                                    .template contract<std::array{-1, -2, -3, 1, 2}, std::array{-4, -5, 2, -6, 1}, 3>(A->Adags[pos].twist(3))
                                    .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_ll.twist(1).twist(2))
                                    .template contract<std::array{-1, 1, -2, 2, -3}, std::array{1, 2, -4}, 3>(fuse_rr);
                    TA4s[pos] = A->As[pos]
                                    .template contract<std::array{1, -1, -2, -3, 2}, std::array{1, -4, 2, -5, -6}, 3>(A->Adags[pos].twist(3).twist(4))
                                    .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_uu.twist(1).twist(2))
                                    .template contract<std::array{-3, 1, -4, 2, -1}, std::array{1, 2, -2}, 1>(fuse_dd);
                } else if constexpr(TRank == 1) {
                    TA1s[pos] = A->As[pos]
                                    .template contract<std::array{-1, 1, -2, -3, 2}, std::array{-4, 1, 2, -5, -6}, 3>(A->Adags[pos].twist(3).twist(4))
                                    .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_ll.twist(1).twist(2))
                                    .template contract<std::array{1, -3, 2, -4, -1}, std::array{1, 2, -2}, 1>(fuse_rr)
                                    .template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3}, 1>(fuse_dd);

                    TA2s[pos] = A->As[pos]
                                    .template contract<std::array{-1, -2, 1, -3, 2}, std::array{-4, -5, 2, 1, -6}, 3>(A->Adags[pos].twist(4))
                                    .template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{-5, 1, 2}, 4>(fuse_uu.twist(1).twist(2))
                                    .template contract<std::array{-1, 1, -2, 2, -3}, std::array{1, 2, -4}, 3>(fuse_dd)
                                    .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 2>(fuse_ll.twist(1).twist(2));
                    TA3s[pos] = A->As[pos]
                                    .template contract<std::array{-1, -2, -3, 1, 2}, std::array{-4, -5, 2, -6, 1}, 3>(A->Adags[pos].twist(3))
                                    .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_ll.twist(1).twist(2))
                                    .template contract<std::array{-1, 1, -2, 2, -3}, std::array{1, 2, -4}, 3>(fuse_rr)
                                    .template contract<std::array{1, 2, -2, -3}, std::array{-1, 1, 2}, 2>(fuse_uu.twist(1).twist(2));
                    TA4s[pos] = A->As[pos]
                                    .template contract<std::array{1, -1, -2, -3, 2}, std::array{1, -4, 2, -5, -6}, 3>(A->Adags[pos].twist(3).twist(4))
                                    .template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, 1, 2}, 4>(fuse_uu.twist(1).twist(2))
                                    .template contract<std::array{-3, 1, -4, 2, -1}, std::array{1, 2, -2}, 1>(fuse_dd)
                                    .template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3}, 1>(fuse_rr);
                }
                break;
            }
            }
        }
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::loadFromMatlab(const std::filesystem::path& p,
                                                                                   const std::string& root_name,
                                                                                   int qn_scale)
{
    HighFive::File file(p.string(), HighFive::File::ReadOnly);
    auto root = file.getGroup(root_name);

    cell_.loadFromMatlab(p, root_name);

    CA1s.resize(cell_.pattern);
    CA2s.resize(cell_.pattern);
    CA3s.resize(cell_.pattern);
    CA4s.resize(cell_.pattern);
    TA1s.resize(cell_.pattern);
    TA2s.resize(cell_.pattern);
    TA3s.resize(cell_.pattern);
    TA4s.resize(cell_.pattern);

    Svs.resize(cell_.pattern);
    if constexpr(TRank == 2) {
        for(std::size_t i = 0; i < cell_.pattern.uniqueSize(); ++i) {
            auto C1_ref = root.getDataSet("C1");
            std::vector<HighFive::Reference> C1;
            C1_ref.read(C1);
            auto g_C1_0 = C1[i].template dereference<HighFive::Group>(root);
            CA1s[i] = Xped::IO::loadMatlabTensor<double, 2, 0, Symmetry, Xped::HeapPolicy>(g_C1_0, root, std::array{true, true}, qn_scale)
                          .template permute<2, 1, 0>();

            auto C2_ref = root.getDataSet("C2");
            std::vector<HighFive::Reference> C2;
            C2_ref.read(C2);
            auto g_C2_0 = C2[i].template dereference<HighFive::Group>(root);
            CA2s[i] = Xped::IO::loadMatlabTensor<double, 2, 0, Symmetry, Xped::HeapPolicy>(g_C2_0, root, std::array{false, true}, qn_scale)
                          .template permute<1, 0, 1>();

            auto C3_ref = root.getDataSet("C3");
            std::vector<HighFive::Reference> C3;
            C3_ref.read(C3);
            auto g_C3_0 = C3[i].template dereference<HighFive::Group>(root);
            CA3s[i] = Xped::IO::loadMatlabTensor<double, 2, 0, Symmetry, Xped::HeapPolicy>(g_C3_0, root, std::array{false, false}, qn_scale);

            auto C4_ref = root.getDataSet("C4");
            std::vector<HighFive::Reference> C4;
            C4_ref.read(C4);
            auto g_C4_0 = C4[i].template dereference<HighFive::Group>(root);
            CA4s[i] = Xped::IO::loadMatlabTensor<double, 2, 0, Symmetry, Xped::HeapPolicy>(g_C4_0, root, std::array{true, false}, qn_scale)
                          .template permute<1, 1, 0>();

            auto T1_ref = root.getDataSet("T1");
            std::vector<HighFive::Reference> T1;
            T1_ref.read(T1);
            auto g_T1_0 = T1[i].template dereference<HighFive::Group>(root);
            TA1s[i] = Xped::IO::loadMatlabTensor<double, TRank + 2, 0, Symmetry, Xped::HeapPolicy>(
                          g_T1_0, root, std::array{true, false, true, false}, qn_scale)
                          .template permute<TRank + 1, 1, 0, 2, 3>();

            auto T2_ref = root.getDataSet("T2");
            std::vector<HighFive::Reference> T2;
            T2_ref.read(T2);
            auto g_T2_0 = T2[i].template dereference<HighFive::Group>(root);
            TA2s[i] = Xped::IO::loadMatlabTensor<double, TRank + 2, 0, Symmetry, Xped::HeapPolicy>(
                          g_T2_0, root, std::array{false, true, false, true}, qn_scale)
                          .template permute<1, 2, 3, 0, 1>();

            auto T3_ref = root.getDataSet("T3");
            std::vector<HighFive::Reference> T3;
            T3_ref.read(T3);
            auto g_T3_0 = T3[i].template dereference<HighFive::Group>(root);
            TA3s[i] = Xped::IO::loadMatlabTensor<double, TRank + 2, 0, Symmetry, Xped::HeapPolicy>(
                          g_T3_0, root, std::array{true, false, false, true}, qn_scale)
                          .template permute<1, 2, 3, 1, 0>();

            auto T4_ref = root.getDataSet("T4");
            std::vector<HighFive::Reference> T4;
            T4_ref.read(T4);
            auto g_T4_0 = T4[i].template dereference<HighFive::Group>(root);
            TA4s[i] = Xped::IO::loadMatlabTensor<double, TRank + 2, 0, Symmetry, Xped::HeapPolicy>(
                          g_T4_0, root, std::array{false, true, true, false}, qn_scale)
                          .template permute<TRank + 1, 0, 1, 2, 3>();
        }
    } else {
        init();
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::computeMs()
{
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);

            auto fuse_ll = Tensor<Scalar, 1, 2, Symmetry, false>::Identity(
                {{A->ketBasis(x, y, Opts::Leg::Left).combine(A->braBasis(x, y, Opts::Leg::Left)).forgetHistory()}},
                {{A->ketBasis(x, y, Opts::Leg::Left), A->braBasis(x, y, Opts::Leg::Left)}},
                A->As(x, y).world());
            auto fuse_tt = Tensor<Scalar, 1, 2, Symmetry, false>::Identity(
                {{A->ketBasis(x, y, Opts::Leg::Top).combine(A->braBasis(x, y, Opts::Leg::Top)).forgetHistory()}},
                {{A->ketBasis(x, y, Opts::Leg::Top), A->braBasis(x, y, Opts::Leg::Top)}},
                A->As(x, y).world());
            auto fuse_rr = Tensor<Scalar, 2, 1, Symmetry, false>::Identity(
                {{A->ketBasis(x, y, Opts::Leg::Right), A->braBasis(x, y, Opts::Leg::Right)}},
                {{A->ketBasis(x, y, Opts::Leg::Right).combine(A->braBasis(x, y, Opts::Leg::Right)).forgetHistory()}},
                A->As(x, y).world());
            auto fuse_dd = Tensor<Scalar, 2, 1, Symmetry, false>::Identity(
                {{A->ketBasis(x, y, Opts::Leg::Bottom), A->braBasis(x, y, Opts::Leg::Bottom)}},
                {{A->ketBasis(x, y, Opts::Leg::Bottom).combine(A->braBasis(x, y, Opts::Leg::Bottom)).forgetHistory()}},
                A->As(x, y).world());
            Ms[pos] = A->As[pos]
                          .template contract<std::array{-1, -2, -3, -4, 1}, std::array{-5, -6, 1, -7, -8}, 8, TRACK>(A->Adags[pos].twist(3).twist(4))
                          .template contract<std::array{1, -1, -2, -3, 2, -4, -5, -6}, std::array{-7, 1, 2}, 6, TRACK>(fuse_ll.twist(1).twist(2))
                          .template contract<std::array{1, -1, -2, 2, -3, -4, -5}, std::array{-6, 1, 2}, 5, TRACK>(fuse_tt.twist(1).twist(2))
                          .template contract<std::array{1, -1, 2, -2, -3, -4}, std::array{1, 2, -5}, 4, TRACK>(fuse_rr)
                          .template contract<std::array{1, 2, -1, -2, -3}, std::array{1, 2, -4}, 2, TRACK>(fuse_dd);
        }
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::solve(std::size_t steps)
{
    // info();
    // stan::math::print_stack(std::cout);
    for(std::size_t step = 0; step < steps; ++step) {
        // SPDLOG_CRITICAL("Step={}", step);
        grow_all<TRACK>();
    }
    computeRDM<true>();
    // stan::math::print_stack(std::cout);
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::grow_all()
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;
    [[maybe_unused]] auto curr = *this;
    left_move<TRACK_INNER>();
    right_move<TRACK_INNER>();
    top_move<TRACK_INNER>();
    bottom_move<TRACK_INNER>();

    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([curr_ = curr, res = *this]() mutable {
            stan::math::nested_rev_autodiff nested;
            curr_.template grow_all<TRACK, false>();
            for(auto i = 0ul; i < curr_.cell().uniqueSize(); ++i) {
                curr_.CA1s[i].adj() = res.CA1s[i].adj();
                curr_.CA2s[i].adj() = res.CA2s[i].adj();
                curr_.CA3s[i].adj() = res.CA3s[i].adj();
                curr_.CA4s[i].adj() = res.CA4s[i].adj();
                curr_.TA1s[i].adj() = res.TA1s[i].adj();
                curr_.TA2s[i].adj() = res.TA2s[i].adj();
                curr_.TA3s[i].adj() = res.TA3s[i].adj();
                curr_.TA4s[i].adj() = res.TA4s[i].adj();
            }
            // curr.template left_move<true>();
            // curr.template right_move<true>();
            // curr.template top_move<true>();
            // curr.template bottom_move<true>();
            stan::math::grad();
        });
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::computeRDM()
{
    computeRDM_h<TRACK>();
    computeRDM_v<TRACK>();
    // computeRDM_1s<TRACK>();
    HAS_RDM = true;
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
auto CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::info() const
{
    // return fmt::format("CTM(χ={}, {}): UnitCell=({}x{}), init={}", chi_, Symmetry::name(), cell_.Lx, cell_.Ly, fmt::streamed(init_m));
    // std::cout << "CTM(χ=" << chi << "): UnitCell=(" << cell_.Lx << "x" << cell_.Ly << ")"
    //           << ", init=" << mode_string << std::endl;
    std::cout << "Tensors:" << std::endl;
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) {
                std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
                continue;
            }
            std::cout << "Cell site: (" << x << "," << y << "), C1: " << CA1s(x, y) << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), C2: " << CA2s(x, y) << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), C3: " << CA3s(x, y) << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), C4: " << CA4s(x, y) << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), T1: " << TA1s(x, y) << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), T2: " << TA2s(x, y) << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), T3: " << TA3s(x, y) << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), T4: " << TA4s(x, y) << std::endl << std::endl;
        }
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::checkHermiticity() const
{
    if(not RDM_COMPUTED()) { return; }
    for(int x = 0; x < cell_.Lx; ++x) {
        for(int y = 0; y < cell_.Ly; ++y) {
            if(rho_h_hermitian_check(x, y) > 1.e-5 or rho_v_hermitian_check(x, y) > 1.e-5) {
                Log::warning(Log::globalLevel,
                             "Nonhermitian reduced density-matrix! Site: x={},y={}, err_h={:2.10g}, err_v={:2.10g}",
                             x,
                             y,
                             rho_h_hermitian_check(x, y),
                             rho_v_hermitian_check(x, y));
            }
        }
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
bool CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::checkConvergence(typename ScalarTraits<Scalar>::Real epsilon)
{
    for(int x = 0; x < cell_.Lx; ++x) {
        for(int y = 0; y < cell_.Ly; ++y) {
            auto C1T4 = CA1s(x, y - 1).template contract<std::array{1, -1}, std::array{1, -2, -3, -4}, 1>(TA4s(x, y));
            auto C1T4C4 = C1T4.template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 2>(CA4s(x, y + 1));

            auto corner = contractCorner(x, y, Opts::CORNER::UPPER_LEFT);
            auto cornerC4 = corner.template contract<std::array{1, -1, -2, -3, -4, -5}, std::array{1, -6}, 5>(CA4s(x - 1, y + 1));
            auto C1T4C4check = cornerC4.template contract<std::array{1, 2, -1, -2, -3, 3}, std::array{1, 2, 3, -4}, 2>(TA3s(x, y + 1));
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

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::computeRDM_h()
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;
    auto curr = *this;

    SPDLOG_INFO("Compute rho_h.");
    rho_h = TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>>(cell_.pattern);
    rho1_h = TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>>(cell_.pattern);
    rho_h_hermitian_check = TMatrix<double>(cell_.pattern);
    rho1_h_hermitian_check = TMatrix<double>(cell_.pattern);

    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(rho_h.isChanged(x, y)) { continue; }
            if constexpr(TRank == 2) {
                auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, TRACK_INNER>(TA1s(x, y - 1));
                auto T4C1T1 = TA4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(
                    C1T1.template twist<TRACK_INNER>(0));
                auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(A->As(x, y));
                auto C4T3 = CA4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1, TRACK_INNER>(TA3s(x, y + 1));
                auto C4T3Ad = C4T3.template contract<std::array{-1, -2, 1, -3}, std::array{-4, -5, -6, -7, 1}, 3, TRACK_INNER>(
                    A->Adags(x, y).template twist<TRACK_INNER>(3));
                auto left_half =
                    T4C1T1A.template contract<std::array{1, 2, -1, 3, -2, 4, -3}, std::array{1, 4, -4, 2, 3, -5, -6}, 3, TRACK_INNER>(C4T3Ad);

                auto T1C2 = TA1s(x + 1, y - 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3, TRACK_INNER>(CA2s(x + 2, y - 1));
                auto T1C2T2 = T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, 1, -6}, 3, TRACK_INNER>(TA2s(x + 2, y));
                auto AT1C2T2 = A->As(x + 1, y)
                                   .template twist<TRACK_INNER>(1)
                                   .template contract<std::array{-1, 1, 2, -2, -3}, std::array{-4, 1, -5, 2, -6, -7}, 3, TRACK_INNER>(T1C2T2);
                auto T3C3 = TA3s(x + 1, y + 1).template contract<std::array{-1, -2, -3, 1}, std::array{-4, 1}, 3, TRACK_INNER>(CA3s(x + 2, y + 1));
                auto T3C3Ad = T3C3.template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, -6, -7, 1}, 3, TRACK_INNER>(A->Adags(x + 1, y));
                auto right_half =
                    AT1C2T2.template contract<std::array{-1, 1, -2, -3, 2, 3, 4}, std::array{1, -4, 4, -5, 2, -6, 3}, 3, TRACK_INNER>(T3C3Ad);
                rho_h(x, y) = left_half.template contract<std::array{1, 2, -3, 3, -1, 4}, std::array{2, -4, 1, 3, 4, -2}, 2, TRACK_INNER>(right_half);
            } else if constexpr(TRank == 1) {
                auto get_fuse = [this](int x, int y, Opts::Leg leg) {
                    return Tensor<Scalar, 2, 1, Symmetry, false>::Identity({{A->ketBasis(x, y, leg), A->braBasis(x, y, leg)}},
                                                                           {{A->ketBasis(x, y, leg).combine(A->braBasis(x, y, leg)).forgetHistory()}},
                                                                           A->As(x, y).world());
                };
                auto get_split = [this](int x, int y, Opts::Leg leg) {
                    return Tensor<Scalar, 1, 2, Symmetry, false>::Identity({{A->ketBasis(x, y, leg).combine(A->braBasis(x, y, leg)).forgetHistory()}},
                                                                           {{A->ketBasis(x, y, leg), A->braBasis(x, y, leg)}},
                                                                           A->As(x, y).world());
                };

                auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3}, 1, TRACK_INNER>(TA1s(x, y - 1));
                auto T4C1T1_ = TA4s(x - 1, y).template contract<std::array{1, -1, -2}, std::array{1, -4, -5}, 2, TRACK_INNER>(
                    C1T1.template twist<TRACK_INNER>(0));
                auto T4C1T1 = T4C1T1_
                                  .template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5}, 3, TRACK_INNER>(
                                      get_fuse(x - 1, y, Opts::Leg::Right).adjoint().eval())
                                  .template contract<std::array{-1, -4, 1, -2, -3}, std::array{1, -5, -6}, 3, TRACK_INNER>(
                                      get_fuse(x, y - 1, Opts::Leg::Bottom).adjoint().eval());
                auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(A->As(x, y));
                auto C4T3_ = CA4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, 1, -3}, 1, TRACK_INNER>(TA3s(x, y + 1));
                auto C4T3 = C4T3_.template contract<std::array{-1, 1, -4}, std::array{-2, -3, 1}, 1, TRACK_INNER>(
                    get_split(x, y + 1, Opts::Leg::Top).adjoint().eval());
                auto C4T3Ad = C4T3.template contract<std::array{-1, -2, 1, -3}, std::array{-4, -5, -6, -7, 1}, 3, TRACK_INNER>(
                    A->Adags(x, y).template twist<TRACK_INNER>(3));
                auto left_half =
                    T4C1T1A.template contract<std::array{1, 2, -1, 3, -2, 4, -3}, std::array{1, 4, -4, 2, 3, -5, -6}, 3, TRACK_INNER>(C4T3Ad);

                auto T1C2 = TA1s(x + 1, y - 1).template contract<std::array{-1, 1, -2}, std::array{1, -3}, 2, TRACK_INNER>(CA2s(x + 2, y - 1));
                auto T1C2T2_ = T1C2.template contract<std::array{-1, -2, 1}, std::array{-3, 1, -4}, 2, TRACK_INNER>(TA2s(x + 2, y));
                auto T1C2T2 = T1C2T2_
                                  .template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5}, 3, TRACK_INNER>(
                                      get_fuse(x + 1, y - 1, Opts::Leg::Bottom).adjoint().eval())
                                  .template contract<std::array{-1, 1, -6, -2, -3}, std::array{-4, -5, 1}, 3, TRACK_INNER>(
                                      get_split(x + 2, y, Opts::Leg::Left).adjoint().eval());
                auto AT1C2T2 = A->As(x + 1, y)
                                   .template twist<TRACK_INNER>(1)
                                   .template contract<std::array{-1, 1, 2, -2, -3}, std::array{-4, 1, -5, 2, -6, -7}, 3, TRACK_INNER>(T1C2T2);
                auto T3C3_ = TA3s(x + 1, y + 1).template contract<std::array{-1, -2, 1}, std::array{-4, 1}, 2, TRACK_INNER>(CA3s(x + 2, y + 1));
                auto T3C3 = T3C3_.template contract<std::array{1, -3, -4}, std::array{-1, -2, 1}, 3, TRACK_INNER>(
                    get_split(x + 1, y + 1, Opts::Leg::Top).adjoint().eval());
                auto T3C3Ad = T3C3.template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, -6, -7, 1}, 3, TRACK_INNER>(A->Adags(x + 1, y));
                auto right_half =
                    AT1C2T2.template contract<std::array{-1, 1, -2, -3, 2, 3, 4}, std::array{1, -4, 4, -5, 2, -6, 3}, 3, TRACK_INNER>(T3C3Ad);
                rho_h(x, y) = left_half.template contract<std::array{1, 2, -3, 3, -1, 4}, std::array{2, -4, 1, 3, 4, -2}, 2, TRACK_INNER>(right_half);
            }
            auto Id2 =
                Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rho_h(x, y).uncoupledCodomain(), rho_h(x, y).uncoupledDomain(), rho_h(x, y).world());
            auto norm = rho_h(x, y)
                            .template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0, TRACK_INNER>(Id2.twist(0).twist(1))
                            .template trace<TRACK_INNER>();
            if(std::real(norm) < 0.) { Log::warning(Log::globalLevel, "RDM_h: Negative norm detected."); }
            rho_h(x, y) = operator*<TRACK_INNER>(rho_h(x, y), (1. / norm));
            if constexpr(not ENABLE_AD) {
                rho_h_hermitian_check(x, y) = (rho_h(x, y) - rho_h(x, y).adjoint()).norm() / (rho_h(x, y) + rho_h(x, y).adjoint()).norm();
            }
            // fmt::print("rho_h at ({},{}):\n", x, y);
            // rho_h(x, y).print(std::cout, true);
            // std::cout << std::endl;
            auto Id = Tensor<Scalar, 1, 1, Symmetry, false>::Identity(
                {{rho_h(x, y).uncoupledCodomain()[1]}}, {{rho_h(x, y).uncoupledDomain()[1]}}, rho_h(x, y).world());
            rho1_h(x, y) = rho_h(x, y).template contract<std::array{-1, 1, -2, 2}, std::array{2, 1}, 1, TRACK_INNER>(Id.twist(0));
            rho1_h(x, y) = operator*<TRACK_INNER>(rho1_h(x, y), (1. / rho1_h(x, y).template twist<TRACK_INNER>(0).template trace<TRACK_INNER>()));
            if constexpr(not ENABLE_AD) {
                rho1_h_hermitian_check(x, y) = (rho1_h(x, y) - rho1_h(x, y).adjoint()).norm() / (rho1_h(x, y) + rho1_h(x, y).adjoint()).norm();
            }
            // rho1_h(x, y).print(std::cout, true);
            // std::cout << std::endl;
        }
    }
    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([curr_ = curr, res = *this]() mutable {
            stan::math::nested_rev_autodiff nested;
            curr_.template computeRDM_h<TRACK, false>();
            for(auto i = 0ul; i < curr_.cell().uniqueSize(); ++i) {
                curr_.rho_h[i].adj() = res.rho_h[i].adj();
                curr_.rho1_h[i].adj() = res.rho1_h[i].adj();
            }
            stan::math::grad();
        });
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::computeRDM_v()
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;
    auto curr = *this;

    SPDLOG_INFO("Compute rho_v.");
    rho_v = TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>>(cell_.pattern);
    rho1_v = TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>>(cell_.pattern);
    rho_v_hermitian_check = TMatrix<double>(cell_.pattern);
    rho1_v_hermitian_check = TMatrix<double>(cell_.pattern);

    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(rho_v.isChanged(x, y)) { continue; }
            if constexpr(TRank == 2) {
                auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, TRACK_INNER>(TA1s(x, y - 1));
                auto T4C1T1 = TA4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(
                    C1T1.template twist<TRACK_INNER>(0));
                auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(A->As(x, y));
                auto C2T2 = CA2s(x + 1, y - 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1, TRACK_INNER>(TA2s(x + 1, y));
                auto AdC2T2 = A->Adags(x, y)
                                  .template twist<TRACK_INNER>(3)
                                  .template twist<TRACK_INNER>(4)
                                  .template contract<std::array{-1, -2, -3, 1, -4}, std::array{-5, -6, 1, -7}, 4, TRACK_INNER>(C2T2);
                auto upper_half =
                    T4C1T1A.template contract<std::array{-1, 1, 2, 3, 4, -2, -3}, std::array{1, 3, -4, -5, 2, 4, -6}, 3, TRACK_INNER>(AdC2T2);

                auto C4T3 = CA4s(x - 1, y + 2).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1, TRACK_INNER>(TA3s(x, y + 2));
                auto T4C4T3 = TA4s(x - 1, y + 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(C4T3);
                auto AT4C4T3 = A->As(x, y + 1)
                                   .template twist<TRACK_INNER>(0)
                                   .template contract<std::array{1, -1, -2, 2, -3}, std::array{-4, 1, -5, 2, -6, -7}, 3, TRACK_INNER>(T4C4T3);
                auto T2C3 = TA2s(x + 1, y + 1).template contract<std::array{-1, -2, -3, 1}, std::array{1, -4}, 3, TRACK_INNER>(CA3s(x + 1, y + 2));
                auto T2C3Ad = T2C3.template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, -6, 1, -7}, 3, TRACK_INNER>(A->Adags(x, y + 1));
                auto lower_half =
                    AT4C4T3.template contract<std::array{-1, 1, -2, -3, 2, 3, 4}, std::array{1, -4, 4, 2, -5, -6, 3}, 3, TRACK_INNER>(T2C3Ad);

                rho_v(x, y) =
                    upper_half.template contract<std::array{1, 2, -3, -1, 3, 4}, std::array{2, -4, 1, 4, 3, -2}, 2, TRACK_INNER>(lower_half);
            } else if constexpr(TRank == 1) {
                auto get_fuse = [this](int x, int y, Opts::Leg leg) {
                    return Tensor<Scalar, 2, 1, Symmetry, false>::Identity({{A->ketBasis(x, y, leg), A->braBasis(x, y, leg)}},
                                                                           {{A->ketBasis(x, y, leg).combine(A->braBasis(x, y, leg)).forgetHistory()}},
                                                                           A->As(x, y).world());
                };
                auto get_split = [this](int x, int y, Opts::Leg leg) {
                    return Tensor<Scalar, 1, 2, Symmetry, false>::Identity({{A->ketBasis(x, y, leg).combine(A->braBasis(x, y, leg)).forgetHistory()}},
                                                                           {{A->ketBasis(x, y, leg), A->braBasis(x, y, leg)}},
                                                                           A->As(x, y).world());
                };

                auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3}, 1, TRACK_INNER>(TA1s(x, y - 1));
                auto T4C1T1_ = TA4s(x - 1, y).template contract<std::array{1, -1, -2}, std::array{1, -4, -5}, 3, TRACK_INNER>(
                    C1T1.template twist<TRACK_INNER>(0));
                auto T4C1T1 = T4C1T1_
                                  .template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5}, 3, TRACK_INNER>(
                                      get_fuse(x - 1, y, Opts::Leg::Right).adjoint().eval())
                                  .template contract<std::array{-1, -4, 1, -2, -3}, std::array{1, -5, -6}, 3, TRACK_INNER>(
                                      get_fuse(x, y - 1, Opts::Leg::Bottom).adjoint().eval());
                auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(A->As(x, y));
                auto C2T2_ = CA2s(x + 1, y - 1).template contract<std::array{-1, 1}, std::array{-2, 1, -3}, 1, TRACK_INNER>(TA2s(x + 1, y));
                auto C2T2 = C2T2_.template contract<std::array{-1, 1, -4}, std::array{-2, -3, 1}, 1, TRACK_INNER>(
                    get_split(x + 1, y, Opts::Leg::Left).adjoint().eval());
                auto AdC2T2 = A->Adags(x, y)
                                  .template twist<TRACK_INNER>(3)
                                  .template twist<TRACK_INNER>(4)
                                  .template contract<std::array{-1, -2, -3, 1, -4}, std::array{-5, -6, 1, -7}, 4, TRACK_INNER>(C2T2);
                auto upper_half =
                    T4C1T1A.template contract<std::array{-1, 1, 2, 3, 4, -2, -3}, std::array{1, 3, -4, -5, 2, 4, -6}, 3, TRACK_INNER>(AdC2T2);

                auto C4T3 = CA4s(x - 1, y + 2).template contract<std::array{-1, 1}, std::array{-2, 1, -3}, 1, TRACK_INNER>(TA3s(x, y + 2));
                auto T4C4T3_ = TA4s(x - 1, y + 1).template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2, TRACK_INNER>(C4T3);
                auto T4C4T3 = T4C4T3_
                                  .template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5}, 3, TRACK_INNER>(
                                      get_fuse(x - 1, y + 1, Opts::Leg::Right).adjoint().eval())
                                  .template contract<std::array{-1, 1, -6, -2, -3}, std::array{-4, -5, 1}, 3, TRACK_INNER>(
                                      get_split(x, y + 2, Opts::Leg::Top).adjoint().eval());
                auto AT4C4T3 = A->As(x, y + 1)
                                   .template twist<TRACK_INNER>(0)
                                   .template contract<std::array{1, -1, -2, 2, -3}, std::array{-4, 1, -5, 2, -6, -7}, 3, TRACK_INNER>(T4C4T3);
                auto T2C3_ = TA2s(x + 1, y + 1).template contract<std::array{-1, -2, 1}, std::array{1, -3}, 2, TRACK_INNER>(CA3s(x + 1, y + 2));
                auto T2C3 = T2C3_.template contract<std::array{1, -3, -4}, std::array{-1, -2, 1}, 3, TRACK_INNER>(
                    get_split(x + 1, y + 1, Opts::Leg::Left).adjoint().eval());
                auto T2C3Ad = T2C3.template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, -6, 1, -7}, 3, TRACK_INNER>(A->Adags(x, y + 1));
                auto lower_half =
                    AT4C4T3.template contract<std::array{-1, 1, -2, -3, 2, 3, 4}, std::array{1, -4, 4, 2, -5, -6, 3}, 3, TRACK_INNER>(T2C3Ad);

                rho_v(x, y) =
                    upper_half.template contract<std::array{1, 2, -3, -1, 3, 4}, std::array{2, -4, 1, 4, 3, -2}, 2, TRACK_INNER>(lower_half);
            }
            auto Id2 =
                Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rho_v(x, y).uncoupledCodomain(), rho_v(x, y).uncoupledDomain(), rho_v(x, y).world());
            auto norm = rho_v(x, y)
                            .template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0, TRACK_INNER>(Id2.twist(0).twist(1))
                            .template trace<TRACK_INNER>();
            if(std::real(norm) < 0.) { Log::warning(Log::globalLevel, "RDM_v: Negative norm detected."); }
            rho_v(x, y) = operator*<TRACK_INNER>(rho_v(x, y), (1. / norm));
            if constexpr(not ENABLE_AD) {
                rho_v_hermitian_check(x, y) = (rho_v(x, y) - rho_v(x, y).adjoint()).norm() / (rho_v(x, y) + rho_v(x, y).adjoint()).norm();
            }
            // fmt::print("rho_v at ({},{}):\n", x, y);
            // rho_v(x, y).print(std::cout, true);
            // std::cout << std::endl;
            auto Id = Tensor<Scalar, 1, 1, Symmetry, false>::Identity(
                {{rho_v(x, y).uncoupledCodomain()[1]}}, {{rho_v(x, y).uncoupledDomain()[1]}}, rho_v(x, y).world());
            rho1_v(x, y) = rho_v(x, y).template contract<std::array{-1, 1, -2, 2}, std::array{2, 1}, 1, TRACK_INNER>(Id.twist(0));
            rho1_v(x, y) = operator*<TRACK_INNER>(rho1_v(x, y), (1. / rho1_v(x, y).template twist<TRACK_INNER>(0).template trace<TRACK_INNER>()));
            if constexpr(not ENABLE_AD) {
                rho1_v_hermitian_check(x, y) = (rho1_v(x, y) - rho1_v(x, y).adjoint()).norm() / (rho1_v(x, y) + rho1_v(x, y).adjoint()).norm();
            }
            // rho1_v(x, y).print(std::cout, true);
            // std::cout << std::endl;
        }
    }
    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([curr_ = curr, res = *this]() mutable {
            stan::math::nested_rev_autodiff nested;
            curr_.template computeRDM_v<TRACK, false>();
            for(auto i = 0ul; i < curr_.cell().uniqueSize(); ++i) {
                curr_.rho_v[i].adj() = res.rho_v[i].adj();
                curr_.rho1_v[i].adj() = res.rho1_v[i].adj();
            }
            stan::math::grad();
        });
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::left_move()
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;
    auto curr = *this;

    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 0, 2, Symmetry, TRACK_INNER>> C1_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER>> T4_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER>> C4_new(cell_.pattern);

    CA1s.resetChange();
    CA4s.resetChange();
    TA4s.resetChange();
    SPDLOG_INFO("left move");
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors<TRACK_INNER>(x, y, Opts::DIRECTION::LEFT); // move assignment
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(C1_new.isChanged(x, y - 1) == T4_new.isChanged(x, y) and C1_new.isChanged(x, y - 1) == C4_new.isChanged(x, y + 1));
            if(C1_new.isChanged(x, y - 1)) { continue; }
            std::tie(C1_new(x, y - 1), T4_new(x, y), C4_new(x, y + 1)) = renormalize_left<TRACK_INNER>(x, y, P1, P2);
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(CA1s.isChanged(x, y) == TA4s.isChanged(x, y) and CA1s.isChanged(x, y) == CA4s.isChanged(x, y));
            if(CA1s.isChanged(x, y)) { continue; }
            CA1s(x, y) = std::as_const(C1_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC1new: ([{}],[{}⊗{}])",
                        x,
                        y,
                        CA1s(x, y).coupledDomain().dim(),
                        CA1s(x, y).uncoupledCodomain()[0].dim(),
                        CA1s(x, y).uncoupledCodomain()[1].dim());
            TA4s(x, y) = std::as_const(T4_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT4new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        TA4s(x, y).coupledDomain().dim(),
                        TA4s(x, y).uncoupledCodomain()[0].dim(),
                        TA4s(x, y).uncoupledCodomain()[1].dim(),
                        TA4s(x, y).uncoupledCodomain()[2].dim());

            CA4s(x, y) = std::as_const(C4_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC4new: ([{}],[{}])", x, y, CA4s(x, y).uncoupledDomain()[0].dim(), CA4s(x, y).uncoupledCodomain()[0].dim());
        }
    }
    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([curr_ = curr, res = *this]() mutable {
            stan::math::nested_rev_autodiff nested;
            curr_.template left_move<TRACK, false>();
            for(auto i = 0ul; i < curr_.cell().uniqueSize(); ++i) {
                curr_.CA1s[i].adj() = res.CA1s[i].adj();
                curr_.CA4s[i].adj() = res.CA4s[i].adj();
                curr_.TA4s[i].adj() = res.TA4s[i].adj();
            }
            stan::math::grad();
        });
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::right_move()
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;
    auto curr = *this;

    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER>> C2_new(cell_.pattern);
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER>> T2_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 2, 0, Symmetry, TRACK_INNER>> C3_new(cell_.pattern);

    CA2s.resetChange();
    CA3s.resetChange();
    TA2s.resetChange();
    SPDLOG_INFO("right move");
    for(int x = cell_.Lx; x >= 0; --x) {
        for(int y = 0; y < cell_.Ly; y++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors<TRACK_INNER>(x, y, Opts::DIRECTION::RIGHT); // move assignment
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(C2_new.isChanged(x, y - 1) == T2_new.isChanged(x, y) and C2_new.isChanged(x, y - 1) == C3_new.isChanged(x, y + 1));
            if(C2_new.isChanged(x, y - 1)) { continue; }
            std::tie(C2_new(x, y - 1), T2_new(x, y), C3_new(x, y + 1)) = renormalize_right<TRACK_INNER>(x, y, P1, P2);
        }
        for(int y = 0; y < cell_.Ly; y++) {
            assert(CA2s.isChanged(x, y) == TA2s.isChanged(x, y) and CA2s.isChanged(x, y) == CA3s.isChanged(x, y));
            if(CA2s.isChanged(x, y)) { continue; }
            CA2s(x, y) = std::as_const(C2_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC2new: ([{}],[{}])", x, y, CA2s(x, y).uncoupledDomain()[0].dim(), CA2s(x, y).uncoupledCodomain()[0].dim());
            TA2s(x, y) = std::as_const(T2_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT2new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        TA2s(x, y).uncoupledDomain()[0].dim(),
                        TA2s(x, y).uncoupledDomain()[1].dim(),
                        TA2s(x, y).uncoupledDomain()[2].dim(),
                        TA2s(x, y).coupledCodomain().dim());
            CA3s(x, y) = std::as_const(C3_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC3new: ([{}⊗{}],[{}])",
                        x,
                        y,
                        CA3s(x, y).uncoupledDomain()[0].dim(),
                        CA3s(x, y).uncoupledDomain()[1].dim(),
                        CA3s(x, y).coupledCodomain().dim());
        }
    }
    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([curr_ = curr, res = *this]() mutable {
            stan::math::nested_rev_autodiff nested;
            curr_.template right_move<TRACK, false>();
            for(auto i = 0ul; i < curr_.cell().uniqueSize(); ++i) {
                curr_.CA2s[i].adj() = res.CA2s[i].adj();
                curr_.CA3s[i].adj() = res.CA3s[i].adj();
                curr_.TA2s[i].adj() = res.TA2s[i].adj();
            }
            stan::math::grad();
        });
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::top_move()
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;
    auto curr = *this;

    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 0, 2, Symmetry, TRACK_INNER>> C1_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER>> T1_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER>> C2_new(cell_.pattern);

    CA1s.resetChange();
    CA2s.resetChange();
    TA1s.resetChange();
    SPDLOG_INFO("top move");
    for(int y = 0; y < cell_.Ly; y++) {
        for(int x = 0; x < cell_.Lx; x++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors<TRACK_INNER>(x, y, Opts::DIRECTION::TOP); // move assignment
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(C1_new.isChanged(x - 1, y) == C2_new.isChanged(x + 1, y) and C1_new.isChanged(x - 1, y) == T1_new.isChanged(x, y));
            if(C1_new.isChanged(x - 1, y)) { continue; }
            std::tie(C1_new(x - 1, y), T1_new(x, y), C2_new(x + 1, y)) = renormalize_top<TRACK_INNER>(x, y, P1, P2);
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(CA1s.isChanged(x, y) == CA2s.isChanged(x, y) and CA1s.isChanged(x, y) == TA1s.isChanged(x, y));
            if(CA1s.isChanged(x, y)) { continue; }
            CA1s(x, y) = std::as_const(C1_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC1new: ([{}],[{}⊗{}])",
                        x,
                        y,
                        CA1s(x, y).coupledDomain().dim(),
                        CA1s(x, y).uncoupledCodomain()[0].dim(),
                        CA1s(x, y).uncoupledCodomain()[1].dim());
            TA1s(x, y) = std::as_const(T1_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT1new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        TA1s(x, y).coupledDomain().dim(),
                        TA1s(x, y).uncoupledCodomain()[0].dim(),
                        TA1s(x, y).uncoupledCodomain()[1].dim(),
                        TA1s(x, y).uncoupledCodomain()[2].dim());
            CA2s(x, y) = std::as_const(C2_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC2new: ([{}],[{}])", x, y, CA2s(x, y).uncoupledDomain()[0].dim(), CA2s(x, y).uncoupledCodomain()[0].dim());
        }
    }
    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([curr_ = curr, res = *this]() mutable {
            stan::math::nested_rev_autodiff nested;
            curr_.template top_move<TRACK, false>();
            for(auto i = 0ul; i < curr_.cell().uniqueSize(); ++i) {
                curr_.CA1s[i].adj() = res.CA1s[i].adj();
                curr_.CA2s[i].adj() = res.CA2s[i].adj();
                curr_.TA1s[i].adj() = res.TA1s[i].adj();
            }
            stan::math::grad();
        });
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
void CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::bottom_move()
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;
    auto curr = *this;

    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER>> P1(cell_.pattern);
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER>> P2(cell_.pattern);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER>> C4_new(cell_.pattern);
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER>> T3_new(cell_.pattern);
    TMatrix<Tensor<Scalar, 2, 0, Symmetry, TRACK_INNER>> C3_new(cell_.pattern);

    CA4s.resetChange();
    CA3s.resetChange();
    TA3s.resetChange();
    SPDLOG_INFO("bottom move");
    for(int y = cell_.Ly; y >= 0; --y) {
        for(int x = 0; x < cell_.Lx; x++) {
            assert(P1.isChanged(x, y) == P2.isChanged(x, y));
            if(P1.isChanged(x, y)) { continue; }
            std::tie(P1(x, y), P2(x, y)) = get_projectors<TRACK_INNER>(x, y, Opts::DIRECTION::BOTTOM); // move assignment
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(C4_new.isChanged(x - 1, y) == C3_new.isChanged(x + 1, y) and C4_new.isChanged(x - 1, y) == T3_new.isChanged(x, y));
            if(C4_new.isChanged(x - 1, y)) { continue; }
            std::tie(C4_new(x - 1, y), T3_new(x, y), C3_new(x + 1, y)) = renormalize_bottom<TRACK_INNER>(x, y, P1, P2);
        }
        for(int x = 0; x < cell_.Lx; x++) {
            assert(CA4s.isChanged(x, y) == CA3s.isChanged(x, y) and CA4s.isChanged(x, y) == TA3s.isChanged(x, y));
            if(CA4s.isChanged(x, y)) { continue; }
            CA4s(x, y) = std::as_const(C4_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC4new: ([{}],[{}])", x, y, CA4s(x, y).uncoupledDomain()[0].dim(), CA4s(x, y).uncoupledCodomain()[0].dim());
            TA3s(x, y) = std::as_const(T3_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tT3new: ([{}],[{}⊗{}⊗{}])",
                        x,
                        y,
                        TA3s(x, y).uncoupledDomain()[0].dim(),
                        TA3s(x, y).uncoupledDomain()[1].dim(),
                        TA3s(x, y).uncoupledDomain()[2].dim(),
                        TA3s(x, y).coupledCodomain().dim());
            CA3s(x, y) = std::as_const(C3_new)(x, y);
            SPDLOG_INFO("site: ({},{})\tC3new: ([{}⊗{}],[{}])",
                        x,
                        y,
                        CA3s(x, y).uncoupledDomain()[0].dim(),
                        CA3s(x, y).uncoupledDomain()[1].dim(),
                        CA3s(x, y).coupledCodomain().dim());
        }
    }
    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([curr_ = curr, res = *this]() mutable {
            stan::math::nested_rev_autodiff nested;
            curr_.template bottom_move<TRACK, false>();
            for(auto i = 0ul; i < curr_.cell().uniqueSize(); ++i) {
                curr_.CA3s[i].adj() = res.CA3s[i].adj();
                curr_.CA4s[i].adj() = res.CA4s[i].adj();
                curr_.TA3s[i].adj() = res.TA3s[i].adj();
            }
            stan::math::grad();
        });
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
std::pair<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>, Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>
CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::get_projectors(const int x, const int y, const Opts::DIRECTION dir) XPED_CONST
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;

    Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK> P1;
    Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK> P2;
    Tensor<Scalar, TRank + 1, TRank + 1, Symmetry, TRACK_INNER> Q1, Q2, Q3, Q4;
    switch(dir) {
    case Opts::DIRECTION::LEFT: {
        switch(proj_m) {
        case Opts::PROJECTION::CORNER: {
            Q1 = contractCorner<TRACK_INNER>(x, y, Opts::CORNER::UPPER_LEFT);
            if constexpr(TRank == 2) {
                Q4 = contractCorner<TRACK_INNER>(x, y + 1, Opts::CORNER::LOWER_LEFT)
                         .template twist<TRACK_INNER>(3)
                         .template twist<TRACK_INNER>(4)
                         .template twist<TRACK_INNER>(5);
            } else if constexpr(TRank == 1) {
                Q4 = contractCorner<TRACK_INNER>(x, y + 1, Opts::CORNER::LOWER_LEFT).template twist<TRACK_INNER>(2).template twist<TRACK_INNER>(3);
            }
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
            std::tie(P1, P2) = decompose(Q4, Q1, chi_);
            break;
        }
        case Opts::PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case Opts::PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    case Opts::DIRECTION::RIGHT: {
        switch(proj_m) {
        case Opts::PROJECTION::CORNER: {
            Q2 = contractCorner<TRACK_INNER>(x, y, Opts::CORNER::UPPER_RIGHT);
            Q3 = contractCorner<TRACK_INNER>(x, y + 1, Opts::CORNER::LOWER_RIGHT);
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
            std::tie(P1, P2) = decompose(Q2, Q3, chi_);
            break;
        }
        case Opts::PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case Opts::PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    case Opts::DIRECTION::TOP: {
        switch(proj_m) {
        case Opts::PROJECTION::CORNER: {
            Q1 = contractCorner<TRACK_INNER>(x, y, Opts::CORNER::UPPER_LEFT);
            Q2 = contractCorner<TRACK_INNER>(x + 1, y, Opts::CORNER::UPPER_RIGHT);
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
            std::tie(P1, P2) = decompose(Q1, Q2, chi_);
            break;
        }
        case Opts::PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case Opts::PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    case Opts::DIRECTION::BOTTOM: {
        switch(proj_m) {
        case Opts::PROJECTION::CORNER: {
            if constexpr(TRank == 2) {
                Q4 = contractCorner<TRACK_INNER>(x, y, Opts::CORNER::LOWER_LEFT)
                         .template twist<TRACK_INNER>(0)
                         .template twist<TRACK_INNER>(1)
                         .template twist<TRACK_INNER>(2);
            } else if constexpr(TRank == 1) {
                Q4 = contractCorner<TRACK_INNER>(x, y, Opts::CORNER::LOWER_LEFT).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1);
            }
            Q3 = contractCorner<TRACK_INNER>(x + 1, y, Opts::CORNER::LOWER_RIGHT);
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
            std::tie(P1, P2) = decompose(Q3, Q4, chi_);
            break;
        }
        case Opts::PROJECTION::HALF: {
            assert(false and "Not implemented.");
            break;
        }
        case Opts::PROJECTION::FULL: {
            assert(false and "Not implemented.");
            break;
        }
        }
        break;
    }
    }
    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([P1_ = P1, P2_ = P2, curr = *this, x, y, dir]() mutable {
            stan::math::nested_rev_autodiff nested;
            auto [P1__, P2__] = curr.template get_projectors<TRACK, false>(x, y, dir);
            P1__.adj() = P1_.adj();
            P2__.adj() = P2_.adj();
            stan::math::grad();
        });
    }
    return std::make_pair(P1, P2);
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
std::tuple<Tensor<Scalar, 0, 2, Symmetry, TRACK>, Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>, Tensor<Scalar, 1, 1, Symmetry, TRACK>>
CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::renormalize_left(const int x,
                                                                                const int y,
                                                                                XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
                                                                                XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
                                                                                bool NORMALIZE) XPED_CONST
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;

    Tensor<Scalar, 0, 2, Symmetry, TRACK> C1_new;
    Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK> T4_new;
    Tensor<Scalar, 1, 1, Symmetry, TRACK> C4_new;

    Tensor<Scalar, 0, 2, Symmetry, TRACK_INNER> C1_new_tmp;
    Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER> T4_new_tmp;
    Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER> C4_new_tmp;

    if constexpr(TRank == 2) {
        auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -4, -2, -3}, 3, TRACK_INNER>(TA1s(x, y - 1));
        C1_new_tmp = P1(x, y - 1)
                         .template twist<TRACK_INNER>(1)
                         .template twist<TRACK_INNER>(2)
                         .template twist<TRACK_INNER>(3)
                         .template contract<std::array{-1, 1, 2, 3}, std::array{1, 2, 3, -2}, 0, TRACK_INNER>(C1T1);
    } else if constexpr(TRank == 1) {
        auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -3, -2}, 2, TRACK_INNER>(TA1s(x, y - 1));
        C1_new_tmp = P1(x, y - 1)
                         .template twist<TRACK_INNER>(1)
                         .template twist<TRACK_INNER>(2)
                         .template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 0, TRACK_INNER>(C1T1);
    }
    C1_new = NORMALIZE ? C1_new_tmp * (1. / C1_new_tmp.maxNorm()) : C1_new_tmp;

    if constexpr(TRank == 2) {
        auto C4T3 = CA4s(x - 1, y + 1).template contract<std::array{-2, 1}, std::array{-3, -4, 1, -1}, 1, TRACK_INNER>(TA3s(x, y + 1));
        C4_new_tmp = C4T3.template contract<std::array{-2, 1, 2, 3}, std::array{1, 2, 3, -1}, 1, TRACK_INNER>(
            P2(x, y).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1).template twist<TRACK_INNER>(2));
    } else if constexpr(TRank == 1) {
        auto C4T3 = CA4s(x - 1, y + 1).template contract<std::array{-2, 1}, std::array{-3, 1, -1}, 1, TRACK_INNER>(TA3s(x, y + 1));
        C4_new_tmp = C4T3.template contract<std::array{-2, 1, 2}, std::array{1, 2, -1}, 1, TRACK_INNER>(
            P2(x, y).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1));
    }
    C4_new = NORMALIZE ? C4_new_tmp * (1. / C4_new_tmp.maxNorm()) : C4_new_tmp;

    if constexpr(TRank == 2) {
        auto P1T4 = P1(x, y)
                        .template twist<TRACK_INNER>(1)
                        .template twist<TRACK_INNER>(2)
                        .template twist<TRACK_INNER>(3)
                        .template contract<std::array{-1, 1, -2, -3}, std::array{-4, 1, -5, -6}, 3, TRACK_INNER>(TA4s(x - 1, y));
        auto P1T4A = P1T4.template contract<std::array{-1, 2, -2, -3, 1, -4}, std::array{1, -5, -6, 2, -7}, 4, TRACK_INNER>(A->As(x, y));
        auto P1T4AAdag = P1T4A.template contract<std::array{-1, 2, -4, 1, -5, -2, 3}, std::array{1, -6, 3, -3, 2}, 3, TRACK_INNER>(
            A->Adags(x, y).template twist<TRACK_INNER>(3).template twist<TRACK_INNER>(4));
        T4_new_tmp = P1T4AAdag.template contract<std::array{-2, -3, -4, 1, 2, 3}, std::array{1, 2, 3, -1}, 1, TRACK_INNER>(
            P2(x, y - 1).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1).template twist<TRACK_INNER>(2));
    } else if constexpr(TRank == 1) {
        auto P1T4 = P1(x, y)
                        .template twist<TRACK_INNER>(1)
                        .template twist<TRACK_INNER>(2)
                        .template contract<std::array{-1, 1, -2}, std::array{-3, 1, -4}, 2, TRACK_INNER>(TA4s(x - 1, y));
        auto P1T4M = P1T4.template contract<std::array{-1, 2, -2, 1}, std::array{1, -3, -4, 2}, 2, TRACK_INNER>(Ms(x, y));
        T4_new_tmp = P1T4M.template contract<std::array{-2, 1, 2, -3}, std::array{1, 2, -1}, 1, TRACK_INNER>(
            P2(x, y - 1).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1));
    }
    T4_new = NORMALIZE ? T4_new_tmp * (1. / T4_new_tmp.maxNorm()) : T4_new_tmp;

    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([C1_new, C4_new, T4_new, P1_ = P1, P2_ = P2, curr = *this, x, y, NORMALIZE]() mutable {
            stan::math::nested_rev_autodiff nested;
            auto [C1_new_, T4_new_, C4_new_] = curr.template renormalize_left<TRACK, false>(x, y, P1_, P2_, NORMALIZE);
            C1_new_.adj() = C1_new.adj();
            C4_new_.adj() = C4_new.adj();
            T4_new_.adj() = T4_new.adj();
            stan::math::grad();
        });
    }

    return std::make_tuple(C1_new, T4_new, C4_new);
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
std::tuple<Tensor<Scalar, 1, 1, Symmetry, TRACK>, Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>, Tensor<Scalar, 2, 0, Symmetry, TRACK>>
CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::renormalize_right(
    const int x,
    const int y,
    XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
    XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
    bool NORMALIZE) XPED_CONST
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;

    Tensor<Scalar, 1, 1, Symmetry, TRACK> C2_new;
    Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK> T2_new;
    Tensor<Scalar, 2, 0, Symmetry, TRACK> C3_new;

    Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER> C2_new_tmp;
    Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER> T2_new_tmp;
    Tensor<Scalar, 2, 0, Symmetry, TRACK_INNER> C3_new_tmp;

    if constexpr(TRank == 2) {
        auto T1C2 = TA1s(x, y - 1).template contract<std::array{-1, 1, -3, -4}, std::array{1, -2}, 1, TRACK_INNER>(CA2s(x + 1, y - 1));
        C2_new_tmp = T1C2.template contract<std::array{-1, 1, 2, 3}, std::array{1, 2, 3, -2}, 1, TRACK_INNER>(P2(x, y - 1));
    } else if constexpr(TRank == 1) {
        auto T1C2 = TA1s(x, y - 1).template contract<std::array{-1, 1, -3}, std::array{1, -2}, 1, TRACK_INNER>(CA2s(x + 1, y - 1));
        C2_new_tmp = T1C2.template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1, TRACK_INNER>(P2(x, y - 1));
    }
    C2_new = NORMALIZE ? C2_new_tmp * (1. / C2_new_tmp.maxNorm()) : C2_new_tmp;

    if constexpr(TRank == 2) {
        auto C3T3 = CA3s(x + 1, y + 1)
                        .template twist<TRACK_INNER>(1)
                        .template contract<std::array{-1, 1}, std::array{-2, -3, -4, 1}, 3, TRACK_INNER>(TA3s(x, y + 1));
        C3_new_tmp = P1(x, y).template contract<std::array{-1, 1, 2, 3}, std::array{1, 2, 3, -2}, 2, TRACK_INNER>(C3T3);
    } else if constexpr(TRank == 1) {
        auto C3T3 = CA3s(x + 1, y + 1)
                        .template twist<TRACK_INNER>(1)
                        .template contract<std::array{-1, 1}, std::array{-2, -3, 1}, 2, TRACK_INNER>(TA3s(x, y + 1));
        C3_new_tmp = P1(x, y).template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 2, TRACK_INNER>(C3T3);
    }
    C3_new = NORMALIZE ? C3_new_tmp * (1. / C3_new_tmp.maxNorm()) : C3_new_tmp;

    if constexpr(TRank == 2) {
        auto P1T2 = P1(x, y - 1).template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, 1, -6}, 3, TRACK_INNER>(TA2s(x + 1, y));
        auto P1T2A = P1T2.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, 1, 2, -6, -7}, 4, TRACK_INNER>(
            A->As(x, y).template twist<TRACK_INNER>(2));
        auto P1T2AAdag = P1T2A.template contract<std::array{-1, 1, 2, -4, -2, -5, 3}, std::array{-3, 1, 3, 2, -6}, 3, TRACK_INNER>(
            A->Adags(x, y).template twist<TRACK_INNER>(4));
        T2_new_tmp = P1T2AAdag.template contract<std::array{-3, -1, -2, 1, 2, 3}, std::array{1, 2, 3, -4}, 3, TRACK_INNER>(P2(x, y));
    } else if constexpr(TRank == 1) {
        auto P1T2 = P1(x, y - 1).template contract<std::array{-1, 1, -2}, std::array{-4, 1, -5}, 2, TRACK_INNER>(TA2s(x + 1, y));
        auto P1T2M =
            P1T2.template contract<std::array{-1, 1, 2, -2}, std::array{-3, 1, 2, -4}, 2, TRACK_INNER>(Ms(x, y)); //.template twist<TRACK_INNER>(2));
        T2_new_tmp = P1T2M.template contract<std::array{-2, 1, -1, 2}, std::array{1, 2, -3}, 2, TRACK_INNER>(P2(x, y));
    }
    T2_new = NORMALIZE ? T2_new_tmp * (1. / T2_new_tmp.maxNorm()) : T2_new_tmp;

    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([C2_new, C3_new, T2_new, P1_ = P1, P2_ = P2, curr = *this, x, y, NORMALIZE]() mutable {
            stan::math::nested_rev_autodiff nested;
            auto [C2_new_, T2_new_, C3_new_] = curr.template renormalize_right<TRACK, false>(x, y, P1_, P2_, NORMALIZE);
            C2_new_.adj() = C2_new.adj();
            C3_new_.adj() = C3_new.adj();
            T2_new_.adj() = T2_new.adj();
            stan::math::grad();
        });
    }

    return std::make_tuple(C2_new, T2_new, C3_new);
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
std::tuple<Tensor<Scalar, 0, 2, Symmetry, TRACK>, Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>, Tensor<Scalar, 1, 1, Symmetry, TRACK>>
CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::renormalize_top(const int x,
                                                                               const int y,
                                                                               XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
                                                                               XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
                                                                               bool NORMALIZE) XPED_CONST
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;

    Tensor<Scalar, 0, 2, Symmetry, TRACK> C1_new;
    Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK> T1_new;
    Tensor<Scalar, 1, 1, Symmetry, TRACK> C2_new;

    Tensor<Scalar, 0, 2, Symmetry, TRACK_INNER> C1_new_tmp;
    Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK_INNER> T1_new_tmp;
    Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER> C2_new_tmp;

    if constexpr(TRank == 2) {
        auto T4C1 = TA4s(x - 1, y).template contract<std::array{1, -1, -3, -4}, std::array{1, -2}, 1, TRACK_INNER>(
            CA1s(x - 1, y - 1).template twist<TRACK_INNER>(0));
        C1_new_tmp = T4C1.template contract<std::array{-1, 1, 2, 3}, std::array{1, 2, 3, -2}, 0, TRACK_INNER>(P2(x - 1, y));
    } else if constexpr(TRank == 1) {
        auto T4C1 = TA4s(x - 1, y).template contract<std::array{1, -1, -3}, std::array{1, -2}, 1, TRACK_INNER>(
            CA1s(x - 1, y - 1).template twist<TRACK_INNER>(0));
        C1_new_tmp = T4C1.template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 0, TRACK_INNER>(P2(x - 1, y));
    }
    C1_new = NORMALIZE ? C1_new_tmp * (1. / C1_new_tmp.maxNorm()) : C1_new_tmp;

    if constexpr(TRank == 2) {
        auto C2T2 = CA2s(x + 1, y - 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 3, TRACK_INNER>(TA2s(x + 1, y));
        C2_new_tmp = P1(x, y).template contract<std::array{-1, 1, 2, 3}, std::array{1, 2, 3, -2}, 1, TRACK_INNER>(C2T2);
    } else if constexpr(TRank == 1) {
        auto C2T2 = CA2s(x + 1, y - 1).template contract<std::array{-1, 1}, std::array{-2, 1, -3}, 2, TRACK_INNER>(TA2s(x + 1, y));
        C2_new_tmp = P1(x, y).template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 1, TRACK_INNER>(C2T2);
    }
    C2_new = NORMALIZE ? C2_new_tmp * (1. / C2_new_tmp.maxNorm()) : C2_new_tmp;

    if constexpr(TRank == 2) {
        auto P1T1 = P1(x - 1, y).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(TA1s(x, y - 1));
        auto P1T1A = P1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(A->As(x, y));
        auto P1T1AAdag = P1T1A.template contract<std::array{-1, 1, -4, 2, -5, -2, 3}, std::array{1, 2, 3, -6, -3}, 3, TRACK_INNER>(
            A->Adags(x, y).template twist<TRACK_INNER>(3).template twist<TRACK_INNER>(4));
        T1_new_tmp = P1T1AAdag.template contract<std::array{-1, -3, -4, 1, 2, 3}, std::array{1, 2, 3, -2}, 1, TRACK_INNER>(P2(x, y));
    } else if constexpr(TRank == 1) {
        auto P1T1 = P1(x - 1, y).template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2, TRACK_INNER>(TA1s(x, y - 1));
        auto P1T1M = P1T1.template contract<std::array{-1, 1, -2, 2}, std::array{1, 2, -3, -4}, 2, TRACK_INNER>(Ms(x, y));
        T1_new_tmp = P1T1M.template contract<std::array{-1, 1, 2, -3}, std::array{1, 2, -2}, 1, TRACK_INNER>(P2(x, y));
    }
    T1_new = NORMALIZE ? T1_new_tmp * (1. / T1_new_tmp.maxNorm()) : T1_new_tmp;

    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([C1_new, C2_new, T1_new, P1_ = P1, P2_ = P2, curr = *this, x, y, NORMALIZE]() mutable {
            stan::math::nested_rev_autodiff nested;
            auto [C1_new_, T1_new_, C2_new_] = curr.template renormalize_top<TRACK, false>(x, y, P1_, P2_, NORMALIZE);
            C1_new_.adj() = C1_new.adj();
            C2_new_.adj() = C2_new.adj();
            T1_new_.adj() = T1_new.adj();
            stan::math::grad();
        });
    }

    return std::make_tuple(C1_new, T1_new, C2_new);
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
std::tuple<Tensor<Scalar, 1, 1, Symmetry, TRACK>, Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>, Tensor<Scalar, 2, 0, Symmetry, TRACK>>
CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::renormalize_bottom(
    const int x,
    const int y,
    XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
    XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
    bool NORMALIZE) XPED_CONST
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;

    Tensor<Scalar, 1, 1, Symmetry, TRACK> C4_new;
    Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK> T3_new;
    Tensor<Scalar, 2, 0, Symmetry, TRACK> C3_new;

    Tensor<Scalar, 1, 1, Symmetry, TRACK_INNER> C4_new_tmp;
    Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK_INNER> T3_new_tmp;
    Tensor<Scalar, 2, 0, Symmetry, TRACK_INNER> C3_new_tmp;

    if constexpr(TRank == 2) {
        auto T4C4 = TA4s(x - 1, y).template contract<std::array{-4, 1, -2, -3}, std::array{1, -1}, 3, TRACK_INNER>(CA4s(x - 1, y + 1));
        C4_new_tmp = P1(x - 1, y)
                         .template twist<TRACK_INNER>(1)
                         .template twist<TRACK_INNER>(2)
                         .template twist<TRACK_INNER>(3)
                         .template contract<std::array{-2, 1, 2, 3}, std::array{1, 2, 3, -1}, 1, TRACK_INNER>(T4C4);
    } else if constexpr(TRank == 1) {
        auto T4C4 = TA4s(x - 1, y).template contract<std::array{-3, 1, -2}, std::array{1, -1}, 2, TRACK_INNER>(CA4s(x - 1, y + 1));
        C4_new_tmp = P1(x - 1, y)
                         .template twist<TRACK_INNER>(1)
                         .template twist<TRACK_INNER>(2)
                         .template contract<std::array{-2, 1, 2}, std::array{1, 2, -1}, 1, TRACK_INNER>(T4C4);
    }
    C4_new = NORMALIZE ? C4_new_tmp * (1. / C4_new_tmp.maxNorm()) : C4_new_tmp;

    if constexpr(TRank == 2) {
        auto T2C3 = TA2s(x + 1, y).template contract<std::array{-3, -4, -1, 1}, std::array{1, -2}, 1, TRACK_INNER>(CA3s(x + 1, y + 1));
        C3_new_tmp = T2C3.template contract<std::array{-1, 1, 2, 3}, std::array{1, 2, 3, -2}, 2, TRACK_INNER>(
            P2(x, y).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1).template twist<TRACK_INNER>(2));
    } else if constexpr(TRank == 1) {
        auto T2C3 = TA2s(x + 1, y).template contract<std::array{-3, -1, 1}, std::array{1, -2}, 1, TRACK_INNER>(CA3s(x + 1, y + 1));
        C3_new_tmp = T2C3.template contract<std::array{-1, 1, 2}, std::array{1, 2, -2}, 2, TRACK_INNER>(
            P2(x, y).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1));
    }
    C3_new = NORMALIZE ? C3_new_tmp * (1. / C3_new_tmp.maxNorm()) : C3_new_tmp;

    if constexpr(TRank == 2) {
        auto P1T3 = P1(x, y)
                        .template twist<TRACK_INNER>(1)
                        .template twist<TRACK_INNER>(2)
                        .template twist<TRACK_INNER>(3)
                        .template contract<std::array{-1, 1, -2, -3}, std::array{-4, -5, -6, 1}, 3, TRACK_INNER>(
                            TA3s(x, y + 1).template twist<TRACK_INNER>(0));
        auto P1T3A = P1T3.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, -6, 1, 2, -7}, 4, TRACK_INNER>(A->As(x, y));
        auto P1T3AAdag = P1T3A.template contract<std::array{-1, 1, 2, -4, -5, -2, 3}, std::array{-6, -3, 3, 1, 2}, 3, TRACK_INNER>(
            A->Adags(x, y).template twist<TRACK_INNER>(3));
        T3_new_tmp = P1T3AAdag.template contract<std::array{-4, -1, -2, 1, 2, 3}, std::array{1, 2, 3, -3}, 3, TRACK_INNER>(
            P2(x - 1, y).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1).template twist<TRACK_INNER>(2));
    } else if constexpr(TRank == 1) {
        auto P1T3 =
            P1(x, y)
                .template twist<TRACK_INNER>(1)
                .template twist<TRACK_INNER>(2)
                .template contract<std::array{-1, 1, -2}, std::array{-3, -4, 1}, 2, TRACK_INNER>(TA3s(x, y + 1).template twist<TRACK_INNER>(0));
        auto P1T3M = P1T3.template contract<std::array{-1, 1, 2, -2}, std::array{-3, -4, 1, 2}, 2, TRACK_INNER>(Ms(x, y));
        T3_new_tmp = P1T3M.template contract<std::array{-3, 1, 2, -1}, std::array{1, 2, -2}, 2, TRACK_INNER>(
            P2(x - 1, y).template twist<TRACK_INNER>(0).template twist<TRACK_INNER>(1));
    }
    T3_new = NORMALIZE ? T3_new_tmp * (1. / T3_new_tmp.maxNorm()) : T3_new_tmp;

    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([C3_new, C4_new, T3_new, P1_ = P1, P2_ = P2, curr = *this, x, y, NORMALIZE]() mutable {
            stan::math::nested_rev_autodiff nested;
            auto [C4_new_, T3_new_, C3_new_] = curr.template renormalize_bottom<TRACK, false>(x, y, P1_, P2_, NORMALIZE);
            C3_new_.adj() = C3_new.adj();
            C4_new_.adj() = C4_new.adj();
            T3_new_.adj() = T3_new.adj();
            stan::math::grad();
        });
    }

    return std::make_tuple(C4_new, T3_new, C3_new);
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD, Opts::CTMCheckpoint CPOpts>
template <bool TRACK, bool CP>
Tensor<Scalar, TRank + 1, TRank + 1, Symmetry, TRACK>
CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>::contractCorner(const int x, const int y, const Opts::CORNER corner) XPED_CONST
{
    constexpr bool TRACK_INNER = TRACK ? not CP : false;

    Tensor<Scalar, TRank + 1, TRank + 1, Symmetry, TRACK> Q;
    switch(corner) {
    case Opts::CORNER::UPPER_LEFT: {
        // ooooo -->--
        // o Q o
        // ooooo ==>==
        // |  ||
        // ^  ^^
        // |  ||
        if constexpr(TRank == 2) {
            auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, TRACK_INNER>(TA1s(x, y - 1));
            auto T4C1T1 = TA4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(
                C1T1.template twist<TRACK_INNER>(0));
            auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(A->As(x, y));
            Q = T4C1T1A.template contract<std::array{-1, 1, -4, 2, -5, -2, 3}, std::array{1, 2, 3, -6, -3}, 3, TRACK_INNER>(
                A->Adags(x, y).template twist<TRACK_INNER>(3).template twist<TRACK_INNER>(4));
        } else if constexpr(TRank == 1) {
            auto C1T1 = CA1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3}, 1, TRACK_INNER>(TA1s(x, y - 1));
            auto T4C1T1 =
                TA4s(x - 1, y).template contract<std::array{1, -1, -2}, std::array{1, -4, -5}, 2, TRACK_INNER>(C1T1.template twist<TRACK_INNER>(0));
            Q = T4C1T1.template contract<std::array{-1, 1, -3, 2}, std::array{1, 2, -4, -2}, 2>(Ms(x, y));
        }
        break;
    }
    case Opts::CORNER::LOWER_LEFT: {
        // |  ||
        // ^  ^^
        // |  ||
        // ooooo --<--
        // o Q o
        // ooooo ==<==
        if constexpr(TRank == 2) {
            auto C4T3 = CA4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1, TRACK_INNER>(TA3s(x, y + 1));
            auto T4C4T3 = TA4s(x - 1, y).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(C4T3);
            auto T4C4T3A = T4C4T3.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{1, -5, -6, 2, -7}, 4, TRACK_INNER>(
                A->As(x, y).template twist<TRACK_INNER>(3));
            Q = T4C4T3A.template contract<std::array{-4, 1, 2, -1, -5, -2, 3}, std::array{1, -6, 3, -3, 2}, 3, TRACK_INNER>(
                A->Adags(x, y).template twist<TRACK_INNER>(3));
        } else if constexpr(TRank == 1) {
            auto C4T3 = CA4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, 1, -3}, 1, TRACK_INNER>(TA3s(x, y + 1));
            auto T4C4T3 = TA4s(x - 1, y).template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2, TRACK_INNER>(C4T3);
            Q = T4C4T3.template contract<std::array{-3, 1, 2, -1}, std::array{1, -4, -2, 2}, 2>(Ms(x, y));
        }
        break;
    }
    case Opts::CORNER::UPPER_RIGHT: {
        // -->--ooooo
        //      o Q o
        // ==>==ooooo
        //      |  ||
        //      v  vv
        //      |  ||
        if constexpr(TRank == 2) {
            auto T1C2 = TA1s(x, y - 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3, TRACK_INNER>(CA2s(x + 1, y - 1));
            auto T1C2T2 = T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, 1, -6}, 3, TRACK_INNER>(TA2s(x + 1, y));
            auto T1C2T2A = T1C2T2.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, 1, 2, -6, -7}, 4, TRACK_INNER>(
                A->As(x, y).template twist<TRACK_INNER>(2));
            Q = T1C2T2A.template contract<std::array{-1, 1, 2, -4, -2, -5, 3}, std::array{-3, 1, 3, 2, -6}, 3, TRACK_INNER>(
                A->Adags(x, y).template twist<TRACK_INNER>(4));
        } else if constexpr(TRank == 1) {
            auto T1C2 = TA1s(x, y - 1).template contract<std::array{-1, 1, -2}, std::array{1, -3}, 2, TRACK_INNER>(CA2s(x + 1, y - 1));
            auto T1C2T2 = T1C2.template contract<std::array{-1, -2, 1}, std::array{-3, 1, -4}, 2, TRACK_INNER>(TA2s(x + 1, y));
            Q = T1C2T2.template contract<std::array{-1, 1, 2, -3}, std::array{-2, 1, 2, -4}, 2>(Ms(x, y));
        }
        break;
    }
    case Opts::CORNER::LOWER_RIGHT: {
        //      |  ||
        //      v  vv
        //      |  ||
        // --<--ooooo
        //      o Q o
        // ==<==ooooo
        if constexpr(TRank == 2) {
            auto C3T3 = CA3s(x + 1, y + 1)
                            .template twist<TRACK_INNER>(1)
                            .template contract<std::array{-1, 1}, std::array{-2, -3, -4, 1}, 1, TRACK_INNER>(TA3s(x, y + 1));
            auto T2C3T3 = TA2s(x + 1, y).template contract<std::array{-1, -2, -3, 1}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(C3T3);
            auto T2C3T3A = T2C3T3.template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, -6, 1, 2, -7}, 4, TRACK_INNER>(
                A->As(x, y).template twist<TRACK_INNER>(2).template twist<TRACK_INNER>(3));
            Q = T2C3T3A.template contract<std::array{1, -1, 2, -4, -5, -2, 3}, std::array{-6, -3, 3, 1, 2}, 3, TRACK_INNER>(A->Adags(x, y));
        } else if constexpr(TRank == 1) {
            auto C3T3 = CA3s(x + 1, y + 1)
                            .template twist<TRACK_INNER>(1)
                            .template contract<std::array{-1, 1}, std::array{-2, -3, 1}, 1, TRACK_INNER>(TA3s(x, y + 1));
            auto T2C3T3 = TA2s(x + 1, y).template contract<std::array{-1, -2, 1}, std::array{1, -3, -4}, 2, TRACK_INNER>(C3T3);
            Q = T2C3T3.template contract<std::array{1, -1, 2, -3}, std::array{-4, -2, 1, 2}, 2>(Ms(x, y));
        }
        break;
    }
    }

    if constexpr(TRACK and CP) {
        Xped::reverse_pass_callback_alloc([Q, curr = *this, x, y, corner]() mutable {
            stan::math::nested_rev_autodiff nested;
            auto Q_ = curr.template contractCorner<TRACK, false>(x, y, corner);
            Q_.adj() = Q.adj();
            stan::math::grad();
        });
    }
    return Q;
}

} // namespace Xped
