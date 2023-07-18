#ifndef XPED_EXACT_CONTRACTIONS_HPP_
#define XPED_EXACT_CONTRACTIONS_HPP_

#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/PEPS/iPEPS.hpp"
#include "Xped/Physics/SpinBase.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
std::conditional_t<ENABLE_AD, stan::math::var, Scalar> fourByfour(iPEPS<Scalar, Symmetry, false, ENABLE_AD>& Psi,
                                                                  TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    // SpinBase<Symmetry> B(1);
    // OneSiteObservable<Scalar, Symmetry> obs(Psi.cell().pattern, "Sz");
    // obs.data[0] = B.Sz().data.template trim<2>();

    auto AA = Psi.As[0].template contract<std::array{-1, -2, 1, -3, -7}, std::array{1, -4, -5, -6, -8}, 6>(Psi.As[0]);

    auto AAAA = AA.template contract<std::array{-1, -2, 1, -3, -4, 2, -9, -10}, std::array{-5, 1, -6, 2, -7, -8, -11, -12}, 8>(AA);
    auto AAAAAAAA = AAAA.template contract<std::array{-1, 1, 2, -2, -3, 3, -4, 4, -9, -10, -11, -12},
                                           std::array{-5, 3, 4, -6, -7, 1, -8, 2, -13, -14, -15, -16},
                                           8>(AAAA);
    auto psi = AAAAAAAA.template contract<std::array{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8},
                                          std::array{2, 1, 4, 3, 6, 5, 8, 7, -9, -10, -11, -12, -13, -14, -15, -16},
                                          0>(AAAAAAAA);

    auto rho = psi.template contract<std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, -2},
                                     std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -3, -4},
                                     2>(psi.adjoint().eval());
    auto Id2 = Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rho.uncoupledCodomain(), rho.uncoupledDomain(), rho.world());
    auto norm = rho.template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(Id2.twist(0).twist(1)).trace();
    if constexpr(not ENABLE_AD) {
        fmt::print("Norm: {}\n", norm);
    } else {
        fmt::print("Norm: {}\n", norm.val());
    }
    if(std::real(norm) < 0.) { Log::warning(Log::globalLevel, "rho: Negative norm detected."); }
    rho = (rho * (1. / norm)).eval();
    if constexpr(not ENABLE_AD) {
        double herm = (rho - rho.adjoint()).norm() / (rho + rho.adjoint()).norm();
        fmt::print("Hermiticity check: {}\n", herm);
    } else {
        double herm = (rho.val() - rho.val().adjoint()).norm() / (rho.val() + rho.val().adjoint()).norm();
        fmt::print("Hermiticity check: {}\n", herm);
    }
    auto Id = Tensor<Scalar, 1, 1, Symmetry, false>::Identity({{rho.uncoupledCodomain()[1]}}, {{rho.uncoupledDomain()[1]}}, rho.world());
    // auto rho1 = rho.template contract<std::array{-1, 1, -2, 2}, std::array{2, 1}, 1>(Id.twist(0));
    // rho1 = operator*<false>(rho1, (1. / rho1.twist(0).trace()));
    // auto Sz = rho1_h.twist(0).template contract<std::array{1, 2}, std::array{2, 1}, 0>(obs.data).trace();
    auto res = rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(0, 0)).trace();
    return res;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
std::conditional_t<ENABLE_AD, stan::math::var, Scalar> fourByfour(iPEPS<Scalar, Symmetry, true, ENABLE_AD>& Psi,
                                                                  TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    // SpinBase<Symmetry> B(1);
    // OneSiteObservable<Scalar, Symmetry> obs(Psi.cell().pattern, "Sz");
    // obs.data[0] = B.Sz().data.template trim<2>();

    auto AB = Psi.As[0].template contract<std::array{-1, -2, 1, -3, -7}, std::array{1, -4, -5, -6, -8}, 6>(Psi.Bs[0]);
    auto BA = Psi.Bs[0].template contract<std::array{-1, -2, 1, -3, -7}, std::array{1, -4, -5, -6, -8}, 6>(Psi.As[0]);
    Log::debug("Contracted AB and BA");
    auto ABBA = AB.template contract<std::array{-1, -2, 1, -3, -4, 2, -9, -10}, std::array{-5, 1, -6, 2, -7, -8, -11, -12}, 8>(BA);
    Log::debug("Contracted ABBA");
    auto ABBAABBA = ABBA.template contract<std::array{-1, 1, 2, -2, -3, 3, -4, 4, -9, -10, -11, -12},
                                           std::array{-5, 3, 4, -6, -7, 1, -8, 2, -13, -14, -15, -16},
                                           8>(ABBA);
    Log::debug("Contracted ABBAABBA");
    auto psi = ABBAABBA.template contract<std::array{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8},
                                          std::array{2, 1, 4, 3, 6, 5, 8, 7, -9, -10, -11, -12, -13, -14, -15, -16},
                                          0>(ABBAABBA);
    Log::debug("Contracted psi");
    auto resh0 = calcE_h<0>(psi, op);
    auto resh2 = calcE_h<2>(psi, op);
    // auto resh4 = calcE_h<4>(psi, op);
    // auto resh6 = calcE_h<6>(psi, op);
    // auto resh8 = calcE_h<8>(psi, op);
    // auto resh10 = calcE_h<10>(psi, op);
    // auto resh12 = calcE_h<12>(psi, op);
    // auto resh14 = calcE_h<14>(psi, op);
    // auto resh1 = calcE_h2<1>(psi, op);
    // auto resh3 = calcE_h2<3>(psi, op);
    // auto resh5 = calcE_h2<5>(psi, op);
    // auto resh7 = calcE_h2<7>(psi, op);
    // auto resh9 = calcE_hp<9>(psi, op);
    // auto resh11 = calcE_hp<11>(psi, op);
    // auto resh13 = calcE_hp<13>(psi, op);
    // auto resh15 = calcE_hp<15>(psi, op);

    auto resv0 = calcE_v<0>(psi, op);
    auto resv1 = calcE_v<1>(psi, op);
    // auto resv2 = calcE_v<2>(psi, op);
    // auto resv3 = calcE_v<3>(psi, op);
    // auto resv4 = calcE_v<4>(psi, op);
    // auto resv5 = calcE_v<5>(psi, op);
    // auto resv6 = calcE_vp<6>(psi, op);
    // auto resv7 = calcE_vp<7>(psi, op);
    // auto resv8 = calcE_v<8>(psi, op);
    // auto resv9 = calcE_v<9>(psi, op);
    // auto resv10 = calcE_v<10>(psi, op);
    // auto resv11 = calcE_v<11>(psi, op);
    // auto resv12 = calcE_v<12>(psi, op);
    // auto resv13 = calcE_v<13>(psi, op);
    // auto resv14 = calcE_vp<14>(psi, op);
    // auto resv15 = calcE_vp<15>(psi, op);
    // auto resv = 1. / 16. *
    //             (resv0 + resv1 + resv2 + resv3 + resv4 + resv5 + resv6 + resv7 + resv8 + resv9 + resv10 + resv11 + resv12 + resv13 + resv14 +
    //             resv15);
    // auto resh = 1. / 16. *
    //             (resh0 + resh1 + resh2 + resh3 + resh4 + resh5 + resh6 + resh7 + resh8 + resh9 + resh10 + resh11 + resh12 + resh13 + resh14 +
    //             resh15);
    auto res = 0.25 * (resh0 + resh2 + resv0 + resv1);
    return res;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array1_h()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos, 1);
    std::iota(res.begin() + pos + 2, res.end(), pos + 1);
    res[pos] = -1;
    res[pos + 1] = -2;
    return res;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array2_h()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos, 1);
    std::iota(res.begin() + pos + 2, res.end(), pos + 1);
    res[pos] = -3;
    res[pos + 1] = -4;
    return res;
}

template <std::size_t pos, typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
auto calcE_h(const Tensor<Scalar, 0, 16, Symmetry, ENABLE_AD>& psi, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    constexpr auto a1 = get_array1_h<pos>();
    constexpr auto a2 = get_array2_h<pos>();
    auto rhoA = psi.template contract<a1, a2, 2>(psi.adjoint().eval());
    auto Id2A = Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rhoA.uncoupledCodomain(), rhoA.uncoupledDomain(), rhoA.world());
    auto normA = rhoA.template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(Id2A.twist(0).twist(1)).trace();
    if(std::real(normA) < 0.) { Log::warning(Log::globalLevel, "rhoA: Negative norm detected."); }
    rhoA = (rhoA * (1. / normA)).eval();
    auto resA = rhoA.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(0, 0)).trace();
    // if constexpr(not ENABLE_AD) {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA);
    // } else {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA.val());
    // }
    return resA;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array1_h2()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos, 1);
    res[pos] = -1;
    std::iota(res.begin() + pos + 1, res.begin() + pos + 7, pos + 1);
    res[pos + 7] = -2;
    std::iota(res.begin() + pos + 8, res.end(), pos + 7);
    return res;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array2_h2()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos, 1);
    res[pos] = -3;
    std::iota(res.begin() + pos + 1, res.begin() + pos + 7, pos + 1);
    res[pos + 7] = -4;
    std::iota(res.begin() + pos + 8, res.end(), pos + 7);
    return res;
}

template <std::size_t pos, typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
auto calcE_h2(const Tensor<Scalar, 0, 16, Symmetry, ENABLE_AD>& psi, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    constexpr auto a1 = get_array1_h2<pos>();
    constexpr auto a2 = get_array2_h2<pos>();
    auto rhoA = psi.template contract<a1, a2, 2>(psi.adjoint().eval());
    auto Id2A = Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rhoA.uncoupledCodomain(), rhoA.uncoupledDomain(), rhoA.world());
    auto normA = rhoA.template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(Id2A.twist(0).twist(1)).trace();
    if(std::real(normA) < 0.) { Log::warning(Log::globalLevel, "rhoA: Negative norm detected."); }
    rhoA = (rhoA * (1. / normA)).eval();
    auto resA = rhoA.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(0, 0)).trace();
    // if constexpr(not ENABLE_AD) {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA);
    // } else {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA.val());
    // }
    return resA;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array1_hp()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos - 9, 1);
    res[pos - 9] = -1;
    std::iota(res.begin() + pos - 8, res.begin() + pos, pos - 8);
    res[pos] = -2;
    std::iota(res.begin() + pos + 1, res.end(), pos);
    return res;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array2_hp()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos - 9, 1);
    res[pos - 9] = -3;
    std::iota(res.begin() + pos - 8, res.begin() + pos, pos - 8);
    res[pos] = -4;
    std::iota(res.begin() + pos + 1, res.end(), pos);
    return res;
}

template <std::size_t pos, typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
auto calcE_hp(const Tensor<Scalar, 0, 16, Symmetry, ENABLE_AD>& psi, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    constexpr auto a1 = get_array1_hp<pos>();
    constexpr auto a2 = get_array2_hp<pos>();
    auto rhoA = psi.template contract<a1, a2, 2>(psi.adjoint().eval());
    auto Id2A = Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rhoA.uncoupledCodomain(), rhoA.uncoupledDomain(), rhoA.world());
    auto normA = rhoA.template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(Id2A.twist(0).twist(1)).trace();
    if(std::real(normA) < 0.) { Log::warning(Log::globalLevel, "rhoA: Negative norm detected."); }
    rhoA = (rhoA * (1. / normA)).eval();
    auto resA = rhoA.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(0, 0)).trace();
    // if constexpr(not ENABLE_AD) {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA);
    // } else {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA.val());
    // }
    return resA;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array1_v()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos, 1);
    res[pos] = -1;
    res[pos + 1] = pos + 1;
    res[pos + 2] = -2;
    std::iota(res.begin() + pos + 3, res.end(), pos + 2);
    return res;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array2_v()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos, 1);
    res[pos] = -3;
    res[pos + 1] = pos + 1;
    res[pos + 2] = -4;
    std::iota(res.begin() + pos + 3, res.end(), pos + 2);
    return res;
}

template <std::size_t pos, typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
auto calcE_v(const Tensor<Scalar, 0, 16, Symmetry, ENABLE_AD>& psi, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    constexpr auto a1 = get_array1_v<pos>();
    constexpr auto a2 = get_array2_v<pos>();
    auto rhoA = psi.template contract<a1, a2, 2>(psi.adjoint().eval());
    auto Id2A = Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rhoA.uncoupledCodomain(), rhoA.uncoupledDomain(), rhoA.world());
    auto normA = rhoA.template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(Id2A.twist(0).twist(1)).trace();
    if(std::real(normA) < 0.) { Log::warning(Log::globalLevel, "rhoA: Negative norm detected."); }
    rhoA = (rhoA * (1. / normA)).eval();
    auto resA = rhoA.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(0, 0)).trace();
    // if constexpr(not ENABLE_AD) {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA);
    // } else {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA.val());
    // }
    return resA;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array1_vp()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos - 6, 1);
    res[pos - 6] = -1;
    res[pos - 5] = pos - 5;
    res[pos - 4] = pos - 4;
    res[pos - 3] = pos - 3;
    res[pos - 2] = pos - 2;
    res[pos - 1] = pos - 1;
    res[pos] = -2;
    std::iota(res.begin() + pos + 1, res.end(), pos);
    return res;
}

template <std::size_t pos>
consteval std::array<int, 16> get_array2_vp()
{
    std::array<int, 16> res{};
    std::iota(res.begin(), res.begin() + pos - 6, 1);
    res[pos - 6] = -3;
    res[pos - 5] = pos - 5;
    res[pos - 4] = pos - 4;
    res[pos - 3] = pos - 3;
    res[pos - 2] = pos - 2;
    res[pos - 1] = pos - 1;
    res[pos] = -4;
    std::iota(res.begin() + pos + 1, res.end(), pos);
    return res;
}

template <std::size_t pos, typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
auto calcE_vp(const Tensor<Scalar, 0, 16, Symmetry, ENABLE_AD>& psi, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    constexpr auto a1 = get_array1_vp<pos>();
    constexpr auto a2 = get_array2_vp<pos>();
    auto rhoA = psi.template contract<a1, a2, 2>(psi.adjoint().eval());
    auto Id2A = Tensor<Scalar, 2, 2, Symmetry, false>::Identity(rhoA.uncoupledCodomain(), rhoA.uncoupledDomain(), rhoA.world());
    auto normA = rhoA.template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(Id2A.twist(0).twist(1)).trace();
    if(std::real(normA) < 0.) { Log::warning(Log::globalLevel, "rhoA: Negative norm detected."); }
    rhoA = (rhoA * (1. / normA)).eval();
    auto resA = rhoA.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(0, 0)).trace();
    // if constexpr(not ENABLE_AD) {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA);
    // } else {
    //     fmt::print("E({}, {}) = {}\n", pos, pos + 2, resA.val());
    // }
    return resA;
}

} // namespace Xped
#endif
