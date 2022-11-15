#include "spdlog/spdlog.h"

#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/PEPS/PEPSContractions.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/CoeffUnaryOp.hpp"
#include "Xped/Core/DiagCoeffUnaryOp.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Derived1, typename Derived2>
auto
// std::pair<Tensor<typename TensorTraits<Derived1>::Scalar, 1, 3, typename TensorTraits<Derived1>::Symmetry>,
//           Tensor<typename TensorTraits<Derived1>::Scalar, 3, 1, typename TensorTraits<Derived1>::Symmetry>>
decompose(XPED_CONST TensorBase<Derived1>& T1, XPED_CONST TensorBase<Derived2>& T2, const std::size_t max_nsv)
{
    // using Scalar = typename TensorTraits<Derived1>::Scalar;
    // using Symmetry = typename TensorTraits<Derived1>::Symmetry;
    // std::cout << T1.derived().coupledCodomain() << std::endl << T2.derived().coupledDomain() << std::endl;
    auto prod = T1 * T2;
    [[maybe_unused]] double t_weight;
    auto [U, S, Vdag] = prod.tSVD(max_nsv, 1.e-10, t_weight, false);
    SPDLOG_INFO("Leading sv: {}", PlainInterface::getVal(S.block(0), 0, 0));
    // std::cout << "Leading sv: " << S.block(0)(0, 0) << " in sector q: " << S.sector(0) << std::endl;
    // S.print(std::cout, true);
    // Svs(x, y) = S;
    auto isqrtS = S.diag_sqrt().eval().diag_inv().eval();
    // SPDLOG_CRITICAL("svs: ({}[{}],{}[{}])",
    //                 isqrtS.coupledDomain().fullDim(),
    //                 isqrtS.coupledDomain().dim(),
    //                 isqrtS.coupledCodomain().fullDim(),
    //                 isqrtS.coupledCodomain().dim());
    auto P1 = isqrtS * U.adjoint() * T1;
    auto P2 = T2 * Vdag.adjoint() * isqrtS;
    // (P1 * P2).eval().print(std::cout, true);
    // (P2 * P1).eval().print(std::cout, true);
    return std::make_pair(P1, P2);
}

template <typename Scalar, typename Symmetry, typename AllocationPolicy>
std::pair<Tensor<Scalar, 1, 3, Symmetry, true, AllocationPolicy>, Tensor<Scalar, 3, 1, Symmetry, true, AllocationPolicy>>
decompose(XPED_CONST Tensor<Scalar, 3, 3, Symmetry, true, AllocationPolicy>& T1,
          XPED_CONST Tensor<Scalar, 3, 3, Symmetry, true, AllocationPolicy>& T2,
          const std::size_t max_nsv)
{
    auto prod = T1 * T2;
    [[maybe_unused]] double t_weight;
    auto [U, S, Vdag] = prod.tSVD(max_nsv, 1.e-10, t_weight, false);
    SPDLOG_INFO("Leading sv: {}", PlainInterface::getVal(S.val().block(0), 0, 0));
    // std::cout << "Leading sv: " << S.block(0)(0, 0) << " in sector q: " << S.sector(0) << std::endl;
    // S.print(std::cout, true);
    // Svs(x, y) = S;
    auto isqrtS = S.diag_sqrt().diag_inv().eval();
    // SPDLOG_CRITICAL("svs: ({}[{}],{}[{}])",
    //                 isqrtS.coupledDomain().fullDim(),
    //                 isqrtS.coupledDomain().dim(),
    //                 isqrtS.coupledCodomain().fullDim(),
    //                 isqrtS.coupledCodomain().dim());
    Tensor<Scalar, 1, 3, Symmetry, true> P1 = isqrtS * U.adjoint() * T1;
    Tensor<Scalar, 3, 1, Symmetry, true> P2 = T2 * Vdag.adjoint() * isqrtS;
    return std::make_pair(P1, P2);
}

template <typename Scalar, typename Symmetry, typename AllocationPolicy>
std::pair<Tensor<Scalar, 1, 2, Symmetry, true, AllocationPolicy>, Tensor<Scalar, 2, 1, Symmetry, true, AllocationPolicy>>
decompose(XPED_CONST Tensor<Scalar, 2, 2, Symmetry, true, AllocationPolicy>& T1,
          XPED_CONST Tensor<Scalar, 2, 2, Symmetry, true, AllocationPolicy>& T2,
          const std::size_t max_nsv)
{
    auto prod = T1 * T2;
    [[maybe_unused]] double t_weight;
    auto [U, S, Vdag] = prod.tSVD(max_nsv, 1.e-10, t_weight, false);
    SPDLOG_INFO("Leading sv: {}", PlainInterface::getVal(S.val().block(0), 0, 0));
    // std::cout << "Leading sv: " << S.block(0)(0, 0) << " in sector q: " << S.sector(0) << std::endl;
    // S.print(std::cout, true);
    // Svs(x, y) = S;
    auto isqrtS = S.diag_sqrt().diag_inv().eval();
    // SPDLOG_CRITICAL("svs: ({}[{}],{}[{}])",
    //                 isqrtS.coupledDomain().fullDim(),
    //                 isqrtS.coupledDomain().dim(),
    //                 isqrtS.coupledCodomain().fullDim(),
    //                 isqrtS.coupledCodomain().dim());
    Tensor<Scalar, 1, 2, Symmetry, true> P1 = isqrtS * U.adjoint() * T1;
    Tensor<Scalar, 2, 1, Symmetry, true> P2 = T2 * Vdag.adjoint() * isqrtS;
    return std::make_pair(P1, P2);
}

template <typename Scalar, typename Symmetry, typename AllocationPolicy, typename DerivedL, typename DerivedT, typename DerivedR, typename DerivedB>
Tensor<Scalar, 2, 3, Symmetry, false, AllocationPolicy> applyWeights(XPED_CONST Tensor<Scalar, 2, 3, Symmetry, false, AllocationPolicy>& A,
                                                                     XPED_CONST TensorBase<DerivedL>& wL,
                                                                     XPED_CONST TensorBase<DerivedT>& wT,
                                                                     XPED_CONST TensorBase<DerivedR>& wR,
                                                                     XPED_CONST TensorBase<DerivedB>& wB)
{
    return A.template contract<std::array{1, -2, -3, -4, -5}, std::array{-1, 1}, 2>(wL.eval())
        .template contract<std::array{-1, 1, -3, -4, -5}, std::array{-2, 1}, 2>(wT.eval())
        .template contract<std::array{-1, -2, 1, -4, -5}, std::array{1, -3}, 2>(wR.eval())
        .template contract<std::array{-1, -2, -3, 1, -5}, std::array{1, -4}, 2>(wB.eval());
}
} // namespace Xped
