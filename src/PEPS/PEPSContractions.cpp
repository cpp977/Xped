#include "Xped/PEPS/PEPSContractions.hpp"

namespace Xped {

template <typename Derived1, typename Derived2>
std::pair<Tensor<typename TensorTraits<Derived1>::Scalar, 1, 3, typename TensorTraits<Derived1>::Symmetry>,
          Tensor<typename TensorTraits<Derived1>::Scalar, 3, 1, typename TensorTraits<Derived1>::Symmetry>>
decompose(const TensorBase<Derived1>& T1, const TensorBase<Derived2>& T2, const std::size_t max_nsv)
{
    using Scalar = typename TensorTraits<Derived1>::Scalar;
    using Symmetry = typename TensorTraits<Derived1>::Symmetry;
    // std::cout << T1.derived().coupledCodomain() << std::endl << T2.derived().coupledDomain() << std::endl;
    auto prod = T1 * T2;
    [[maybe_unused]] double t_weight;
    auto [U, S, Vdag] = prod.tSVD(max_nsv, 1.e-10, t_weight, false);
    std::cout << "Leading sv: " << S.block(0)(0, 0) << " in sector q: " << S.sector(0) << std::endl;
    // S.print(std::cout, true);
    // Svs(x, y) = S;
    auto isqrtS = S.sqrt().diag_inv().eval();
    // SPDLOG_CRITICAL("svs: ({}[{}],{}[{}])",
    //                 isqrtS.coupledDomain().fullDim(),
    //                 isqrtS.coupledDomain().dim(),
    //                 isqrtS.coupledCodomain().fullDim(),
    //                 isqrtS.coupledCodomain().dim());
    Tensor<Scalar, 1, 3, Symmetry> P1 = isqrtS * U.adjoint() * T1;
    Tensor<Scalar, 3, 1, Symmetry> P2 = T2 * Vdag.adjoint() * isqrtS;
    return std::make_pair(P1, P2);
}

} // namespace Xped
