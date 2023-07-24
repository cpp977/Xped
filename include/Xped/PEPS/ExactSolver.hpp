#ifndef XPED_EXACT_SOLVER_HPP_
#define XPED_EXACT_SOLVER_HPP_

#include <filesystem>

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/Util/Logging.hpp"

#include "Xped/Core/ScalarTraits.hpp"

#include "Xped/PEPS/ExactContractions.hpp"
#include "Xped/PEPS/iPEPS.hpp"

namespace Xped {

template <typename Scalar_, typename Symmetry_, std::size_t TRank = 2>
struct ExactSolver
{
    using Scalar = Scalar_;
    using Symmetry = Symmetry_;

    Tensor<Scalar, 2, 2, Symmetry> rho;
    Tensor<Scalar, 1, 1, Symmetry> rho1;

    Verbosity verbosity = Verbosity::ON_ENTRY;

    ExactSolver(Verbosity verbosity_)
        : verbosity(verbosity_){};

    template <typename HamScalar, bool AD>
    typename ScalarTraits<Scalar>::Real solve(std::shared_ptr<iPEPS<Scalar, Symmetry, true>> Psi, Scalar* gradient, Hamiltonian<double, Symmetry>& H)
    {
        Log::on_entry(verbosity, "  ExactSolver({}): UnitCell=({}x{})", Symmetry::name(), Psi->cell().Lx, Psi->cell().Ly);

        // Psi->normalize();
        stan::math::nested_rev_autodiff nested;
        iPEPS<Scalar, Symmetry, true, true> ad_Psi = *Psi;

        util::Stopwatch<> forward_t;
        auto Hobs = H.asObservable();
        auto [res, rho_, rho1_] = fourByfour(ad_Psi, Hobs);
        rho = rho_;
        rho1 = rho1_;

        auto forward_time = forward_t.time_string();
        Log::per_iteration(verbosity, "  {: >3} forward pass: {}", "•", forward_time);
        util::Stopwatch<> backward_t;
        stan::math::grad(res.vi_);
        auto backward_time = backward_t.time_string();
        Log::per_iteration(verbosity, "  {: >3} backward pass: {}", "•", backward_time);
        auto grad = ad_Psi.graddata();
        for(std::size_t i = 0; i < grad.size(); ++i) { gradient[i] = grad[i]; }
        double grad_norm = 0.;
        if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
            grad_norm = std::sqrt(
                std::inner_product(gradient,
                                   gradient + Psi->plainSize(),
                                   gradient,
                                   0.,
                                   std::plus<>(),
                                   [](const std::complex<double>& c1, const std::complex<double>& c2) { return std::real(c1 * std::conj(c2)); }));
        } else {
            grad_norm = std::sqrt(std::inner_product(gradient, gradient + Psi->plainSize(), gradient, 0.));
        }
        Log::per_iteration(verbosity, "  {: >3} E={:.8f}, |∇|={:.1e}", "•", res.val(), grad_norm);
        return res.val();
    }
};

} // namespace Xped

#endif
