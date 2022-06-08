#ifndef XPED_CTM_SOLVER_HPP_
#define XPED_CTM_SOLVER_HPP_

#include "Xped/Core/ScalarTraits.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/CTMOpts.hpp"

namespace Xped {

template <typename Scalar_, typename Symmetry_>
class CTMSolver
{
public:
    using Scalar = Scalar_;
    using Symmetry = Symmetry_;
    template <typename Sc, typename Sym>
    using Hamiltonian = Tensor<Sc, 2, 2, Sym, false>;

    explicit CTMSolver(Opts::CTM opts)
        : opts(opts)
    {}

    template <typename HamScalar>
    typename ScalarTraits<Scalar>::Real
    solve(const std::shared_ptr<iPEPS<Scalar, Symmetry>>& Psi, Scalar* gradient, const Hamiltonian<HamScalar, Symmetry>& H, bool CALC_GRAD = true);

private:
    CTM<Scalar, Symmetry, false> Jack;
    Opts::CTM opts;
    bool REINIT_ENV = true;
    typename ScalarTraits<Scalar>::Real grad_norm = 1000.;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/CTMSolver.cpp"
#endif

#endif
