#ifndef XPED_CTM_SOLVER_HPP_
#define XPED_CTM_SOLVER_HPP_

#include "Xped/Core/ScalarTraits.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/CTMOpts.hpp"

namespace Xped {

template <typename Scalar_, typename Symmetry_, Opts::CTMCheckpoint CPOpts, std::size_t TRank = 2>
class CTMSolver
{
public:
    using Scalar = Scalar_;
    using Symmetry = Symmetry_;
    template <typename Sym>
    using Hamiltonian = TwoSiteObservable<Sym>;

    explicit CTMSolver(Opts::CTM opts)
        : opts(opts)
    {
        Jack = CTM<Scalar, Symmetry, TRank, false>(opts.chi, opts.init); //, opts.cell);
    }

    template <typename HamScalar>
    typename ScalarTraits<Scalar>::Real
    solve(const std::shared_ptr<iPEPS<Scalar, Symmetry>>& Psi, Scalar* gradient, Hamiltonian<Symmetry>& H, bool CALC_GRAD = true);

    XPED_CONST CTM<Scalar, Symmetry, TRank, false>& getCTM() XPED_CONST { return Jack; }

private:
    CTM<Scalar, Symmetry, TRank> Jack;
    Opts::CTM opts;
    bool REINIT_ENV = true;
    typename ScalarTraits<Scalar>::Real grad_norm = 1000.;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/CTMSolver.cpp"
#endif

#endif
