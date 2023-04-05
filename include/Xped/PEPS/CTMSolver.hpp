#ifndef XPED_CTM_SOLVER_HPP_
#define XPED_CTM_SOLVER_HPP_

#include <filesystem>

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/Core/ScalarTraits.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/CTMOpts.hpp"

namespace Xped {

template <typename Scalar_, typename Symmetry_, Opts::CTMCheckpoint CPOpts = Opts::CTMCheckpoint{}, std::size_t TRank = 2>
class CTMSolver
{
public:
    using Scalar = Scalar_;
    using Symmetry = Symmetry_;
    template <typename Sym>
    using Hamiltonian = TwoSiteObservable<Sym>;

    CTMSolver() = default;

    explicit CTMSolver(Opts::CTM opts)
        : opts(opts)
    {
        Jack = CTM<Scalar, Symmetry, TRank, false>(opts.chi, opts.init); //, opts.cell);
        if(opts.load != "") {
            Jack.loadFromMatlab(std::filesystem::path(opts.load), "cpp", opts.qn_scale);
            REINIT_ENV = false;
            // Jack.info();
        }
    }

    template <typename HamScalar, bool AD>
    typename ScalarTraits<Scalar>::Real solve(std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi, Scalar* gradient, Hamiltonian<Symmetry>& H);

    XPED_CONST CTM<Scalar, Symmetry, TRank, false>& getCTM() XPED_CONST { return Jack; }

    void setCTM(XPED_CONST CTM<Scalar, Symmetry, TRank, false>& in) { Jack = in; }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("CTMSolver", ("solver", Jack), ("opts", opts), ("REINIT_ENV", REINIT_ENV), ("grad_norm", grad_norm));
    }

    Opts::CTM opts{};

private:
    CTM<Scalar, Symmetry, TRank> Jack;
    bool REINIT_ENV = true;
    typename ScalarTraits<Scalar>::Real grad_norm = 1000.;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/CTMSolver.cpp"
#endif

#endif
