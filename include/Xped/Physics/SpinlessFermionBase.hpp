#ifndef XPED_SPINLESS_FERMIONBASE_HPP_
#define XPED_SPINLESS_FERMIONBASE_HPP_

#include <sstream>

#include "Xped/Physics/Properties.hpp"
#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Physics/SpinOp.hpp"
#include "Xped/Physics/SubLattice.hpp"
#include "Xped/Physics/sites/SpinlessFermion.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

namespace Xped {

// Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/**
 * \ingroup Bases
 *
 * This class provides the local operators for fermions.
 *
 */
template <typename Symmetry_>
class SpinlessFermionBase : public SpinlessFermion<Symmetry_>
{
    using Scalar = double;

public:
    using Symmetry = Symmetry_;
    using OperatorType = SiteOperator<Scalar, Symmetry>;
    using qType = typename Symmetry::qType;

    SpinlessFermionBase() = default;

    /**
     * \param L_input : the amount of orbitals
     */
    SpinlessFermionBase(std::size_t L_input);

    /**amount of states*/
    inline std::size_t dim() const { return N_states; }

    /**amount of orbitals*/
    inline std::size_t orbitals() const { return N_orbitals; }

    ///\{
    /**
     * Annihilation operator
     * \param orbital : orbital index
     * \note The annihilation spinor is build as follows
     * \f$c^{1/2} = \left(
     * \begin{array}{c}
     * -c_{\downarrow} \\
     *  c_{\uparrow} \\
     * \end{array}
     * \right)\f$
     * where the upper component corresponds to \f$ m=+1/2\f$ and the lower to \f$ m=-1/2\f$.
     */
    OperatorType c(std::size_t orbital = 0) const;

    /**
     * Creation operator.
     * \param orbital : orbital index
     * \note The creation spinor is computed as \f$ \left(c^{1/2}\right)^\dagger\f$.
     * The definition of cdag which is consistent with this computation is:
     * \f$\left(c^{\dagger}\right)^{1/2} = \left(
     * \begin{array}{c}
     *  c^\dagger_{\uparrow} \\
     *  c^\dagger_{\downarrow} \\
     * \end{array}
     * \right)\f$
     * where the upper component corresponds to \f$ m=+1/2\f$ and the lower to \f$ m=-1/2\f$.
     */
    OperatorType cdag(std::size_t orbital = 0) const;

    /**
     * Fermionic sign for one orbital of a supersite.
     * \param orbital : orbital index
     */
    OperatorType sign_local(std::size_t orbital = 0) const;

    /**
     * Occupation number operator
     * \param orbital : orbital index
     */
    OperatorType n(std::size_t orbital = 0) const;

    /**Identity*/
    OperatorType Id(std::size_t orbital = 0) const;

    /**Returns the basis.*/
    Qbasis<Symmetry, 1> get_basis() const { return TensorBasis; }

private:
    OperatorType make_operator(const OperatorType& Op_1s, std::size_t orbital = 0, bool FERMIONIC = false, std::string label = "") const;
    std::size_t N_orbitals;
    std::size_t N_states;

    Qbasis<Symmetry, 1> TensorBasis; // Final basis for N_orbital sites

    // operators defined on zero orbitals
    OperatorType Id_vac, Zero_vac;
};

template <typename Symmetry>
SpinlessFermionBase<Symmetry>::SpinlessFermionBase(std::size_t L_input)
    : SpinlessFermion<Symmetry>()
    , N_orbitals(L_input)
{
    // create basis for zero orbitals
    qType Q = Symmetry::qvacuum();
    Qbasis<Symmetry, 1> vacuum;
    vacuum.push_back(Q, 1);

    // create operators for zero orbitals
    Zero_vac = OperatorType(Symmetry::qvacuum(), vacuum);
    Zero_vac.setZero();
    Id_vac = OperatorType(Symmetry::qvacuum(), vacuum);
    Id_vac.setIdentity();

    // create basis for N_orbitals fermionic sites
    if(N_orbitals == 1) {
        TensorBasis = this->basis_1s();
    } else if(N_orbitals == 0) {
        TensorBasis = vacuum;
    } else {
        TensorBasis = this->basis_1s().combine(this->basis_1s()).forgetHistory();
        for(std::size_t o = 2; o < N_orbitals; o++) { TensorBasis = TensorBasis.combine(this->basis_1s()).forgetHistory(); }
    }

    N_states = TensorBasis.dim();
}

template <typename Symmetry>
SiteOperator<double, Symmetry>
SpinlessFermionBase<Symmetry>::make_operator(const OperatorType& Op_1s, std::size_t orbital, bool FERMIONIC, std::string label) const
{
    assert(orbital < N_orbitals);
    OperatorType out;
    if(N_orbitals == 1) {
        out = Op_1s;
        out.label() = label;
        return out;
    } else if(N_orbitals == 0) {
        return Zero_vac;
    } else {
        OperatorType stringOp;
        if(FERMIONIC) {
            stringOp = this->Id_1s();
        } else {
            stringOp = this->Id_1s();
        }
        bool TOGGLE = false;
        if(orbital == 0) {
            out = OperatorType::outerprod(Op_1s, this->Id_1s(), Op_1s.Q);
            TOGGLE = true;
        } else {
            if(orbital == 1) {
                out = OperatorType::outerprod(stringOp, Op_1s, Op_1s.Q);
                TOGGLE = true;
            } else {
                out = OperatorType::outerprod(stringOp, stringOp, Symmetry::qvacuum());
            }
        }
        for(std::size_t o = 2; o < N_orbitals; o++) {
            if(orbital == o) {
                out = OperatorType::outerprod(out, Op_1s, Op_1s.Q);
                TOGGLE = true;
            } else if(TOGGLE == false) {
                out = OperatorType::outerprod(out, stringOp, Symmetry::qvacuum());
            } else if(TOGGLE == true) {
                out = OperatorType::outerprod(out, this->Id_1s(), Op_1s.Q);
            }
        }
        out.label() = label;
        return out;
    }
}

template <typename Symmetry>
SiteOperator<double, Symmetry> SpinlessFermionBase<Symmetry>::c(std::size_t orbital) const
{
    return make_operator(this->c_1s(), orbital, PROP::FERMIONIC, "c");
}

template <typename Symmetry>
SiteOperator<double, Symmetry> SpinlessFermionBase<Symmetry>::cdag(std::size_t orbital) const
{
    return c(orbital).adjoint();
}

template <typename Symmetry>
SiteOperator<double, Symmetry> SpinlessFermionBase<Symmetry>::sign_local(std::size_t orbital) const
{
    return make_operator(this->F_1s(), orbital, PROP::FERMIONIC, "F");
}

template <typename Symmetry>
SiteOperator<double, Symmetry> SpinlessFermionBase<Symmetry>::n(std::size_t orbital) const
{
    return make_operator(this->n_1s(), orbital, PROP::NON_FERMIONIC, "n");
}

template <typename Symmetry>
SiteOperator<double, Symmetry> SpinlessFermionBase<Symmetry>::Id(std::size_t orbital) const
{
    return make_operator(this->Id_1s(), orbital, PROP::NON_FERMIONIC, "Id");
}

} // namespace Xped
#endif
