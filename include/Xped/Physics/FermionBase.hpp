#ifndef XPED_FERMIONBASE_HPP_
#define XPED_FERMIONBASE_HPP_

#include <sstream>

#include "Xped/Physics/Properties.hpp"
#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Physics/SpinIndex.hpp"
#include "Xped/Physics/SpinOp.hpp"
#include "Xped/Physics/SubLattice.hpp"
#include "Xped/Physics/sites/Fermion.hpp"
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
class FermionBase : public Fermion<Symmetry_>
{
    using Scalar = double;

public:
    using Symmetry = Symmetry_;
    using OperatorType = SiteOperator<Scalar, Symmetry>;
    using OperatorTypeC = SiteOperator<std::complex<Scalar>, Symmetry>;
    using qType = typename Symmetry::qType;

    FermionBase() = default;

    /**
     * \param L_input : the amount of orbitals
     */
    FermionBase(std::size_t L_input, bool REMOVE_DOUBLE = false, bool REMVOVE_EMPTY = false, bool REMOVE_SINGLE = false);

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
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), OperatorType>::type c(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), OperatorType>::type c(SPIN_INDEX sigma, std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(), OperatorType>::type
    c(SPIN_INDEX sigma, SUB_LATTICE G, std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(), OperatorType>::type c(SUB_LATTICE G, std::size_t orbital = 0) const;

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
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), OperatorType>::type cdag(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), OperatorType>::type cdag(SPIN_INDEX sigma,
                                                                                                        std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(), OperatorType>::type
    cdag(SPIN_INDEX sigma, SUB_LATTICE G, std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(), OperatorType>::type cdag(SUB_LATTICE G, std::size_t orbital = 0) const;

    /**
     * Fermionic sign for the hopping between two orbitals of nearest-neighbour supersites of a ladder.
     * \param orb1 : orbital on supersite i
     * \param orb2 : orbital on supersite i+1
     */
    OperatorType sign(std::size_t orb1 = 0, std::size_t orb2 = 0) const;

    /**
     * Fermionic sign for one orbital of a supersite.
     * \param orbital : orbital index
     */
    OperatorType sign_local(std::size_t orbital = 0) const;

    /**
     * Occupation number operator
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<true, OperatorType>::type n(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_SPIN_SU2(), OperatorType>::type n(SPIN_INDEX sigma, std::size_t orbital = 0) const;

    /**
     * Double occupation
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_CHARGE_SU2(), OperatorType>::type d(std::size_t orbital = 0) const;

    /**
     * Spinon density \f$n_s=n-2d\f$
     * \param orbital : orbital index
     */
    OperatorType ns(std::size_t orbital = 0) const;

    /**
     * Holon density \f$n_h=2d-n-1=1-n_s\f$
     * \param orbital : orbital index
     */
    OperatorType nh(std::size_t orbital = 0) const;
    ///\}

    ///\{
    /**
     * Orbital spin
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_SPIN_SU2(), OperatorType>::type S(std::size_t orbital = 0) const;

    /**
     * Orbital spin†
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_SPIN_SU2(), OperatorType>::type Sdag(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_SPIN_SU2(), OperatorType>::type Sz(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_SPIN_SU2(), OperatorType>::type Sp(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_SPIN_SU2(), OperatorType>::type Sm(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::NO_SPIN_SYM(), OperatorType>::type Sx(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::NO_SPIN_SYM(), OperatorTypeC>::type Sy(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::NO_SPIN_SYM(), OperatorType>::type iSy(std::size_t orbital = 0) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_SPIN_SU2(), OperatorType>::type Scomp(SPINOP_LABEL Sa, int orbital) const
    {
        assert(Sa != SPINOP_LABEL::SY);
        OperatorType out;
        if constexpr(Dummy::NO_SPIN_SYM()) {
            if(Sa == SPINOP_LABEL::SX) {
                out = Sx(orbital);
            } else if(Sa == SPINOP_LABEL::iSY) {
                out = iSy(orbital);
            } else if(Sa == SPINOP_LABEL::SZ) {
                out = Sz(orbital);
            } else if(Sa == SPINOP_LABEL::SP) {
                out = Sp(orbital);
            } else if(Sa == SPINOP_LABEL::SM) {
                out = Sm(orbital);
            }
        } else {
            if(Sa == SPINOP_LABEL::SZ) {
                out = Sz(orbital);
            } else if(Sa == SPINOP_LABEL::SP) {
                out = Sp(orbital);
            } else if(Sa == SPINOP_LABEL::SM) {
                out = Sm(orbital);
            }
        }
        return out;
    };
    ///\}

    // template <class Dummy = Symmetry>
    // typename std::enable_if<!Dummy::IS_SPIN_SU2(), OperatorType>::type Rcomp(SPINOP_LABEL Sa, int orbital) const;

    ///\{
    /**
     * Orbital Isospin
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_CHARGE_SU2(), OperatorType>::type T(std::size_t orbital = 0) const;

    /**
     * Orbital Isospin†
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::IS_CHARGE_SU2(), OperatorType>::type Tdag(std::size_t orbital = 0) const;

    /**
     * Isospin z-component
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_CHARGE_SU2(), OperatorType>::type Tz(std::size_t orbital = 0) const;

    /**
     * Isospin x-component
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::NO_CHARGE_SYM(), OperatorType>::type Tx(std::size_t orbital = 0, SUB_LATTICE G = A) const;

    /**
     * Isospin y-component
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<Dummy::NO_CHARGE_SYM(), OperatorType>::type iTy(std::size_t orbital = 0, SUB_LATTICE G = A) const;

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_CHARGE_SU2(), OperatorType>::type Tp(std::size_t orbital = 0, SUB_LATTICE G = A) const
    {
        double factor = static_cast<double>(static_cast<int>(G));
        return factor * cc(orbital);
    };

    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_CHARGE_SU2(), OperatorType>::type Tm(std::size_t orbital = 0, SUB_LATTICE G = A) const
    {
        double factor = static_cast<double>(static_cast<int>(G));
        return factor * cdagcdag(orbital);
    };

    ///\{
    /**
     * Orbital pairing η
     * \param orbital : orbital index
     */
    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_CHARGE_SU2(), OperatorType>::type cc(std::size_t orbital = 0) const;

    /**
     * Orbital paring η†
     * \param orbital : orbital index
     **/
    template <class Dummy = Symmetry>
    typename std::enable_if<!Dummy::IS_CHARGE_SU2(), OperatorType>::type cdagcdag(std::size_t orbital = 0) const;
    ///\}

    /**Identity*/
    OperatorType Id(std::size_t orbital = 0) const;

    // /**Returns an array of size dim() with zeros.*/
    // ArrayXd ZeroField() const { return ArrayXd::Zero(N_orbitals); }

    // /**Returns an array of size dim()xdim() with zeros.*/
    // ArrayXXd ZeroHopping() const { return ArrayXXd::Zero(N_orbitals, N_orbitals); }

    /**
     * Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
     * \param U : \f$U\f$ for each orbital
     * \param Uph : particle-hole symmetric \f$U\f$ for each orbital (times \f$(n_{\uparrow}-1/2)(n_{\downarrow}-1/2)+1/4\f$)
     * \param Eorb : \f$\varepsilon\f$ onsite energy for each orbital
     * \param t : \f$t\f$
     * \param V : \f$V\f$
     * \param Vz : \f$V_z\f$
     * \param Vxy : \f$V_{xy}\f$
     * \param J : \f$J\f$
     */
    // template <typename Scalar_>
    // OperatorType HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& Uph,
    //                                 const Array<Scalar_, Dynamic, Dynamic>& t,
    //                                 const Array<Scalar_, Dynamic, Dynamic>& V,
    //                                 const Array<Scalar_, Dynamic, Dynamic>& J) const;

    // template <typename Scalar_, typename Dummy = Symmetry>
    // typename std::enable_if<Dummy::ABELIAN, OperatorType>::type HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& U,
    //                                                                                const Array<Scalar_, Dynamic, 1>& Uph,
    //                                                                                const Array<Scalar_, Dynamic, 1>& Eorb,
    //                                                                                const Array<Scalar_, Dynamic, 1>& Bz,
    //                                                                                const Array<Scalar_, Dynamic, Dynamic>& t,
    //                                                                                const Array<Scalar_, Dynamic, Dynamic>& V,
    //                                                                                const Array<Scalar_, Dynamic, Dynamic>& Vz,
    //                                                                                const Array<Scalar_, Dynamic, Dynamic>& Vxy,
    //                                                                                const Array<Scalar_, Dynamic, Dynamic>& Jz,
    //                                                                                const Array<Scalar_, Dynamic, Dynamic>& Jxy,
    //                                                                                const Array<Scalar_, Dynamic, Dynamic>& C) const;

    // template <typename Scalar_, typename Dummy = Symmetry>
    // typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), OperatorType>::type
    // HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& U,
    //                    const Array<Scalar_, Dynamic, 1>& Uph,
    //                    const Array<Scalar_, Dynamic, 1>& Eorb,
    //                    const Array<Scalar_, Dynamic, Dynamic>& t,
    //                    const Array<Scalar_, Dynamic, Dynamic>& V,
    //                    const Array<Scalar_, Dynamic, Dynamic>& Vz,
    //                    const Array<Scalar_, Dynamic, Dynamic>& Vxy,
    //                    const Array<Scalar_, Dynamic, Dynamic>& J) const;

    // template <typename Scalar_, typename Dummy = Symmetry>
    // typename std::enable_if<Dummy::NO_SPIN_SYM() and Dummy::IS_CHARGE_SU2(), SiteOperatorQ<Symmetry_, Eigen::Matrix<Scalar_, -1, -1>>>::type
    // HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& Uph,
    //                    const Array<Scalar_, Dynamic, Dynamic>& t,
    //                    const Array<Scalar_, Dynamic, Dynamic>& V,
    //                    const Array<Scalar_, Dynamic, Dynamic>& Jz,
    //                    const Array<Scalar_, Dynamic, Dynamic>& Jxy,
    //                    const Array<Scalar_, Dynamic, 1>& Bz,
    //                    const Array<Scalar_, Dynamic, 1>& Bx) const;

    // template <typename Scalar_, typename Dummy = Symmetry>
    // typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry_, Eigen::Matrix<Scalar_, -1, -1>>>::type
    // coupling_Bx(const Array<double, Dynamic, 1>& Bx) const;

    // template <typename Dummy = Symmetry>
    // typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry_, Eigen::Matrix<complex<double>, -1, -1>>>::type
    // coupling_By(const Array<double, Dynamic, 1>& By) const;

    // template <typename Scalar_, typename Dummy = Symmetry>
    // typename std::enable_if<Dummy::IS_TRIVIAL, SiteOperatorQ<Symmetry_, Eigen::Matrix<Scalar_, -1, -1>>>::type
    // coupling_singleFermion(const Array<double, Dynamic, 1>& Fp) const;

    // template <typename Scalar_, typename Dummy = Symmetry>
    // typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry_, Eigen::Matrix<Scalar_, -1, -1>>>::type
    // coupling_XYZspin(const Array<double, Dynamic, Dynamic>& Jx,
    //                  const Array<double, Dynamic, Dynamic>& Jy,
    //                  const Array<double, Dynamic, Dynamic>& Jz) const;

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
FermionBase<Symmetry>::FermionBase(std::size_t L_input, bool REMOVE_DOUBLE, bool REMVOVE_EMPTY, bool REMOVE_SINGLE)
    : Fermion<Symmetry>(REMOVE_DOUBLE, REMVOVE_EMPTY, REMOVE_SINGLE)
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
FermionBase<Symmetry>::make_operator(const OperatorType& Op_1s, std::size_t orbital, bool FERMIONIC, std::string label) const
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
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::c(std::size_t orbital) const
{
    return make_operator(this->c_1s(), orbital, PROP::FERMIONIC, "c");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::cdag(std::size_t orbital) const
{
    return c(orbital).adjoint();
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::c(SPIN_INDEX sigma, std::size_t orbital) const
{
    std::stringstream ss;
    ss << "c" << sigma;
    return make_operator(this->c_1s(sigma), orbital, PROP::FERMIONIC, ss.str());
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::cdag(SPIN_INDEX sigma, std::size_t orbital) const
{
    //	return c(sigma,orbital).adjoint();
    std::stringstream ss;
    ss << "c†" << sigma;
    return make_operator(this->cdag_1s(sigma), orbital, PROP::FERMIONIC, ss.str());
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::c(SPIN_INDEX sigma, SUB_LATTICE G, std::size_t orbital) const
{
    std::stringstream ss;
    return make_operator(this->c_1s(sigma, G), orbital, PROP::FERMIONIC, ss.str());
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::cdag(SPIN_INDEX sigma, SUB_LATTICE G, std::size_t orbital) const
{
    return c(sigma, G, orbital).adjoint();
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::c(SUB_LATTICE G, std::size_t orbital) const
{
    std::stringstream ss;
    ss << "c" << G;
    return make_operator(this->c_1s(G), orbital, PROP::FERMIONIC, ss.str());
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type
FermionBase<Symmetry>::cdag(SUB_LATTICE G, std::size_t orbital) const
{
    return c(G, orbital).adjoint();
}

template <typename Symmetry>
SiteOperator<double, Symmetry> FermionBase<Symmetry>::sign_local(std::size_t orbital) const
{
    return make_operator(this->F_1s(), orbital, PROP::FERMIONIC, "F");
}

template <typename Symmetry>
SiteOperator<double, Symmetry> FermionBase<Symmetry>::sign(std::size_t orb1, std::size_t orb2) const
{
    OperatorType Oout;
    if(N_orbitals == 1) {
        Oout = this->F_1s();
        Oout.label() = "sign";
        return Oout;
    } else if(N_orbitals == 0) {
        return Zero_vac;
    } else {
        Oout = Id();
        for(int i = orb1; i < N_orbitals; ++i) { Oout = Oout * (nh(i) - ns(i)); }
        for(int i = 0; i < orb2; ++i) { Oout = Oout * (nh(i) - ns(i)); }
        Oout.label() = "sign";
        return Oout;
    }
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<true, SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::n(std::size_t orbital) const
{
    return make_operator(this->n_1s(), orbital, PROP::NON_FERMIONIC, "n");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::n(SPIN_INDEX sigma,
                                                                                                              std::size_t orbital) const
{
    std::stringstream ss;
    ss << "n" << sigma;
    return make_operator(this->n_1s(sigma), orbital, PROP::NON_FERMIONIC, ss.str());
}

template <typename Symmetry>
SiteOperator<double, Symmetry> FermionBase<Symmetry>::ns(std::size_t orbital) const
{
    return make_operator(this->ns_1s(), orbital, PROP::NON_FERMIONIC, "ns");
}

template <typename Symmetry>
SiteOperator<double, Symmetry> FermionBase<Symmetry>::nh(std::size_t orbital) const
{
    return make_operator(this->nh_1s(), orbital, PROP::NON_FERMIONIC, "nh");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::d(std::size_t orbital) const
{
    return make_operator(this->d_1s(), orbital, PROP::NON_FERMIONIC, "d");
}

// template <typename Symmetry>
// template <typename Dummy>
// typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry, Eigen::Matrix<complex<double>, Eigen::Dynamic, Eigen::Dynamic>>>::type
// FermionBase<Symmetry>::Rcomp(SPINOP_LABEL Sa, int orbital) const
// {
//     assert(orbital < N_orbitals);
//     SiteOperatorQ<Symmetry, Eigen::Matrix<complex<double>, Eigen::Dynamic, Eigen::Dynamic>> Oout;
//     SiteOperatorQ<Symmetry, Eigen::Matrix<complex<double>, Eigen::Dynamic, Eigen::Dynamic>> Scomp_cmplx =
//         Scomp(Sa, orbital).template cast<complex<double>>();
//     if(Sa == iSY) {
//         Oout = 2. * M_PI * Scomp_cmplx;
//     } else {
//         Oout = 2. * 1.i * M_PI * Scomp_cmplx;
//     }
//     Oout.data() = Oout.data().exp(1.);

//     cout << "Rcomp=" << Oout << endl << endl;
//     // cout << "Re=" << Mtmp.exp().real() << endl << endl;
//     // cout << "Im=" << Mtmp.exp().imag() << endl << endl;
//     // cout << "Op=" << Op << endl << endl;

//     return Oout; // SiteOperator<Symmetry,complex<double>>(Op,getQ(Sa));
// }

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::S(std::size_t orbital) const
{
    return make_operator(this->S_1s(), orbital, PROP::NON_FERMIONIC, "S");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Sdag(std::size_t orbital) const
{
    return S(orbital).adjoint();
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Sz(std::size_t orbital) const
{
    return make_operator(this->Sz_1s(), orbital, PROP::NON_FERMIONIC, "Sz");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Sp(std::size_t orbital) const
{
    return make_operator(this->Sp_1s(), orbital, PROP::NON_FERMIONIC, "Sp");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Sm(std::size_t orbital) const
{
    return make_operator(this->Sm_1s(), orbital, PROP::NON_FERMIONIC, "Sm");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Sx(std::size_t orbital) const
{
    OperatorType out = 0.5 * (Sp(orbital) + Sm(orbital));
    out.label() = "Sx";
    return out;
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperator<std::complex<double>, Symmetry>>::type FermionBase<Symmetry>::Sy(std::size_t orbital) const
{
    using namespace std::complex_literals;
    OperatorTypeC out = -1i * 0.5 * (Sp(orbital) - Sm(orbital));
    out.label() = "Sy";
    return out;
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::iSy(std::size_t orbital) const
{
    OperatorType out = 0.5 * (Sp(orbital) - Sm(orbital));
    out.label() = "iSy";
    return out;
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::T(std::size_t orbital) const
{
    return make_operator(this->T_1s(), orbital, PROP::NON_FERMIONIC, "T");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Tdag(std::size_t orbital) const
{
    return T(orbital).adjoint();
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::cc(std::size_t orbital) const
{
    return make_operator(this->cc_1s(), orbital, PROP::NON_FERMIONIC, "cc");
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::cdagcdag(std::size_t orbital) const
{
    //	return cc(orbital).adjoint();
    return make_operator(this->cdagcdag_1s(), orbital, PROP::NON_FERMIONIC, "c†c†");
}

// SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
// FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >:: Tp (std::size_t orbital) const
//{
//	return Eta(orbital);
// }

// SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
// FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >:: Tm (std::size_t orbital) const
//{
//	return Eta(orbital).adjoint();
// }

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Tz(std::size_t orbital) const
{
    OperatorType out = 0.5 * (n(orbital) - Id());
    out.label() = "Tz=1/2*(n-Id)";
    return out;
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::NO_CHARGE_SYM(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::Tx(std::size_t orbital,
                                                                                                                SUB_LATTICE G) const
{
    OperatorType out = 0.5 * (Tp(orbital, G) + Tm(orbital, G));
    out.label() = "Tx";
    return out;
}

template <typename Symmetry>
template <typename Dummy>
typename std::enable_if<Dummy::NO_CHARGE_SYM(), SiteOperator<double, Symmetry>>::type FermionBase<Symmetry>::iTy(std::size_t orbital,
                                                                                                                 SUB_LATTICE G) const
{
    OperatorType out = 0.5 * (Tp(orbital, G) - Tm(orbital, G));
    out.label() = "iTy";
    return out;
}

template <typename Symmetry>
SiteOperator<double, Symmetry> FermionBase<Symmetry>::Id(std::size_t orbital) const
{
    return make_operator(this->Id_1s(), orbital, PROP::NON_FERMIONIC, "Id");
}

// template <typename Symmetry>
// template <typename Scalar_>
// SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>>
// FermionBase<Symmetry>::HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& Uph,
//                                            const Array<Scalar_, Dynamic, Dynamic>& t,
//                                            const Array<Scalar_, Dynamic, Dynamic>& V,
//                                            const Array<Scalar_, Dynamic, Dynamic>& J) const
// {
//     SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>> Oout(Symmetry::qvacuum(), TensorBasis);

//     for(int i = 0; i < N_orbitals; ++i)
//         for(int j = 0; j < N_orbitals; ++j) {
//             auto G1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1, i)));
//             auto G2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1, j)));
//             if(t(i, j) != 0.) {
//                 if constexpr(Symmetry::IS_SPIN_SU2() and Symmetry::IS_CHARGE_SU2()) {
//                     Oout += -t(i, j) * std::sqrt(2.) * std::sqrt(2.) *
//                             OperatorType::prod(cdag(G1, i), c(G2, j), Symmetry::qvacuum()).template cast<Scalar_>();
//                 } else if constexpr(Symmetry::IS_SPIN_SU2() and !Symmetry::IS_CHARGE_SU2()) {
//                     Oout += -t(i, j) * std::sqrt(2.) * (OperatorType::prod(cdag(i), c(j), Symmetry::qvacuum())).template cast<Scalar_>();
//                     Oout += -t(j, i) * std::sqrt(2.) * (OperatorType::prod(c(i), cdag(j), Symmetry::qvacuum())).template cast<Scalar_>();
//                 } else if constexpr(!Symmetry::IS_SPIN_SU2() and Symmetry::IS_CHARGE_SU2()) {
//                     Oout += -t(i, j) * std::sqrt(2.) *
//                             (OperatorType::prod(cdag(UP, G1, i), c(UP, G2, j), Symmetry::qvacuum()) +
//                              OperatorType::prod(cdag(DN, G1, i), c(DN, G2, j), Symmetry::qvacuum()))
//                                 .template cast<Scalar_>();
//                 } else if constexpr(!Symmetry::IS_SPIN_SU2() and !Symmetry::IS_CHARGE_SU2()) {
//                     Oout += -t(i, j) * (cdag(UP, i) * c(UP, j) + cdag(DN, i) * c(DN, j)).template cast<Scalar_>();
//                     Oout += -t(j, i) * (cdag(UP, j) * c(UP, i) + cdag(DN, j) * c(DN, i)).template cast<Scalar_>();
//                 } else {
//                     // static_assert(false, "You use a symmetry combination for which there is no implementation of the hopping part in
//                     // FermionBase::HubbardHamiltonian()");
//                 }
//             }
//             if(V(i, j) != 0.) {
//                 if constexpr(Symmetry::IS_CHARGE_SU2()) {
//                     Oout += V(i, j) * std::sqrt(3.) * (OperatorType::prod(Tdag(i), T(j), Symmetry::qvacuum())).template cast<Scalar_>();
//                 } else {
//                     Oout += V(i, j) * (OperatorType::prod(Tz(i), Tz(j), Symmetry::qvacuum())).template cast<Scalar_>();
//                     Oout += 0.5 * V(i, j) *
//                             (OperatorType::prod(Tp(i, G1), Tm(j, G2), Symmetry::qvacuum()) +
//                              OperatorType::prod(Tm(i, G1), Tp(j, G2), Symmetry::qvacuum()))
//                                 .template cast<Scalar_>();
//                 }
//             }
//             if(J(i, j) != 0.) {
//                 if constexpr(Symmetry::IS_SPIN_SU2()) {
//                     Oout += J(i, j) * std::sqrt(3.) * (OperatorType::prod(Sdag(i), S(j), Symmetry::qvacuum())).template cast<Scalar_>();
//                 } else {
//                     Oout += J(i, j) * (OperatorType::prod(Sz(i), Sz(j), Symmetry::qvacuum())).template cast<Scalar_>();
//                     Oout += 0.5 * J(i, j) *
//                             (OperatorType::prod(Sp(i), Sm(j), Symmetry::qvacuum()) + OperatorType::prod(Sm(i), Sp(j), Symmetry::qvacuum()))
//                                 .template cast<Scalar_>();
//                 }
//             }
//         }

//     for(int i = 0; i < N_orbitals; ++i) {
//         if(Uph(i) != 0. and Uph(i) != std::numeric_limits<double>::infinity()) { Oout += 0.5 * Uph(i) * nh(i).template cast<Scalar_>(); }
//     }
//     Oout.label() = "Hloc";
//     return Oout;
// }

// template <typename Symmetry>
// template <typename Scalar_, typename Dummy>
// typename std::enable_if<Dummy::ABELIAN, SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>>>::type
// FermionBase<Symmetry>::HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& U,
//                                            const Array<Scalar_, Dynamic, 1>& Uph,
//                                            const Array<Scalar_, Dynamic, 1>& Eorb,
//                                            const Array<Scalar_, Dynamic, 1>& Bz,
//                                            const Array<Scalar_, Dynamic, Dynamic>& t,
//                                            const Array<Scalar_, Dynamic, Dynamic>& V,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Vz,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Vxy,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Jz,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Jxy,
//                                            const Array<Scalar_, Dynamic, Dynamic>& C) const
// {
//     auto Oout = HubbardHamiltonian<Scalar_>(Uph, t, ZeroHopping(), ZeroHopping());

//     for(int i = 0; i < N_orbitals; ++i)
//         for(int j = 0; j < N_orbitals; ++j) {
//             auto G1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1, i)));
//             auto G2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1, j)));

//             if(V(i, j) != 0.) { Oout += V(i, j) * (n(i) * n(j)).template cast<Scalar_>(); }
//             if(Vz(i, j) != 0.) { Oout += Vz(i, j) * (Tz(i) * Tz(j)).template cast<Scalar_>(); }
//             if(Vxy(i, j) != 0.) {
//                 Oout +=
//                     0.5 * Vxy(i, j) *
//                     (OperatorType::prod(Tp(i, G1), Tm(j, G2), Symmetry::qvacuum()) + OperatorType::prod(Tm(i, G1), Tp(j, G2), Symmetry::qvacuum()))
//                         .template cast<Scalar_>();
//             }
//             if(Jz(i, j) != 0.) { Oout += Jz(i, j) * Sz(i) * Sz(j).template cast<Scalar_>(); }
//             if(Jxy(i, j) != 0.) { Oout += 0.5 * Jxy(i, j) * (Sp(i) * Sm(j) + Sm(i) * Sp(j)).template cast<Scalar_>(); }
//         }

//     for(int i = 0; i < N_orbitals; ++i) {
//         if(U(i) != 0. and U(i) != numeric_limits<double>::infinity()) { Oout += U(i) * d(i).template cast<Scalar_>(); }
//         if(Eorb(i) != 0.) { Oout += Eorb(i) * n(i).template cast<Scalar_>(); }
//         if(Bz(i) != 0.) { Oout -= Bz(i) * Sz(i).template cast<Scalar_>(); }
//         if(C(i) != 0.) {
//             // convention: cUP*cDN + cdagDN*cdagUP
//             // convention for cc, cdagcdag is opposite, therefore needs one commutation
//             //			Oout -= C(i) * cc(i).template cast<Scalar_>();
//             //			Oout -= C(i) * cdagcdag(i).template cast<Scalar_>();
//             Oout += C(i) * (c(UP, i) * c(DN, i)).template cast<Scalar_>();
//             Oout += C(i) * (cdag(DN, i) * cdag(UP, i)).template cast<Scalar_>();
//         }
//     }
//     Oout.label() = "Hloc";
//     return Oout;
// }

// template <typename Symmetry>
// template <typename Scalar_, typename Dummy>
// typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(), SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>>>::type
// FermionBase<Symmetry>::HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& U,
//                                            const Array<Scalar_, Dynamic, 1>& Uph,
//                                            const Array<Scalar_, Dynamic, 1>& Eorb,
//                                            const Array<Scalar_, Dynamic, Dynamic>& t,
//                                            const Array<Scalar_, Dynamic, Dynamic>& V,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Vz,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Vxy,
//                                            const Array<Scalar_, Dynamic, Dynamic>& J) const
// {
//     auto Oout = HubbardHamiltonian<Scalar_>(Uph, t, ZeroHopping(), J);

//     for(int i = 0; i < N_orbitals; ++i)
//         for(int j = 0; j < N_orbitals; ++j) {
//             auto G1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1, i)));
//             auto G2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1, j)));
//             if(V(i, j) != 0.) { Oout += V(i, j) * (n(i) * n(j)).template cast<Scalar_>(); }
//             if(Vz(i, j) != 0.) { Oout += Vz(i, j) * (Tz(i) * Tz(j)).template cast<Scalar_>(); }
//             if(Vxy(i, j) != 0.) {
//                 Oout +=
//                     0.5 * Vxy(i, j) *
//                     (OperatorType::prod(Tp(i, G1), Tm(j, G2), Symmetry::qvacuum()) + OperatorType::prod(Tm(i, G1), Tp(j, G2), Symmetry::qvacuum()))
//                         .template cast<Scalar_>();
//             }
//         }

//     for(int i = 0; i < N_orbitals; ++i) {
//         if(U(i) != 0. and U(i) != numeric_limits<double>::infinity()) { Oout += U(i) * d(i).template cast<Scalar_>(); }
//         if(Eorb(i) != 0.) { Oout += Eorb(i) * n(i).template cast<Scalar_>(); }
//     }
//     Oout.label() = "Hloc";
//     return Oout;
// }

// template <typename Symmetry>
// template <typename Scalar_, typename Dummy>
// typename std::enable_if<Dummy::NO_SPIN_SYM() and Dummy::IS_CHARGE_SU2(), SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>>>::type
// FermionBase<Symmetry>::HubbardHamiltonian(const Array<Scalar_, Dynamic, 1>& Uph,
//                                            const Array<Scalar_, Dynamic, Dynamic>& t,
//                                            const Array<Scalar_, Dynamic, Dynamic>& V,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Jz,
//                                            const Array<Scalar_, Dynamic, Dynamic>& Jxy,
//                                            const Array<Scalar_, Dynamic, 1>& Bz,
//                                            const Array<Scalar_, Dynamic, 1>& Bx) const
// {
//     auto Oout = HubbardHamiltonian<Scalar_>(Uph, t, V, ZeroHopping());

//     for(int i = 0; i < N_orbitals; ++i)
//         for(int j = 0; j < N_orbitals; ++j) {
//             if(Jz(i, j) != 0.) { Oout += Jz(i, j) * (Sz(i) * Sz(j)).template cast<Scalar_>(); }
//             if(Jxy(i, j) != 0.) {
//                 Oout += 0.5 * Jxy(i, j) *
//                         (OperatorType::prod(Sp(i), Sm(j), Symmetry::qvacuum()) + OperatorType::prod(Sm(i), Sp(j), Symmetry::qvacuum()))
//                             .template cast<Scalar_>();
//             }
//         }

//     for(int i = 0; i < N_orbitals; ++i) {
//         if(Bz(i) != 0.) { Oout += -1. * Bz(i) * Sz(i).template cast<Scalar_>(); }
//         if(Bx(i) != 0.) { Oout += -1. * Bx(i) * Sx(i).template cast<Scalar_>(); }
//     }
//     Oout.label() = "Hloc";
//     return Oout;
// }

// template <typename Symmetry>
// template <typename Scalar_, typename Dummy>
// typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>>>::type
// FermionBase<Symmetry>::coupling_Bx(const Array<double, Dynamic, 1>& Bx) const
// {
//     SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>> Mout(Symmetry::qvacuum(), TensorBasis);
//     for(int i = 0; i < N_orbitals; ++i) {
//         if(Bx(i) != 0.) { Mout -= Bx(i) * Sx(i).template cast<Scalar_>(); }
//     }
//     return Mout;
// }

// template <typename Symmetry>
// template <typename Dummy>
// typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry, Eigen::Matrix<complex<double>, -1, -1>>>::type
// FermionBase<Symmetry>::coupling_By(const Array<double, Dynamic, 1>& By) const
// {
//     SiteOperatorQ<Symmetry, Eigen::Matrix<complex<double>, -1, -1>> Mout(Symmetry::qvacuum(), TensorBasis);
//     for(int i = 0; i < N_orbitals; ++i) {
//         if(By(i) != 0.) { Mout -= -1i * By(i) * iSy(i).template cast<complex<double>>(); }
//     }
//     return Mout;
// }

// template <typename Symmetry>
// template <typename Scalar_, typename Dummy>
// typename std::enable_if<Dummy::IS_TRIVIAL, SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>>>::type
// FermionBase<Symmetry>::coupling_singleFermion(const Array<double, Dynamic, 1>& Fp) const
// {
//     SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>> Mout(Symmetry::qvacuum(), TensorBasis);
//     for(int i = 0; i < N_orbitals; ++i) {
//         if(Fp(i) != 0.) { Mout += Fp(i) * (c(UP, i) + cdag(UP, i) + c(DN, i) + cdag(DN, i)).template cast<Scalar_>(); }
//     }
//     return Mout;
// }

// template <typename Symmetry>
// template <typename Scalar_, typename Dummy>
// typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>>>::type
// FermionBase<Symmetry>::coupling_XYZspin(const Array<double, Dynamic, Dynamic>& Jx,
//                                          const Array<double, Dynamic, Dynamic>& Jy,
//                                          const Array<double, Dynamic, Dynamic>& Jz) const
// {
//     SiteOperatorQ<Symmetry, Eigen::Matrix<Scalar_, -1, -1>> Mout(Symmetry::qvacuum(), TensorBasis);
//     for(int i = 0; i < N_orbitals; ++i)
//         for(int j = 0; j < N_orbitals; ++j) {
//             if(Jx(i, j) != 0.) { Mout += Jx(i, j) * OperatorType::prod(Sx(i), Sx(j), Symmetry::qvacuum()).template cast<Scalar_>(); }
//             if(Jy(i, j) != 0.) { Mout += -Jy(i, j) * OperatorType::prod(iSy(i), iSy(j), Symmetry::qvacuum()).template cast<Scalar_>(); }
//             if(Jz(i, j) != 0.) { Mout += Jz(i, j) * OperatorType::prod(Sz(i), Sz(j), Symmetry::qvacuum()).template cast<Scalar_>(); }
//         }
//     return Mout;
// }

} // namespace Xped
#endif
