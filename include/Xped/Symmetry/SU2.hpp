#ifndef SU2_H_
#define SU2_H_

/// \cond
#include <array>
#include <cstddef>
#include <vector>
/// \endcond

#include "Xped/Symmetry/SymBase.hpp"
#include "Xped/Symmetry/functions.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"
#include "Xped/Symmetry/qarray.hpp"
#include "Xped/Util/Mpi.hpp"

namespace Xped::Sym {

template <typename Kind, typename Scalar_>
struct SU2;

template <typename Kind_, typename Scalar__>
struct SymTraits<SU2<Kind_, Scalar__>>
{
    constexpr static int Nq = 1;
    typedef qarray<Nq> qType;
    typedef Scalar__ Scalar;
};

/**
 * \class SU2
 * \ingroup Symmetry
 *
 * Class for handling a SU(2) symmetry of a Hamiltonian without explicitly store the Clebsch-Gordon coefficients but with computing
 * \f$(3n)j\f$-symbols.
 *
 * \describe_Scalar
 * \note An implementation for the basic \f$(3n)j\f$ symbols is used from SU2Wrappers.h.
 *       Currently, the gsl-implementation and the wig3j/fastwig3j library can be used, but other libraries which calculates the symbols can be
 * included. Just add a wrapper in SU2Wrappers.h.
 */
template <typename Kind, typename Scalar_ = double>
struct SU2 : public SymBase<SU2<Kind, Scalar_>>
{
    typedef Scalar_ Scalar;
    typedef SymBase<SU2<Kind, Scalar>> Base;

    static constexpr std::size_t Nq = 1;

    static constexpr std::array<bool, Nq> HAS_MULTIPLICITIES = {false};
    static constexpr std::array<bool, Nq> NON_ABELIAN = {true};
    static constexpr std::array<bool, Nq> ABELIAN = {false};
    static constexpr std::array<bool, Nq> IS_TRIVIAL = {false};
    static constexpr std::array<bool, Nq> IS_MODULAR = {false};
    static constexpr std::array<bool, Nq> IS_FERMIONIC = {(Kind::name == KIND::FT)};
    static constexpr std::array<bool, Nq> IS_BOSONIC = {(Kind::name == KIND::T)};
    static constexpr std::array<bool, Nq> IS_SPIN = {(Kind::name == KIND::S)};
    static constexpr std::array<int, Nq> MOD_N = {1};

    static constexpr bool ANY_HAS_MULTIPLICITIES = HAS_MULTIPLICITIES[0];
    static constexpr bool ANY_NON_ABELIAN = NON_ABELIAN[0];
    static constexpr bool ANY_ABELIAN = ABELIAN[0];
    static constexpr bool ANY_IS_TRIVIAL = IS_TRIVIAL[0];
    static constexpr bool ANY_IS_MODULAR = IS_MODULAR[0];
    static constexpr bool ANY_IS_FERMIONIC = IS_FERMIONIC[0];
    static constexpr bool ANY_IS_BOSONIC = IS_BOSONIC[0];
    static constexpr bool ANY_IS_SPIN = IS_SPIN[0];

    static constexpr bool ALL_HAS_MULTIPLICITIES = HAS_MULTIPLICITIES[0];
    static constexpr bool ALL_NON_ABELIAN = NON_ABELIAN[0];
    static constexpr bool ALL_ABELIAN = ABELIAN[0];
    static constexpr bool ALL_IS_TRIVIAL = IS_TRIVIAL[0];
    static constexpr bool ALL_IS_MODULAR = IS_MODULAR[0];
    static constexpr bool ALL_IS_FERMIONIC = IS_FERMIONIC[0];
    static constexpr bool ALL_IS_BOSONIC = IS_BOSONIC[0];
    static constexpr bool ALL_IS_SPIN = IS_SPIN[0];

    static constexpr bool IS_CHARGE_SU2()
    {
        if constexpr(IS_FERMIONIC[0] or IS_BOSONIC[0]) { return true; }
        return false;
    }
    static constexpr bool IS_SPIN_SU2()
    {
        if constexpr(IS_SPIN[0]) { return true; }
        return false;
    }

    static constexpr bool IS_SPIN_U1() { return false; }

    static constexpr bool NO_SPIN_SYM()
    {
        if constexpr(not IS_SPIN[0]) { return true; }
        return false;
    }
    static constexpr bool NO_CHARGE_SYM()
    {
        if constexpr(not IS_FERMIONIC[0] and not IS_BOSONIC[0]) { return true; }
        return false;
    }

    typedef qarray<Nq> qType;

    SU2(){};

    inline static std::string name() { return IS_FERMIONIC[0] ? "ƑSU₂" : "SU₂"; }
    inline static constexpr std::array<KIND, Nq> kind() { return {Kind::name}; }

    inline static constexpr qType qvacuum() { return {1}; }
    inline static constexpr std::array<qType, 1> lowest_qs() { return std::array<qType, 1>{{qarray<1>(std::array<int, 1>{{2}})}}; }

    inline static qType conj(const qType& q) { return q; }
    inline static int degeneracy(const qType& q) { return q[0]; }

    static qType random_q();

    /**
     * Calculate the irreps of the tensor product of \p ql and \p qr.
     */
    static std::vector<qType> basis_combine(const qType& ql, const qType& qr);

    static std::size_t multiplicity(const qType& q1, const qType& q2, const qType& q3) { return triangle(q1, q2, q3) ? 1ul : 0ul; }
    ///@{
    /**
     * Various coeffecients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
     */
    static Scalar coeff_dot(const qType& q1);

    inline static Scalar coeff_twist(const qType& q)
    {
        if constexpr(not IS_FERMIONIC[0]) { return 1.; }
        return ((q[0] - 1) % 2 != 0) ? -1. : 1.;
    }

    static Scalar coeff_FS(const qType& q1);

    template <typename PlainLib>
    static typename PlainLib::template TType<Scalar_, 2> one_j_tensor(const qType& q1, const mpi::XpedWorld& world = mpi::getUniverse());

    static Scalar coeff_rightOrtho(const qType& q1, const qType& q2) { return static_cast<Scalar>(q1[0]) / static_cast<Scalar>(q2[0]); }

    static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3, int q1_z, int q2_z, int q3_z);

    static Scalar coeff_turn(const qType& ql, const qType& qr, const qType& qf)
    {
        return triangle(ql, qr, qf) ? coeff_swap(ql, qr, qf) * std::sqrt(static_cast<Scalar>(qf[0]) / static_cast<Scalar>(ql[0])) : Scalar(0.);
    }

    template <typename PlainLib>
    static typename PlainLib::template TType<Scalar_, 3>
    CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t, const mpi::XpedWorld& world = mpi::getUniverse());

    static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3, const qType& q4, const qType& q5, const qType& q6);

    static Scalar coeff_9j(const qType& q1,
                           const qType& q2,
                           const qType& q3,
                           const qType& q4,
                           const qType& q5,
                           const qType& q6,
                           const qType& q7,
                           const qType& q8,
                           const qType& q9);

    static Scalar coeff_recouple(const qType& q1, const qType& q2, const qType& q3, const qType& Q, const qType& Q12, const qType& Q23)
    {
        return std::sqrt(Q12[0] * Q23[0]) * phase<Scalar>((q1[0] + q2[0] + q3[0] + Q[0]) / 2.) * coeff_6j(q1, q2, Q12, q3, Q, Q23);
    }

    static Scalar coeff_swap(const qType& ql, const qType& qr, const qType& qf)
    {
        Scalar sign = +1.;
        if constexpr(IS_FERMIONIC[0]) {
            bool parity = ((ql[0] - 1) % 2 != 0) and ((qr[0] - 1) % 2 != 0);
            sign = parity ? -1. : 1.;
        }
        return triangle(ql, qr, qf) ? sign * phase<Scalar>((ql[0] + qr[0] - qf[0] - 1) / 2) : Scalar(0.);
    }
    ///@}

    static bool triangle(const qType& q1, const qType& q2, const qType& q3);
};

} // namespace Xped::Sym

#ifndef XPED_COMPILED_LIB
#    include "Symmetry/SU2.cpp"
#endif

#endif
