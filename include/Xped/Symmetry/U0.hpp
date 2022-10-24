#ifndef U0_H_
#define U0_H_

/// \cond
#include <cstddef>
/// \endcond

#include "Xped/Util/Mpi.hpp"

#include "Xped/Symmetry/SymBase.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"
#include "Xped/Symmetry/qarray.hpp"

namespace Xped::Sym {

template <typename Scalar_>
struct U0;

template <typename Scalar__>
struct SymTraits<U0<Scalar__>>
{
    constexpr static int Nq = 0;
    typedef qarray<Nq> qType;
    typedef Scalar__ Scalar;
};

/** \class U0
 * \ingroup Symmetry
 *
 * Dummy class for no symmetry.
 *
 */
template <typename Scalar_ = double>
struct U0 : SymBase<U0<Scalar_>>
{
    typedef Scalar_ Scalar;
    typedef qarray<0> qType;

    U0(){};

    static std::string name() { return "noSymmetry"; }

    static constexpr std::size_t Nq = 0;

    static constexpr std::array<bool, Nq> HAS_MULTIPLICITIES = {};
    static constexpr std::array<bool, Nq> NON_ABELIAN = {};
    static constexpr std::array<bool, Nq> ABELIAN = {};
    static constexpr std::array<bool, Nq> IS_TRIVIAL = {};
    static constexpr std::array<bool, Nq> IS_MODULAR = {};
    static constexpr std::array<bool, Nq> IS_FERMIONIC = {};
    static constexpr std::array<bool, Nq> IS_BOSONIC = {};
    static constexpr std::array<bool, Nq> IS_SPIN = {};
    static constexpr std::array<int, Nq> MOD_N = {};

    static constexpr bool ANY_HAS_MULTIPLICITIES = false;
    static constexpr bool ANY_NON_ABELIAN = false;
    static constexpr bool ANY_ABELIAN = false;
    static constexpr bool ANY_IS_TRIVIAL = true;
    static constexpr bool ANY_IS_MODULAR = false;
    static constexpr bool ANY_IS_FERMIONIC = false;
    static constexpr bool ANY_IS_BOSONIC = false;
    static constexpr bool ANY_IS_SPIN = false;

    static constexpr bool ALL_HAS_MULTIPLICITIES = false;
    static constexpr bool ALL_NON_ABELIAN = false;
    static constexpr bool ALL_ABELIAN = false;
    static constexpr bool ALL_IS_TRIVIAL = true;
    static constexpr bool ALL_IS_MODULAR = false;
    static constexpr bool ALL_IS_FERMIONIC = false;
    static constexpr bool ALL_IS_BOSONIC = false;
    static constexpr bool ALL_IS_SPIN = false;

    static constexpr bool IS_CHARGE_SU2() { return false; }
    static constexpr bool IS_SPIN_SU2() { return false; }

    static constexpr bool IS_SPIN_U1() { return false; }

    static constexpr bool NO_SPIN_SYM() { return true; }
    static constexpr bool NO_CHARGE_SYM() { return true; }

    inline static constexpr std::array<KIND, Nq> kind() { return {}; }

    inline static constexpr qType qvacuum() { return {}; }
    inline static constexpr std::array<qType, 1> lowest_qs() { return std::array<qType, 1>{{qarray<0>(std::array<int, 0>{{}})}}; }

    inline static qType conj(const qType&) { return {}; }
    inline static int degeneracy(const qType&) { return Scalar(1); }

    inline static qType random_q() { return {}; }

    inline static std::vector<qType> basis_combine(const qType&, const qType&) { return {{}}; }

    static std::size_t multiplicity(const qType&, const qType&, const qType&) { return 1ul; }

    inline static Scalar coeff_dot(const qType&) { return Scalar(1.); }

    inline static Scalar coeff_twist(const qType&) { return Scalar(1.); }

    inline static Scalar coeff_FS(const qType&) { return Scalar(1.); }

    template <typename PlainLib>
    inline static typename PlainLib::template TType<Scalar, 2> one_j_tensor(const qType&, const mpi::XpedWorld& world = mpi::getUniverse())
    {
        typedef typename PlainLib::Indextype IndexType;
        auto T = PlainLib::template construct<Scalar>(std::array<IndexType, 2>{1, 1}, world);
        std::array<IndexType, 2> index = {0, 0};
        PlainLib::template setVal<Scalar, 2>(T, index, Scalar(1.));
        return T;
    }

    static Scalar coeff_rightOrtho(const qType&, const qType&) { return Scalar(1.); }

    inline static Scalar coeff_3j(const qType&, const qType&, const qType&, int, int, int) { return Scalar(1.); }

    template <typename PlainLib>
    inline static typename PlainLib::template TType<Scalar, 3>
    CGC(const qType&, const qType&, const qType&, const std::size_t, const mpi::XpedWorld& world = mpi::getUniverse())
    {
        typedef typename PlainLib::Indextype IndexType;
        auto T = PlainLib::template construct<Scalar>(std::array<IndexType, 3>{1, 1, 1}, world);
        std::array<IndexType, 3> index = {0, 0, 0};
        PlainLib::template setVal<Scalar, 3>(T, index, Scalar(1.));
        return T;
    }

    inline static Scalar coeff_6j(const qType&, const qType&, const qType&, const qType&, const qType&, const qType&) { return Scalar(1.); }

    static Scalar coeff_recouple(const qType& q1, const qType& q2, const qType& q3, const qType& Q, const qType& Q12, const qType& Q23)
    {
        return coeff_6j(q1, q2, Q12, q3, Q, Q23);
    }

    inline static Scalar
    coeff_9j(const qType&, const qType&, const qType&, const qType&, const qType&, const qType&, const qType&, const qType&, const qType&)
    {
        return Scalar(1.);
    }

    static Scalar coeff_swap(const qType& ql, const qType& qr, const qType& qf) { return triangle(ql, qr, qf) ? Scalar(1.) : Scalar(0.); };

    inline static bool triangle(const qType& q1, const qType& q2, const qType& q3) { return true; }
};

} // namespace Xped::Sym
#endif
