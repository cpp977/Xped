#ifndef XPED_ZN_H_
#define XPED_ZN_H_

#include <array>

#include "Xped/Util/Constfct.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Util/Random.hpp"

#include "Xped/Symmetry/SymBase.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

namespace Xped::Sym {

template <typename Kind, std::size_t N, typename Scalar_>
struct ZN;

template <typename Kind_, std::size_t N_, typename Scalar__>
struct SymTraits<ZN<Kind_, N_, Scalar__>>
{
    constexpr static int Nq = 1;
    typedef qarray<Nq> qType;
    typedef Scalar__ Scalar;
};

/** \class ZN
 * \ingroup Symmetry
 *
 * Class for handling a Z(N) symmetry of a Hamiltonian.
 *
 * \describe_Scalar
 */
template <typename Kind, std::size_t N, typename Scalar_ = double>
struct ZN : public SymBase<ZN<Kind, N, Scalar_>>
{
    typedef Scalar_ Scalar;
    static constexpr size_t Nq = 1;

    typedef qarray<Nq> qType;

    static constexpr std::array<bool, Nq> HAS_MULTIPLICITIES = {false};
    static constexpr std::array<bool, Nq> NON_ABELIAN = {false};
    static constexpr std::array<bool, Nq> ABELIAN = {true};
    static constexpr std::array<bool, Nq> IS_TRIVIAL = {false};
    static constexpr std::array<bool, Nq> IS_MODULAR = {true};
    static constexpr std::array<bool, Nq> IS_FERMIONIC = {(Kind::name == KIND::FN)};
    static constexpr std::array<bool, Nq> IS_BOSONIC = {(Kind::name == KIND::N)};
    static constexpr std::array<bool, Nq> IS_SPIN = {(Kind::name == KIND::M)};
    static constexpr std::array<int, Nq> MOD_N = {N};

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

    static constexpr bool IS_CHARGE_SU2() { return false; }
    static constexpr bool IS_SPIN_SU2() { return false; }

    static constexpr bool IS_SPIN_U1()
    {
        if constexpr(IS_SPIN[0]) { return true; }
        return false;
    }

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

    ZN(){};

    inline static constexpr qType qvacuum() { return {0}; }
    inline static constexpr auto lowest_qs()
    {
        if constexpr(N == 2) {
            return std::array<qType, 1>{{qarray<1>(std::array<int, 1>{{1}})}};
        } else {
            return std::array<qType, 2>{{qarray<1>(std::array<int, 1>{{1}}), qarray<1>(std::array<int, 1>{{N - 1}})}};
        }
    }

    inline static std::string name()
    {
        std::string prefix = IS_FERMIONIC[0] ? "Ƒ" : "";
        if(N == 2) { return prefix + "Z₂"; }
        if(N == 3) { return prefix + "Z₃"; }
        if(N == 4) { return prefix + "Z₄"; }
        if(N == 5) { return prefix + "Z₅"; }
        if(N == 36) { return prefix + "Z₃₆"; }
        if(N == 64) { return prefix + "Z₆₄"; }
        return prefix + "Z(" + std::to_string(N) + ")";
    }

    inline static constexpr std::array<KIND, Nq> kind() { return {Kind::name}; }

    inline static qType conj(const qType& q) { return {util::constFct::posmod<N>(-q[0])}; }
    inline static int degeneracy(const qType&) { return 1; }

    inline static qType random_q()
    {
        int qval = random::threadSafeRandUniform<int, int>(0, N - 1);
        qType out = {qval};
        return out;
    }

    /**
     * Calculate the irreps of the tensor product of \p ql and \p qr.
     */
    static std::vector<qType> basis_combine(const qType& ql, const qType& qr);

    static std::size_t multiplicity(const qType& q1, const qType& q2, const qType& q3) { return triangle(q1, q2, q3) ? 1ul : 0ul; }

    ///@{
    /**
     * Various coeffecients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
     * \note All coefficients are trivial for U(1) and could be represented by a bunch of Kronecker deltas.
     *       Here we return simply 1, because the algorithm only allows valid combinations of quantumnumbers,
     *       for which the Kronecker deltas are not necessary.
     */
    inline static Scalar coeff_dot(const qType&) { return Scalar(1.); }

    inline static Scalar coeff_twist(const qType& q)
    {
        if constexpr(not IS_FERMIONIC[0]) { return 1.; }
        return (q[0] % 2 != 0) ? -1. : 1.;
    }

    static Scalar coeff_FS(const qType&) { return Scalar(1.); }

    template <typename PlainLib>
    static typename PlainLib::template TType<Scalar_, 2> one_j_tensor(const qType&, const mpi::XpedWorld& world = mpi::getUniverse())
    {
        typedef typename PlainLib::Indextype IndexType;
        auto T = PlainLib::template construct<Scalar>(std::array<IndexType, 2>{1, 1}, world);
        std::array<IndexType, 2> index = {0, 0};
        PlainLib::template setVal<Scalar, 2>(T, index, Scalar(1.));
        return T;
    }

    static Scalar coeff_rightOrtho(const qType&, const qType&) { return Scalar(1.); }

    inline static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3, int, int, int)
    {
        return triangle(q1, q2, conj(q3)) ? Scalar(1.) : Scalar(0.);
    }

    template <typename PlainLib>
    static typename PlainLib::template TType<Scalar_, 3>
    CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t, const mpi::XpedWorld& world = mpi::getUniverse());

    static Scalar coeff_turn(const qType& ql, const qType& qr, const qType& qf) { return triangle(ql, qr, qf) ? Scalar(1.) : Scalar(0.); }

    inline static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3, const qType& q4, const qType& q5, const qType& q6);

    inline static Scalar coeff_recouple(const qType& q1, const qType& q2, const qType& q3, const qType& Q, const qType& Q12, const qType& Q23)
    {
        return coeff_6j(q1, q2, Q12, q3, Q, Q23);
    }

    inline static Scalar
    coeff_9j(const qType&, const qType&, const qType&, const qType&, const qType&, const qType&, const qType&, const qType&, const qType&)
    {
        return Scalar(1);
    }

    static Scalar coeff_swap(const qType& ql, const qType& qr, const qType& qf)
    {
        Scalar sign = +1.;
        if constexpr(IS_FERMIONIC[0]) {
            bool parity = (ql[0] % 2 != 0) and (qr[0] % 2 != 0);
            sign = parity ? -1. : 1.;
        }
        return triangle(ql, qr, qf) ? sign * Scalar(1.) : Scalar(0.);
    };
    ///@}

    static bool triangle(const qType& q1, const qType& q2, const qType& q3);
};

template <typename Kind, std::size_t N, typename Scalar_>
std::vector<typename ZN<Kind, N, Scalar_>::qType> ZN<Kind, N, Scalar_>::basis_combine(const qType& ql, const qType& qr)
{
    std::vector<qType> vout;
    vout.push_back({util::constFct::posmod<N>({ql[0] + qr[0]})});
    return vout;
}

template <typename Kind, std::size_t N, typename Scalar_>
template <typename PlainLib>
typename PlainLib::template TType<Scalar_, 3>
ZN<Kind, N, Scalar_>::CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t, const mpi::XpedWorld& world)
{
    typedef typename PlainLib::Indextype IndexType;
    auto T = PlainLib::template construct<Scalar>(std::array<IndexType, 3>{1, 1, 1}, world);
    if(triangle(q1, q2, q3)) {
        std::array<IndexType, 3> index = {0, 0, 0};
        PlainLib::template setVal<Scalar, 3>(T, index, Scalar(1.));
    } else {
        std::array<IndexType, 3> index = {0, 0, 0};
        PlainLib::template setVal<Scalar, 3>(T, index, Scalar(0.));
    }
    return T;
}

template <typename Kind, std::size_t N, typename Scalar_>
Scalar_ ZN<Kind, N, Scalar_>::coeff_6j(const qType& q1, const qType& q2, const qType& q3, const qType& q4, const qType& q5, const qType& q6)
{
    if(triangle(q1, q2, q3) and triangle(q1, q6, q5) and triangle(q2, q4, q6) and triangle(q3, q4, q5)) { return Scalar(1.); }
    return Scalar(0.);
}

template <typename Kind, std::size_t N, typename Scalar_>
bool ZN<Kind, N, Scalar_>::triangle(const qType& q1, const qType& q2, const qType& q3)
{
    // check the triangle rule for ZN quantum numbers
    if(util::constFct::posmod<N>(q1[0] + q2[0]) == q3[0]) { return true; }
    return false;
}

} // namespace Xped::Sym

#endif
