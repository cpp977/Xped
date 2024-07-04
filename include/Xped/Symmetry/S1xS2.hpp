#ifndef XPED_S1xS2_H_
#define XPED_S1xS2_H_

/// \cond
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
/// \endcond

#include "Xped/Util/JoinArray.hpp"

#include "Xped/Symmetry/qarray.hpp"
#include "Xped/Symmetry/SymBase.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

namespace Xped::Sym {

template <typename S1, typename S2>
struct S1xS2;

template <typename S1, typename S2>
struct SymTraits<S1xS2<S1, S2>>
{
    constexpr static int Nq = S1::Nq + S2::Nq;
    typedef qarray<Nq> qType;
    typedef typename S1::Scalar Scalar;
};

/**
 * \class S1xS2
 * \ingroup Symmetry
 *
 * This class combines two symmetries.
 *
 */
template <typename S1_, typename S2_>
struct S1xS2 : public SymBase<S1xS2<S1_, S2_>>
{
public:
    typedef typename S1_::Scalar Scalar;

    typedef S1_ S1;
    typedef S2_ S2;

    S1xS2(){};

    static std::string name() { return S1_::name() + "×" + S2_::name(); }

    static constexpr std::size_t Nq = S1_::Nq + S2_::Nq;

    static constexpr std::array<bool, Nq> HAS_MULTIPLICITIES = util::join(S1::HAS_MULTIPLICITIES, S2::HAS_MULTIPLICITIES);
    static constexpr std::array<bool, Nq> NON_ABELIAN = util::join(S1::NON_ABELIAN, S2::NON_ABELIAN);
    static constexpr std::array<bool, Nq> ABELIAN = util::join(S1::ABELIAN, S2::ABELIAN);
    static constexpr std::array<bool, Nq> IS_TRIVIAL = util::join(S1::IS_TRIVIAL, S2::IS_TRIVIAL);
    static constexpr std::array<bool, Nq> IS_MODULAR = util::join(S1::IS_MODULAR, S2::IS_MODULAR);
    static constexpr std::array<bool, Nq> IS_FERMIONIC = util::join(S1::IS_FERMIONIC, S2::IS_FERMIONIC);
    static constexpr std::array<bool, Nq> IS_BOSONIC = util::join(S1::IS_BOSONIC, S2::IS_BOSONIC);
    static constexpr std::array<bool, Nq> IS_SPIN = util::join(S1::IS_SPIN, S2::IS_SPIN);
    static constexpr std::array<int, Nq> MOD_N = util::join(S1::MOD_N, S2::MOD_N);

    static constexpr bool ANY_HAS_MULTIPLICITIES = S1::ANY_HAS_MULTIPLICITIES or S2::ANY_HAS_MULTIPLICITIES;
    static constexpr bool ANY_NON_ABELIAN = S1::ANY_NON_ABELIAN or S2::ANY_NON_ABELIAN;
    static constexpr bool ANY_ABELIAN = S1::ANY_ABELIAN or S2::ANY_ABELIAN;
    static constexpr bool ANY_IS_TRIVIAL = S1::ANY_IS_TRIVIAL or S2::ANY_IS_TRIVIAL;
    static constexpr bool ANY_IS_MODULAR = S1::ANY_IS_MODULAR or S2::ANY_IS_MODULAR;
    static constexpr bool ANY_IS_FERMIONIC = S1::ANY_IS_FERMIONIC or S2::ANY_IS_FERMIONIC;
    static constexpr bool ANY_IS_BOSONIC = S1::ANY_IS_BOSONIC or S2::ANY_IS_BOSONIC;
    static constexpr bool ANY_IS_SPIN = S1::ANY_IS_SPIN or S2::ANY_IS_SPIN;

    static constexpr bool ALL_HAS_MULTIPLICITIES = S1::ALL_HAS_MULTIPLICITIES and S2::ALL_HAS_MULTIPLICITIES;
    static constexpr bool ALL_NON_ABELIAN = S1::ALL_NON_ABELIAN and S2::ALL_NON_ABELIAN;
    static constexpr bool ALL_ABELIAN = S1::ALL_ABELIAN and S2::ALL_ABELIAN;
    static constexpr bool ALL_IS_TRIVIAL = S1::ALL_IS_TRIVIAL and S2::ALL_IS_TRIVIAL;
    static constexpr bool ALL_IS_MODULAR = S1::ALL_IS_MODULAR and S2::ALL_IS_MODULAR;
    static constexpr bool ALL_IS_FERMIONIC = S1::ALL_IS_FERMIONIC and S2::ALL_IS_FERMIONIC;
    static constexpr bool ALL_IS_BOSONIC = S1::ALL_IS_BOSONIC and S2::ALL_IS_BOSONIC;
    static constexpr bool ALL_IS_SPIN = S1::ALL_IS_SPIN and S2::ALL_IS_SPIN;

    static constexpr bool IS_CHARGE_SU2() { return S1_::IS_CHARGE_SU2() or S2_::IS_CHARGE_SU2(); }
    static constexpr bool IS_SPIN_SU2() { return S1_::IS_SPIN_SU2() or S2_::IS_SPIN_SU2(); }

    static constexpr bool IS_SPIN_U1() { return S1_::IS_SPIN_U1() or S2_::IS_SPIN_U1(); }

    static constexpr bool NO_SPIN_SYM() { return S1_::NO_SPIN_SYM() and S2_::NO_SPIN_SYM(); }
    static constexpr bool NO_CHARGE_SYM() { return S1_::NO_CHARGE_SYM() and S2_::NO_CHARGE_SYM(); }

    typedef qarray<Nq> qType;

    inline static constexpr std::array<KIND, Nq> kind() { return util::join(S1_::kind(), S2_::kind()); }

    inline static constexpr qType qvacuum() { return join(S1_::qvacuum(), S2_::qvacuum()); }
    inline static constexpr std::array<qType, S1::lowest_qs().size() * S2::lowest_qs().size()> lowest_qs()
    {
        std::array<qType, S1::lowest_qs().size() * S2::lowest_qs().size()> out;
        size_t index = 0;
        for(const auto& q1 : S1::lowest_qs())
            for(const auto& q2 : S2::lowest_qs()) {
                out[index] = join(q1, q2);
                index++;
            }
        return out;
    }

    inline static qType conj(const qType& q)
    {
        auto [ql, qr] = disjoin<S1_::Nq, S2_::Nq>(q);
        return join(S1_::conj(ql), S2_::conj(qr));
    }

    inline static int degeneracy(const qType& q)
    {
        auto [ql, qr] = disjoin<S1_::Nq, S2_::Nq>(q);
        return S1_::degeneracy(ql) * S2_::degeneracy(qr);
    }

    inline static qType random_q() { return join(S1_::random_q(), S2_::random_q()); }

    ///@{
    /**
     * Calculate the irreps of the tensor product of \p ql and \p qr.
     */
    static std::vector<qType> basis_combine(const qType& ql, const qType& qr);
    ///@}

    static std::size_t multiplicity(const qType& q1, const qType& q2, const qType& q3)
    {
        auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
        auto [q2l, q2r] = disjoin<S1_::Nq, S2_::Nq>(q2);
        auto [q3l, q3r] = disjoin<S1_::Nq, S2_::Nq>(q3);
        return S1::multiplicity(q1l, q2l, q3l) * S2::multiplicity(q1r, q2r, q3r);
    }

    ///@{
    /**
     * Various coefficients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
     */
    inline static Scalar coeff_dot(const qType& q1);

    inline static Scalar coeff_twist(const qType& q)
    {
        auto [ql, qr] = disjoin<S1::Nq, S2::Nq>(q);
        return S1::coeff_twist(ql) * S2::coeff_twist(qr);
    }

    static Scalar coeff_FS(const qType& q)
    {
        auto [ql, qr] = disjoin<S1_::Nq, S2_::Nq>(q);
        return S1::coeff_FS(ql) * S2::coeff_FS(qr);
    }

    template <typename PlainLib>
    static typename PlainLib::template TType<Scalar, 2> one_j_tensor(const qType& q, mpi::XpedWorld& world = mpi::getUniverse())
    {
        auto [ql, qr] = disjoin<S1_::Nq, S2_::Nq>(q);

        auto Tl = S1::template one_j_tensor<PlainLib>(ql, world);
        auto Tr = S2::template one_j_tensor<PlainLib>(qr, world);
        return PlainLib::tensorProd(Tl, Tr);
    }

    static Scalar coeff_rightOrtho(const qType&, const qType&);

    static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3, int q1_z, int q2_z, int q3_z) { return 1.; }

    template <typename PlainLib>
    static typename PlainLib::template TType<Scalar, 3>
    CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t alpha, mpi::XpedWorld& world = mpi::getUniverse())
    {
        auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
        auto [q2l, q2r] = disjoin<S1_::Nq, S2_::Nq>(q2);
        auto [q3l, q3r] = disjoin<S1_::Nq, S2_::Nq>(q3);

        auto Tl = S1::template CGC<PlainLib>(q1l, q2l, q3l, alpha, world);
        auto Tr = S2::template CGC<PlainLib>(q1r, q2r, q3r, alpha, world);
        return PlainLib::tensorProd(Tl, Tr);
    }

    static Scalar coeff_turn(const qType& ql, const qType& qr, const qType& qf)
    {
        auto [ql1, ql2] = disjoin<S1_::Nq, S2_::Nq>(ql);
        auto [qr1, qr2] = disjoin<S1_::Nq, S2_::Nq>(qr);
        auto [qf1, qf2] = disjoin<S1_::Nq, S2_::Nq>(qf);
        return S1::coeff_turn(ql1, qr1, qf1) * S2::coeff_turn(ql2, qr2, qf2);
    }

    inline static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3, const qType& q4, const qType& q5, const qType& q6);
    inline static Scalar coeff_recouple(const qType& q1, const qType& q2, const qType& q3, const qType& Q, const qType& Q12, const qType& Q23)
    {
        auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
        auto [q2l, q2r] = disjoin<S1_::Nq, S2_::Nq>(q2);
        auto [q3l, q3r] = disjoin<S1_::Nq, S2_::Nq>(q3);
        auto [Ql, Qr] = disjoin<S1_::Nq, S2_::Nq>(Q);
        auto [Q12l, Q12r] = disjoin<S1_::Nq, S2_::Nq>(Q12);
        auto [Q23l, Q23r] = disjoin<S1_::Nq, S2_::Nq>(Q23);

        Scalar out = S1_::coeff_recouple(q1l, q2l, q3l, Ql, Q12l, Q23l) * S2_::coeff_recouple(q1r, q2r, q3r, Qr, Q12r, Q23r);
        return out;
    }

    static Scalar coeff_swap(const qType& ql, const qType& qr, const qType& qf)
    {
        auto [ql1, ql2] = disjoin<S1_::Nq, S2_::Nq>(ql);
        auto [qr1, qr2] = disjoin<S1_::Nq, S2_::Nq>(qr);
        auto [qf1, qf2] = disjoin<S1_::Nq, S2_::Nq>(qf);
        return S1::coeff_swap(ql1, qr1, qf1) * S2::coeff_swap(ql2, qr2, qf2);
    };

    inline static Scalar coeff_9j(const qType& q1,
                                  const qType& q2,
                                  const qType& q3,
                                  const qType& q4,
                                  const qType& q5,
                                  const qType& q6,
                                  const qType& q7,
                                  const qType& q8,
                                  const qType& q9);
    ///@}
    static bool triangle(const qType& q1, const qType& q2, const qType& q3);
};

template <typename S1_, typename S2_>
std::vector<typename S1xS2<S1_, S2_>::qType> S1xS2<S1_, S2_>::basis_combine(const qType& ql, const qType& qr)
{
    auto [ql1, ql2] = disjoin<S1_::Nq, S2_::Nq>(ql);
    auto [qr1, qr2] = disjoin<S1_::Nq, S2_::Nq>(qr);
    std::vector<typename S1_::qType> firstSym = S1_::basis_combine(ql1, qr1);
    std::vector<typename S2_::qType> secondSym = S2_::basis_combine(ql2, qr2);

    std::vector<qType> vout;
    for(const auto& q1 : firstSym)
        for(const auto& q2 : secondSym) { vout.push_back(join(q1, q2)); }
    return vout;
}

template <typename S1_, typename S2_>
typename S1_::Scalar S1xS2<S1_, S2_>::coeff_dot(const qType& q1)
{
    auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
    Scalar out = S1_::coeff_dot(q1l) * S2_::coeff_dot(q1r);
    return out;
}

template <typename S1_, typename S2_>
typename S1_::Scalar S1xS2<S1_, S2_>::coeff_rightOrtho(const qType& q1, const qType& q2)
{
    auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
    auto [q2l, q2r] = disjoin<S1_::Nq, S2_::Nq>(q2);
    Scalar out = S1_::coeff_rightOrtho(q1l, q2l) * S2_::coeff_rightOrtho(q1r, q2r);
    return out;
}

// template<typename S1_, typename S2_>
// typename S1_::Scalar S1xS2<S1_,S2_>::
// coeff_3j(const qType& q1, const qType& q2, const qType& q3,
// 		 int        q1_z, int        q2_z,        int q3_z)
// {
// 	Scalar out = S1_::coeff_3j({q1[0]}, {q2[0]}, {q3[0]}, q1_z, q2_z, q3_z) * S2_::coeff_3j({q1[1]}, {q2[1]}, {q3[1]}, q1_z, q2_z, q3_z);
// 	return out;
// }

template <typename S1_, typename S2_>
typename S1_::Scalar S1xS2<S1_, S2_>::coeff_6j(const qType& q1, const qType& q2, const qType& q3, const qType& q4, const qType& q5, const qType& q6)
{
    auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
    auto [q2l, q2r] = disjoin<S1_::Nq, S2_::Nq>(q2);
    auto [q3l, q3r] = disjoin<S1_::Nq, S2_::Nq>(q3);
    auto [q4l, q4r] = disjoin<S1_::Nq, S2_::Nq>(q4);
    auto [q5l, q5r] = disjoin<S1_::Nq, S2_::Nq>(q5);
    auto [q6l, q6r] = disjoin<S1_::Nq, S2_::Nq>(q6);

    Scalar out = S1_::coeff_6j(q1l, q2l, q3l, q4l, q5l, q6l) * S2_::coeff_6j(q1r, q2r, q3r, q4r, q5r, q6r);
    return out;
}

template <typename S1_, typename S2_>
typename S1_::Scalar S1xS2<S1_, S2_>::coeff_9j(const qType& q1,
                                               const qType& q2,
                                               const qType& q3,
                                               const qType& q4,
                                               const qType& q5,
                                               const qType& q6,
                                               const qType& q7,
                                               const qType& q8,
                                               const qType& q9)
{
    auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
    auto [q2l, q2r] = disjoin<S1_::Nq, S2_::Nq>(q2);
    auto [q3l, q3r] = disjoin<S1_::Nq, S2_::Nq>(q3);
    auto [q4l, q4r] = disjoin<S1_::Nq, S2_::Nq>(q4);
    auto [q5l, q5r] = disjoin<S1_::Nq, S2_::Nq>(q5);
    auto [q6l, q6r] = disjoin<S1_::Nq, S2_::Nq>(q6);
    auto [q7l, q7r] = disjoin<S1_::Nq, S2_::Nq>(q7);
    auto [q8l, q8r] = disjoin<S1_::Nq, S2_::Nq>(q8);
    auto [q9l, q9r] = disjoin<S1_::Nq, S2_::Nq>(q9);

    Scalar out = S1_::coeff_9j(q1l, q2l, q3l, q4l, q5l, q6l, q7l, q8l, q9l) * S2_::coeff_9j(q1r, q2r, q3r, q4r, q5r, q6r, q7r, q8r, q9r);
    return out;
}

template <typename S1_, typename S2_>
bool S1xS2<S1_, S2_>::triangle(const qType& q1, const qType& q2, const qType& q3)
{
    auto [q1l, q1r] = disjoin<S1_::Nq, S2_::Nq>(q1);
    auto [q2l, q2r] = disjoin<S1_::Nq, S2_::Nq>(q2);
    auto [q3l, q3r] = disjoin<S1_::Nq, S2_::Nq>(q3);
    return (S1_::triangle(q1l, q2l, q3l) and S2_::triangle(q1r, q2r, q3r));
}

} // namespace Xped::Sym

#endif // end XPED_S1xS2_H_
