#ifndef SU2_H_
#define SU2_H_

/// \cond
#include <array>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <boost/rational.hpp>
/// \endcond

#include "SU2Wrappers.hpp"
#include "SymBase.hpp"
#include "functions.hpp"
#include "interfaces/tensor_traits.hpp"
#include "qarray.hpp"
#include "util/Random.hpp"

namespace Sym {

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

    static constexpr bool NON_ABELIAN = true;
    static constexpr bool HAS_MULTIPLICITIES = false;
    static constexpr bool ABELIAN = false;
    static constexpr bool IS_TRIVIAL = false;
    static constexpr bool IS_MODULAR = false;
    static constexpr int MOD_N = 0;

    static constexpr bool IS_CHARGE_SU2()
    {
        if constexpr(SU2<Kind, Scalar>::kind()[0] == KIND::T) { return true; }
        return false;
    }
    static constexpr bool IS_SPIN_SU2()
    {
        if constexpr(SU2<Kind, Scalar>::kind()[0] == KIND::S) { return true; }
        return false;
    }

    static constexpr bool IS_SPIN_U1() { return false; }

    static constexpr bool NO_SPIN_SYM()
    {
        if(SU2<Kind, Scalar>::kind()[0] != KIND::S) { return true; }
        return false;
    }
    static constexpr bool NO_CHARGE_SYM()
    {
        if(SU2<Kind, Scalar>::kind()[0] != KIND::T) { return true; }
        return false;
    }

    typedef qarray<Nq> qType;

    SU2(){};

    inline static std::string name() { return "SU2"; }
    inline static constexpr std::array<KIND, Nq> kind() { return {Kind::name}; }

    inline static constexpr qType qvacuum() { return {1}; }
    inline static constexpr std::array<qType, 1> lowest_qs() { return std::array<qType, 1>{{qarray<1>(std::array<int, 1>{{2}})}}; }

    inline static qType conj(const qType& q) { return q; }
    inline static int degeneracy(const qType& q) { return q[0]; }

    inline static qType random_q()
    {
        int qval = util::random::threadSafeRandUniform<int, int>(1, 20, false);
        qType out = {qval};
        return out;
    }

    /**
     * Calculate the irreps of the tensor product of \p ql and \p qr.
     */
    static std::vector<qType> basis_combine(const qType& ql, const qType& qr);

    std::size_t multiplicity(const qType& q1, const qType& q2, const qType& q3) { return triangle(q1, q2, q3) ? 1ul : 0ul; }
    ///@{
    /**
     * Various coeffecients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
     */
    static Scalar coeff_dot(const qType& q1);

    static Scalar coeff_FS(const qType& q1) { return (q1[0] % 2 == 0) ? -1. : 1.; }

    template <typename TensorLib>
    static typename tensortraits<TensorLib>::template Ttype<Scalar_, 2> one_j_tensor(const qType& q1);

    static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3, int q1_z, int q2_z, int q3_z);

    static Scalar coeff_turn(const qType& ql, const qType& qr, const qType& qf)
    {
        return triangle(ql, qr, qf) ? coeff_swap(ql, qr, qf) * std::sqrt(static_cast<Scalar>(qf[0]) / static_cast<Scalar>(ql[0])) : Scalar(0.);
    }

    template <typename TensorLib>
    static typename tensortraits<TensorLib>::template Ttype<Scalar_, 3> CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t);

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
        return triangle(ql, qr, qf) ? phase<Scalar>((ql[0] + qr[0] - qf[0] - 1) / 2) : Scalar(0.);
    }
    ///@}

    static bool triangle(const qType& q1, const qType& q2, const qType& q3);
};

template <typename Kind, typename Scalar_>
std::vector<typename SU2<Kind, Scalar_>::qType> SU2<Kind, Scalar_>::basis_combine(const qType& ql, const qType& qr)
{
    std::vector<qType> vout;
    int qmin = std::abs(ql[0] - qr[0]) + 1;
    int qmax = std::abs(ql[0] + qr[0]) - 1;
    for(int i = qmin; i <= qmax; i += 2) { vout.push_back({i}); }
    return vout;
}

template <typename Kind, typename Scalar_>
Scalar_ SU2<Kind, Scalar_>::coeff_dot(const qType& q1)
{
    Scalar out = static_cast<Scalar>(q1[0]);
    return out;
}

template <typename Kind, typename Scalar_>
template <typename TensorLib>
typename tensortraits<TensorLib>::template Ttype<Scalar_, 2> SU2<Kind, Scalar_>::one_j_tensor(const qType& q1)
{
    typedef typename tensortraits<TensorLib>::Indextype IndexType;
    auto tmp = CGC<TensorLib>(q1, q1, qvacuum(), 0);
    // typename tensortraits<TensorLib>::template Ttype<Scalar_,2> out(degeneracy(q1), degeneracy(q1));
    auto out = tensortraits<TensorLib>::template construct<Scalar>(std::array<IndexType, 2>{degeneracy(q1), degeneracy(q1)});
    tensortraits<TensorLib>::template setZero<Scalar, 2>(out);

    for(IndexType i = 0; i < degeneracy(q1); i++)
        for(IndexType j = 0; j < degeneracy(q1); j++) { out(i, j) = std::sqrt(degeneracy(q1)) * tmp(i, j, 0); }
    return out;
}

template <typename Kind, typename Scalar_>
Scalar_ SU2<Kind, Scalar_>::coeff_3j(const qType& q1, const qType& q2, const qType& q3, int q1_z, int q2_z, int q3_z)
{
    Scalar out = coupling_3j(q1[0], q2[0], q3[0], q1_z, q2_z, q3_z);
    return out;
}

template <typename Kind, typename Scalar_>
template <typename TensorLib>
typename tensortraits<TensorLib>::template Ttype<Scalar_, 3>
SU2<Kind, Scalar_>::CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t)
{
    typedef typename tensortraits<TensorLib>::Indextype IndexType;
    // typename tensortraits<TensorLib>::template Ttype<Scalar_,3> out(degeneracy(q1),degeneracy(q2),degeneracy(q3));
    auto out = tensortraits<TensorLib>::template construct<Scalar>(std::array<IndexType, 3>{degeneracy(q1), degeneracy(q2), degeneracy(q3)});
    for(int i_q1m = 0; i_q1m < degeneracy(q1); i_q1m++)
        for(int i_q2m = 0; i_q2m < degeneracy(q2); i_q2m++)
            for(int i_q3m = 0; i_q3m < degeneracy(q3); i_q3m++) {
                int q1_2m = -(2 * i_q1m - (q1[0] - 1));
                int q2_2m = -(2 * i_q2m - (q2[0] - 1));
                int q3_2m = -(2 * i_q3m - (q3[0] - 1));
                out(i_q1m, i_q2m, i_q3m) =
                    coupling_3j(q1[0], q2[0], q3[0], q1_2m, q2_2m, -q3_2m) * phase<Scalar>((q1[0] - q2[0] + q3_2m) / 2) * sqrt(q3[0]);
            }

    return out;
}

template <typename Kind, typename Scalar_>
Scalar_ SU2<Kind, Scalar_>::coeff_6j(const qType& q1, const qType& q2, const qType& q3, const qType& q4, const qType& q5, const qType& q6)
{
    Scalar out = coupling_6j(q1[0], q2[0], q3[0], q4[0], q5[0], q6[0]);
    return out;
}

template <typename Kind, typename Scalar_>
Scalar_ SU2<Kind, Scalar_>::coeff_9j(const qType& q1,
                                     const qType& q2,
                                     const qType& q3,
                                     const qType& q4,
                                     const qType& q5,
                                     const qType& q6,
                                     const qType& q7,
                                     const qType& q8,
                                     const qType& q9)
{
    Scalar out = coupling_9j(q1[0], q2[0], q3[0], q4[0], q5[0], q6[0], q7[0], q8[0], q9[0]);
    return out;
}

template <typename Kind, typename Scalar_>
bool SU2<Kind, Scalar_>::triangle(const qType& q1, const qType& q2, const qType& q3)
{
    // check the triangle rule for angular momenta, but remark that we use the convention q=2S+1
    if(q3[0] - 1 >= abs(q1[0] - q2[0]) and q3[0] - 1 <= q1[0] + q2[0] - 2) { return true; }
    return false;
}

} // end namespace Sym

#endif
