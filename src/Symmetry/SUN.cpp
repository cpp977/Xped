#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/Symmetry/SU2Wrappers.hpp"
#include "Xped/Util/Random.hpp"

#include "Xped/Symmetry/SUN.hpp"

namespace Xped::Sym {

template <std::size_t N, typename Kind, typename Scalar_>
typename SUN<N, Kind, Scalar_>::qType SUN<N, Kind, Scalar_>::random_q()
{
    qType out(N);
    for(std::size_t i = 0; i < N - 1; ++i) { out(i) = random::threadSafeRandUniform<int, int>(1, 20); }
    out(N - 1) = 0;
    return out;
}

template <std::size_t N, typename Kind, typename Scalar_>
std::vector<typename SUN<N, Kind, Scalar_>::qType> SUN<N, Kind, Scalar_>::basis_combine(const qType& ql, const qType& qr)
{
    std::vector<qType> vout;
    clebsch::decomposition comp(ql, qr);
    for(std::size_t i = 0; i < comp.size(), ++i) { vout.push_back(comp(i)); }
    return vout;
}

template <std::size_t N, typename Kind, typename Scalar_>
Scalar_ SUN<N, Kind, Scalar_>::coeff_dot(const qType& q1)
{
    return static_cast<Scalar>(q1.dimension());
}

template <std::size_t N, typename Kind, typename Scalar_>
template <typename PlainLib>
typename PlainLib::template TType<Scalar_, 2> SUN<N, Kind, Scalar_>::one_j_tensor(const qType& q1, mpi::XpedWorld& world)
{
    typedef typename PlainLib::Indextype IndexType;

    auto tmp = CGC(q1, conj(q1), vacuum(), 0);
    auto out = PlainLib::reshape<Scalar, 4, 2>(tmp, std::array<IndexType, 2>{degeneracy(q1), degeneracy(q1)});
    return out;
}

template <std::size_t N, typename Kind, typename Scalar_>
template <typename PlainLib>
typename PlainLib::template TType<Scalar_, 3>
SUN<N, Kind, Scalar_>::CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t multiplicity, mpi::XpedWorld& world)
{
    typedef typename PlainLib::Indextype IndexType;
    clebsch::coefficients coeffs(q3, q1, q2);
    // typename TensorInterface<TensorLib>::template TType<Scalar_,3> out(degeneracy(q1),degeneracy(q2),degeneracy(q3));
    auto out = PlainLib::template construct<Scalar>(std::array<IndexType, 3>{degeneracy(q1), degeneracy(q2), degeneracy(q3)}, world);
    for(int i_q1 = 0; i_q1 < degeneracy(q1); i_q1++)
        for(int i_q2 = 0; i_q2 < degeneracy(q2); i_q2++)
            for(int i_q3 = 0; i_q3 < degeneracy(q3); i_q3++) {
                Scalar value = coeffs(i_q1, iq2, static_cast<int>(multiplicity), i_q3);
                std::array<IndexType, 3> index = {i_q1, i_q2, i_q3};
                PlainLib::template setVal<Scalar, 3>(out, index, value);
            }
    return out;
}

template <std::size_t N, typename Kind, typename Scalar_>
template <typename PlainLib>
typename PlainLib::template TType<Scalar_, 4> SUN<N, Kind, Scalar_>::CGC(const qType& q1, const qType& q2, const qType& q3, mpi::XpedWorld& world)
{
    typedef typename PlainLib::Indextype IndexType;
    clebsch::coefficients coeffs(q3, q1, q2);
    // typename TensorInterface<TensorLib>::template TType<Scalar_,3> out(degeneracy(q1),degeneracy(q2),degeneracy(q3));
    auto out = PlainLib::template construct<Scalar>(std::array<IndexType, 4>{degeneracy(q2), degeneracy(q3), degeneracy(q1), q1.multiplicity}, world);
    for(int i_q1 = 0; i_q1 < degeneracy(q2); i_q1++) {
        for(int i_q2 = 0; i_q2 < degeneracy(q3); i_q2++) {
            for(int i_q3 = 0; i_q3 < degeneracy(q1); i_q3++) {
                for(int m = 0; m < q1.multiplicity; ++m) {
                    Scalar value = coeffs(i_q1, iq2, m, i_q3);
                    std::array<IndexType, 4> index = {i_q1, i_q2, i_q3, m};
                    PlainLib::template setVal<Scalar, 4>(out, index, value);
                }
            }
        }
    }
    return out;
}

template <std::size_t N, typename Kind, typename Scalar_>
typename PlainInterface::TType<Scalar, 4>
SUN<N, Kind, Scalar_>::coeff_recouple(const qType& q1, const qType& q2, const qType& q3, const qType& Q, const qType& Q12, const qType& Q23)
{

    int N12 = multiplicity(q1, q2, Q12);
    int N12_3 = multiplicity(Q12, q3, Q);
    int N23 = multiplicity(q2, q3, Q23);
    int N1_23 = multiplicity(q1, Q23, Q);

    if(N12 == 0 or N12_3 == 0 or N23 == 0 or N1_23 == 0) {
        auto res PlainInterface::construct(std::array{N12, N12_3, N23, N1_23});
        PlainInterface::setZero(res);
        return res;
    }

    auto X12 = CGC(q1, q2, Q12);
    auto X12_3 = PlainInterface::chip(CGC(Q12, q3, Q), 0, 2);
    auto X23 = CGC(q2, q3, Q23);
    auto X1_23 = PlainInterface::chip(CGC(q1, Q23, Q), 0, 2);
    auto res = PlainInterface::contract<Scalar, 3, 5, 0, 2, 1, 0>(
        X1_23, PlainInterface::contract<Scalar, 4, 5, 0, 1, 1, 3>(X23, PlainInterface::contract<Scalar, 4, 3, 2, 0>(X12, X12_3)));
    return res;
}

template <std::size_t N, typename Kind, typename Scalar_>
typename PlainInterface::TType<Scalar, 2> SUN<N, Kind, Scalar_>::coeff_swap(const qType& ql, const qType& qr, const qType& qf)
{
    int N = multiplicity(ql, qr, qf);

    if(N == 0) {
        auto res PlainInterface::construct(std::array{N, N});
        PlainInterface::setZero(res);
        return res;
    }

    auto Xlr = PlainInterface::chip(CGC(ql, qr, qf), 0, 2);
    auto Xrl = PlainInterface::chip(CGC(qr, ql, qf), 0, 2);
    auto res = PlainInterface::shuffle<Scalar, 2, 1, 0>(PlainInterface::contract<Scalar, 3, 3, 0, 1, 1, 0>(Xlr, Xrl));
    return res;
}

template <std::size_t N, typename Kind, typename Scalar_>
bool SUN<N, Kind, Scalar_>::triangle(const qType& q1, const qType& q2, const qType& q3)
{
    return multiplicity(q1, q2, q3) > 0 ? true : false;
}

} // namespace Xped::Sym
