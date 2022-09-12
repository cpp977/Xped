#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/Symmetry/SU2Wrappers.hpp"
#include "Xped/Util/Random.hpp"

#include "Xped/Symmetry/SU2.hpp"

namespace Xped::Sym {

template <typename Kind, typename Scalar_>
typename SU2<Kind, Scalar_>::qType SU2<Kind, Scalar_>::random_q()
{
    int qval = random::threadSafeRandUniform<int, int>(1, 20);
    qType out = {qval};
    return out;
}

template <typename Kind, typename Scalar_>
std::vector<typename SU2<Kind, Scalar_>::qType> SU2<Kind, Scalar_>::basis_combine(const qType& ql, const qType& qr)
{
    int qmin = std::abs(ql[0] - qr[0]) + 1;
    int qmax = std::abs(ql[0] + qr[0]) - 1;
    std::vector<qType> vout((qmax - qmin) / 2 + 1);
    std::size_t count = 0;
    for(int i = qmin; i <= qmax; i += 2) { vout[count++] = {i}; }
    return vout;
}

template <typename Kind, typename Scalar_>
Scalar_ SU2<Kind, Scalar_>::coeff_dot(const qType& q1)
{
    return static_cast<Scalar>(q1[0]);
}

template <typename Kind, typename Scalar_>
Scalar_ SU2<Kind, Scalar_>::coeff_FS(const qType& q1)
{
    return (q1[0] % 2 == 0) ? -1. : 1.;
}

template <typename Kind, typename Scalar_>
template <typename PlainLib>
typename PlainLib::template TType<Scalar_, 2> SU2<Kind, Scalar_>::one_j_tensor(const qType& q1, mpi::XpedWorld& world)
{
    typedef typename PlainLib::Indextype IndexType;

    // typename TensorInterface<TensorLib>::template TType<Scalar_,2> out(degeneracy(q1), degeneracy(q1));
    auto out = PlainLib::template construct<Scalar>(std::array<IndexType, 2>{degeneracy(q1), degeneracy(q1)}, world);
    PlainLib::template setZero<Scalar, 2>(out);

    for(IndexType i = 0; i < degeneracy(q1); i++)
        for(IndexType j = 0; j < degeneracy(q1); j++) {
            int im = -(2 * i - (q1[0] - 1));
            int jm = -(2 * j - (q1[0] - 1));
            if(im == -jm) {
                Scalar value = phase<Scalar>((q1[0] - im) / 2);
                PlainLib::template setVal<Scalar, 2>(out, {{i, j}}, value);
            }
        }
    return out;
}

template <typename Kind, typename Scalar_>
Scalar_ SU2<Kind, Scalar_>::coeff_3j(const qType& q1, const qType& q2, const qType& q3, int q1_z, int q2_z, int q3_z)
{
    Scalar out = coupling_3j(q1[0], q2[0], q3[0], q1_z, q2_z, q3_z);
    return out;
}

template <typename Kind, typename Scalar_>
template <typename PlainLib>
typename PlainLib::template TType<Scalar_, 3>
SU2<Kind, Scalar_>::CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t, mpi::XpedWorld& world)
{
    typedef typename PlainLib::Indextype IndexType;
    // typename TensorInterface<TensorLib>::template TType<Scalar_,3> out(degeneracy(q1),degeneracy(q2),degeneracy(q3));
    auto out = PlainLib::template construct<Scalar>(std::array<IndexType, 3>{degeneracy(q1), degeneracy(q2), degeneracy(q3)}, world);
    for(int i_q1m = 0; i_q1m < degeneracy(q1); i_q1m++)
        for(int i_q2m = 0; i_q2m < degeneracy(q2); i_q2m++)
            for(int i_q3m = 0; i_q3m < degeneracy(q3); i_q3m++) {
                int q1_2m = -(2 * i_q1m - (q1[0] - 1));
                int q2_2m = -(2 * i_q2m - (q2[0] - 1));
                int q3_2m = -(2 * i_q3m - (q3[0] - 1));
                Scalar value = coupling_3j(q1[0], q2[0], q3[0], q1_2m, q2_2m, -q3_2m) * phase<Scalar>((q1[0] - q2[0] + q3_2m) / 2) * sqrt(q3[0]);
                std::array<IndexType, 3> index = {i_q1m, i_q2m, i_q3m};
                PlainLib::template setVal<Scalar, 3>(out, index, value);
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

} // namespace Xped::Sym
