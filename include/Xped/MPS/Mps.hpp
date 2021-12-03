#ifndef MPS_H_
#define MPS_H_

#include "Xped/Core/ScalarTraits.hpp"
#include "Xped/Core/Xped.hpp"

template <typename TL, typename TR, typename TC>
struct GaugeTriple
{
    TL Al;
    TR Ar;
    TC Ac;
};

namespace DMRG {
enum class BROOM
{
    SVD = 0,
    QR = 1
};
}

template <typename Scalar_, typename Symmetry_>
class Mps
{
public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;
    typedef typename ScalarTraits<Scalar>::Real RealScalar;
    typedef typename Symmetry::qType qType;
    typedef Xped<Scalar, 2, 1, Symmetry> ALType;
    typedef typename ALType::TensorType TensorType;
    typedef typename ALType::MatrixType MatrixType;
    typedef typename ALType::VectorType VectorType;
    typedef Xped<Scalar, 1, 2, Symmetry> ARType;

    constexpr static std::size_t Nq = Symmetry::Nq;

    Mps() {}
    Mps(std::size_t L);
    //     : N_sites(L)
    // {
    //     resizeArrays();
    // }
    Mps(std::size_t L,
        const std::vector<Qbasis<Symmetry, 1>>& qloc_in,
        const qType& Qtarget_in = Symmetry::qvacuum(),
        std::size_t Mmax_in = 10,
        std::size_t Nqmax_in = 10);
    //     : N_sites(L)
    // {
    //     Qtarget.push_back(Qtarget_in);
    //     resizeArrays();
    //     qloc = qloc_in;
    //     gen_maxBasis();
    //     gen_auxBasis(Mmax_in, Nqmax_in);
    //     for(size_t l = 0; l < N_sites; l++) {
    //         A.Ac[l] = Xped<Scalar, 2, 1, Symmetry>({{inBasis(l), locBasis(l)}}, {{outBasis(l)}});
    //         A.Ac[l].setRandom();
    //     }
    // }

    std::size_t length() const { return N_sites; }

    Qbasis<Symmetry, 1> inBasis(std::size_t l) const
    {
        assert(l < N_sites);
        return qaux[l];
    }
    Qbasis<Symmetry, 1> outBasis(std::size_t l) const
    {
        assert(l < N_sites);
        return qaux[l + 1];
    }
    Qbasis<Symmetry, 1> locBasis(std::size_t l) const
    {
        assert(l < N_sites);
        return qloc[l];
    }
    Qbasis<Symmetry, 1> auxBasis(std::size_t l) const
    {
        assert(l < N_sites + 1);
        return qaux[l];
    }

    void leftSweepStep(const std::size_t loc, const DMRG::BROOM& broom, const bool DISCARD_U = false);
    void rightSweepStep(const std::size_t loc, const DMRG::BROOM& broom, const bool DISCARD_V = false);

    // private:
    std::size_t N_sites;
    std::vector<qType> Qtarget;

    GaugeTriple<std::vector<ALType>, std::vector<ARType>, std::vector<ALType>> A; // A[Mps::GAUGE::L/R/C][l]

    std::vector<Qbasis<Symmetry, 1>> qaux;
    std::vector<Qbasis<Symmetry, 1>> qloc;

    // std::vector<std::unordered_map<qType, std::size_t> > qranges; //set of possible values for quantum numbers at site l
    std::vector<Qbasis<Symmetry, 1>> maxBasis; // set of possible values for quantum numbers at site l

    std::size_t max_Nsv = 10000, min_Nsv = 0;
    RealScalar eps_svd = 1.e-10;
    std::vector<std::map<qType, VectorType>> SVspec;
    /**truncated weight*/
    std::vector<RealScalar> truncWeight;

    /**entropy*/
    std::vector<RealScalar> S;

    void gen_maxBasis();

    void gen_auxBasis(const std::size_t Mmax, const std::size_t Nqmax);

    void resizeArrays();
    // {
    //     A.Ac.resize(N_sites);
    //     qaux.resize(N_sites + 1);
    //     maxBasis.resize(N_sites + 1);
    //     qloc.resize(N_sites);
    //     SVspec.resize(N_sites);
    //     truncWeight.resize(N_sites);
    //     S.resize(N_sites);
    // }
};

#ifndef XPED_COMPILED_LIB
#    include "MPS/Mps.cpp"
#endif

#endif
