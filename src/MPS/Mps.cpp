#include <numeric>

#include "Xped/MPS/Mps.hpp"

#include "spdlog/spdlog.h"

#include "Xped/Core/AdjointOp.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar_, typename Symmetry_>
Mps<Scalar_, Symmetry_>::Mps(std::size_t L)
    : N_sites(L)
{
    resizeArrays();
}

template <typename Scalar_, typename Symmetry_>
Mps<Scalar_, Symmetry_>::Mps(std::size_t L,
                             const std::vector<Qbasis<Symmetry, 1>>& qloc_in,
                             const qType& Qtarget_in,
                             std::size_t Mmax_in,
                             std::size_t Nqmax_in,
                             std::size_t seed)
    : N_sites(L)
{
    static thread_local std::mt19937 engine(std::random_device{}());
    engine.seed(seed);
    Qtarget.push_back(Qtarget_in);
    resizeArrays();
    qloc = qloc_in;
    gen_maxBasis();
    gen_auxBasis(Mmax_in, Nqmax_in);
    for(size_t l = 0; l < N_sites; l++) {
        A.Ac[l] = Tensor<Scalar, 2, 1, Symmetry>({{inBasis(l), locBasis(l)}}, {{outBasis(l)}});
        A.Ac[l].setRandom(engine);
    }
}

template <typename Scalar_, typename Symmetry_>
void Mps<Scalar_, Symmetry_>::resizeArrays()
{
    A.Ac.resize(N_sites);
    qaux.resize(N_sites + 1);
    maxBasis.resize(N_sites + 1);
    qloc.resize(N_sites);
    SVspec.resize(N_sites);
    truncWeight.resize(N_sites);
    S.resize(N_sites);
}

template <typename Scalar_, typename Symmetry_>
void Mps<Scalar_, Symmetry_>::gen_maxBasis()
{
    maxBasis[0].push_back(Symmetry::qvacuum(), 1ul);
    for(const auto& q : Qtarget) { maxBasis[N_sites].push_back(q, 1ul); }

    std::vector<Qbasis<Symmetry, 1>> from_left(N_sites - 1); // l<-->bond(l,l+1)
    std::vector<Qbasis<Symmetry, 1>> from_right(N_sites - 1); // l<-->bond(l,l+1)

    from_left[0] = maxBasis[0].combine(qloc[0]).forgetHistory();
    for(std::size_t l = 1; l < N_sites - 1; l++) { from_left[l] = from_left[l - 1].combine(qloc[l]).forgetHistory(); }

    from_right[N_sites - 2] = maxBasis[N_sites].combine(qloc[N_sites - 1]).forgetHistory();
    for(std::size_t l = N_sites - 2; l > 0; l--) { from_right[l - 1] = from_right[l].combine(qloc[l]).forgetHistory(); }

    for(std::size_t l = 0; l < N_sites - 1; l++) {
        maxBasis[l + 1] = from_left[l].intersection(from_right[l]);
        // std::vector<qType> v_intersection;
        // std::set_intersection(from_left[l].cbegin(), from_left[l].cend(), from_right[l].cbegin(), from_right[l].cend(),
        // std::back_inserter(v_intersection)); for (const auto q: v_intersection) {qranges[l+1].insert(q);}
    }
}

template <typename Scalar_, typename Symmetry_>
void Mps<Scalar_, Symmetry_>::gen_auxBasis(const std::size_t Minit, const std::size_t Qinit)
{
    assert(Minit >= Qinit and "Minit is too small as compared to Qinit");

    // take the first Qinit quantum numbers from qs which have the smallerst distance to mean
    auto take_first_elems = [this, Qinit](const std::vector<qType>& qs, std::array<double, Nq> mean, const size_t& loc) -> std::unordered_set<qType> {
        std::vector<qType> vec = qs;
        if(vec.size() > Qinit) {
            // sort the vector first according to the distance to mean
            sort(vec.begin(), vec.end(), [mean, loc, this](qType q1, qType q2) {
                std::vector<double> dist_q1(Nq);
                std::vector<double> dist_q2(Nq);
                for(size_t q = 0; q < Nq; q++) {
                    const auto Qmin = maxBasis[loc].minQ();
                    const auto Qmax = maxBasis[loc].maxQ();
                    // const auto [Qmin, Qmax] = std::minmax_element(qranges[loc].begin(), qranges[loc].end());
                    double Delta = (Qmax)[q] - (Qmin)[q];
                    dist_q1[q] = (q1[q] - mean[q]) / Delta;
                    dist_q2[q] = (q2[q] - mean[q]) / Delta;
                }
                return (std::sqrt(std::inner_product(dist_q1.begin(), dist_q1.end(), dist_q1.begin(), 0.)) <
                        std::sqrt(std::inner_product(dist_q2.begin(), dist_q2.end(), dist_q2.begin(), 0.)))
                           ? true
                           : false;
            });

            vec.erase(vec.begin() + Qinit, vec.end());
        }
        std::unordered_set<qType> out;
        for(const auto& q : vec) { out.insert(q); }
        return out;
    };

    for(std::size_t l = 0; l < N_sites + 1; l++) {
        std::array<double, Nq> mean;
        for(size_t q = 0; q < Nq; q++) { mean[q] = Qtarget[0][q] * l * 1. / N_sites; }
        auto qs = maxBasis[l].qs();
        auto kept_qs = take_first_elems(qs, mean, l);

        qaux[l] = maxBasis[l];
        qaux[l].truncate(kept_qs, Minit);
    }

    // qaux[0].push_back(Symmetry::qvacuum(), 1);
    // for (const auto& q:Qtarget) {
    //         qaux[N_sites].push_back(q,1);
    // }

    // std::vector<Qbasis<Symmetry,1> > fromLeft(N_sites);
    // fromLeft[0] = qaux[0];

    // for (std::size_t l=1; l<N_sites; l++) {
    //         fromLeft[l] = fromLeft[l-1].combine(qloc[l-1]).forgetHistory();
    //         auto qs = fromLeft[l].qs();
    //         for (const auto& q:qs) {
    //                 if (auto it=qranges[l].find(q); it == qranges[l].end()) {fromLeft[l].remove(q);}
    //         }

    //         std::array<double,Nq> mean;
    // 	for (size_t q=0; q<Nq; q++)
    // 	{
    // 		mean[q] = Qtarget[0][q]*l*1./N_sites;
    // 	}
    //         qs = fromLeft[l].qs();
    //         auto kept_qs = take_first_elems(qs,mean,l);
    //         for (const auto& q:qs) {
    //                 if (auto it=std::find(kept_qs.begin(), kept_qs.end(), q); it == kept_qs.end()) {
    //                         fromLeft[l].remove(q);
    //                 }
    //         }
    //         fromLeft[l].truncate(Minit);
    // }

    // std::vector<Qbasis<Symmetry,1> > fromRight(N_sites);
    // fromRight[N_sites-1] = qaux[N_sites];
    // for (std::size_t l=N_sites-1; l>0; l--) {
    //         fromRight[l-1] = fromRight[l].combine(qloc[l]).forgetHistory();
    //         auto qs = fromRight[l-1].qs();
    //         for (const auto& q:qs) {
    //                 if (auto it=qranges[l].find(q); it == qranges[l].end()) {fromRight[l-1].remove(q);}
    //         }

    //         std::array<double,Nq> mean;
    // 	for (size_t q=0; q<Nq; q++)
    // 	{
    // 		mean[q] = Qtarget[0][q]*l*1./N_sites;
    // 	}
    //         qs = fromRight[l-1].qs();
    //         auto kept_qs = take_first_elems(qs,mean,l);
    //         for (const auto& q:qs) {
    //                 if (auto it=std::find(kept_qs.begin(), kept_qs.end(), q); it == kept_qs.end()) {
    //                         fromRight[l-1].remove(q);
    //                 }
    //         }
    //         fromRight[l-1].truncate(Minit);
    // }
    // for (std::size_t l=1; l<N_sites; l++) {
    //         qaux[l] = fromLeft[l].intersection(fromRight[l-1]);
    // }
}

template <typename Scalar_, typename Symmetry_>
void Mps<Scalar_, Symmetry_>::leftSweepStep(const std::size_t loc, const DMRG::BROOM& broom, const bool DISCARD_U)
{
    bool RETURN_SPEC = false;
    if(loc != 0) { RETURN_SPEC = true; }
    double entropy;
    std::map<qType, VectorType> SVspec_;
    Tensor<Scalar, 1, 1, Symmetry> left;
    Tensor<Scalar, 2, 1, Symmetry> right;
    auto [U, Sigma, Vdag] = A.Ac[loc].template permute<+1, 0, 2, 1>().tSVD(max_Nsv, eps_svd, truncWeight[loc], entropy, SVspec_, false, RETURN_SPEC);
    // std::cout << Sigma << std::endl;
    if(loc != this->N_sites - 1) {
        S[loc] = entropy;
        SVspec[loc] = SVspec_;
    }
    left = U * Sigma;
    right = Vdag.template permute<-1, 0, 2, 1>();

    A.Ac[loc] = right;
    if(loc != 0 and DISCARD_U == false) { A.Ac[loc - 1] = A.Ac[loc - 1] * left; }
}

template <typename Scalar_, typename Symmetry_>
void Mps<Scalar_, Symmetry_>::rightSweepStep(const std::size_t loc, const DMRG::BROOM& broom, const bool DISCARD_V)
{
    SPDLOG_INFO("Entering Mps::rightSweepStep()");
    bool RETURN_SPEC = false;
    if(loc != N_sites - 1) { RETURN_SPEC = true; }
    double entropy;
    std::map<qType, VectorType> SVspec_;
    Tensor<Scalar, 2, 1, Symmetry> left;
    Tensor<Scalar, 1, 1, Symmetry> right;
    SPDLOG_INFO("Set up pre stuff.");
    auto [U, Sigma, Vdag] = A.Ac[loc].tSVD(max_Nsv, eps_svd, truncWeight[loc], entropy, SVspec_, false, RETURN_SPEC);
    SPDLOG_INFO("Applied the svd.");
    // std::cout << Sigma << std::endl;
    if(loc != this->N_sites - 1) {
        S[loc] = entropy;
        SVspec[loc] = SVspec_;
    }
    left = U;
    right = Sigma * Vdag;

    A.Ac[loc] = left;
    if(loc != this->N_sites - 1 and DISCARD_V == false) {
        A.Ac[loc + 1] = (right * (A.Ac[loc + 1].template permute<+1, 0, 1, 2>())).template permute<-1, 0, 1, 2>();
    }
    SPDLOG_INFO("Leaving Mps::rightSweepStep()");
}

} // namespace Xped

#if __has_include("Mps.gen.cpp")
#    include "Mps.gen.cpp"
#endif
